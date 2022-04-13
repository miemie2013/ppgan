#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import copy
import numpy as np

import logging
import datetime

import paddle
from paddle.distributed import ParallelEnv

from ..datasets.builder import build_dataloader, build_dataset
from ..metrics.fid import FeatureStats
from ..models import PastaGANModel, StyleGANv2ADAModel
from ..models.builder import build_model
from ..utils import training_stats
from ..utils.visual import tensor2img, save_image
from ..utils.filesystem import makedirs, save, load
from ..utils.timer import TimeAverager
from ..utils.profiler import add_profiler_step


class IterLoader:
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 1

    @property
    def epoch(self):
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)


class Trainer:
    """
    # trainer calling logic:
    #
    #                build_model                               ||    model(BaseModel)
    #                     |                                    ||
    #               build_dataloader                           ||    dataloader
    #                     |                                    ||
    #               model.setup_lr_schedulers                  ||    lr_scheduler
    #                     |                                    ||
    #               model.setup_optimizers                     ||    optimizers
    #                     |                                    ||
    #     train loop (model.setup_input + model.train_iter)    ||    train loop
    #                     |                                    ||
    #         print log (model.get_current_losses)             ||
    #                     |                                    ||
    #         save checkpoint (model.nets)                     \/
    """
    def __init__(self, cfg):
        # base config
        self.logger = logging.getLogger(__name__)
        self.cfg = cfg
        self.output_dir = cfg.output_dir
        self.max_eval_steps = cfg.model.get('max_eval_steps', None)

        self.local_rank = ParallelEnv().local_rank  # 咩酱：当前训练进程的编号。
        self.world_size = ParallelEnv().nranks      # 咩酱：参与训练进程的数量，一般也是训练所使用GPU卡的数量。
        self.log_interval = cfg.log_config.interval
        self.visual_interval = cfg.log_config.visiual_interval
        self.weight_interval = cfg.snapshot_config.interval

        self.start_epoch = 1
        self.current_epoch = 1
        self.current_iter = 1
        self.inner_iter = 1
        self.batch_id = 0
        self.global_steps = 0

        # build model
        self.model = build_model(cfg.model)
        self.archi_name = cfg.model.name
        self.is_distributed = ParallelEnv().nranks > 1
        # multiple gpus prepare
        if ParallelEnv().nranks > 1:
            self.distributed_data_parallel()
            if self.archi_name == 'StyleGANv2ADAModel':
                self.distributed_data_parallel_ema()
                # 为了同步统计量.必须在torch.distributed.init_process_group()方法之后调用.
                sync_device = paddle.CUDAPlace(self.local_rank) if self.is_distributed else paddle.CPUPlace()
                training_stats.init_multiprocessing(rank=self.local_rank, sync_device=sync_device)
                # 修改model的配置，虽然这样写有点丑
                self.model.rank = self.local_rank

        # build metrics
        self.metrics = None
        self.is_save_img = True
        validate_cfg = cfg.get('validate', None)
        if validate_cfg and 'metrics' in validate_cfg:
            self.metrics = self.model.setup_metrics(validate_cfg['metrics'])
        if validate_cfg and 'save_img' in validate_cfg:
            self.is_save_img = validate_cfg['save_img']

        self.enable_visualdl = cfg.get('enable_visualdl', False)
        if self.enable_visualdl:
            import visualdl
            self.vdl_logger = visualdl.LogWriter(logdir=cfg.output_dir)

        # evaluate only
        if not cfg.is_train:
            return

        # build train dataloader
        self.train_dataloader = build_dataloader(cfg.dataset.train)
        self.iters_per_epoch = len(self.train_dataloader)

        # build lr scheduler
        # TODO: has a better way?
        if self.archi_name == 'StyleGANv2ADAModel':
            learning_rate_g = cfg.lr_scheduler_G.learning_rate
            learning_rate_d = cfg.lr_scheduler_D.learning_rate
            beta1 = cfg.optimizer.generator.beta1
            beta2 = cfg.optimizer.generator.beta2

            G_reg_interval = cfg.model.G_reg_interval
            D_reg_interval = cfg.model.D_reg_interval

            for name, reg_interval in [('G', G_reg_interval), ('D', D_reg_interval)]:
                if reg_interval is None:
                    if name == 'G':
                        cfg.lr_scheduler_G.learning_rate = learning_rate_g
                    elif name == 'D':
                        cfg.lr_scheduler_D.learning_rate = learning_rate_d
                else:  # Lazy regularization.
                    if name == 'G':
                        mb_ratio = reg_interval / (reg_interval + 1)
                        new_lr = learning_rate_g * mb_ratio
                        new_beta1 = beta1 ** mb_ratio
                        new_beta2 = beta2 ** mb_ratio

                        cfg.lr_scheduler_G.learning_rate = new_lr
                        cfg.optimizer.generator.beta1 = new_beta1
                        cfg.optimizer.generator.beta2 = new_beta2
                    elif name == 'D':
                        mb_ratio = reg_interval / (reg_interval + 1)
                        new_lr = learning_rate_d * mb_ratio
                        new_beta1 = beta1 ** mb_ratio
                        new_beta2 = beta2 ** mb_ratio

                        cfg.lr_scheduler_D.learning_rate = new_lr
                        cfg.optimizer.discriminator.beta1 = new_beta1
                        cfg.optimizer.discriminator.beta2 = new_beta2

            if 'lr_scheduler_G' in cfg and 'iters_per_epoch' in cfg.lr_scheduler_G:
                cfg.lr_scheduler_G.iters_per_epoch = self.iters_per_epoch
            self.lr_schedulers_G = self.model.setup_lr_schedulers(cfg.lr_scheduler_G)

            if 'lr_scheduler_D' in cfg and 'iters_per_epoch' in cfg.lr_scheduler_D:
                cfg.lr_scheduler_D.iters_per_epoch = self.iters_per_epoch
            self.lr_schedulers_D = self.model.setup_lr_schedulers(cfg.lr_scheduler_D)

            # build optimizers
            self.optimizers = self.model.setup_optimizers(self.lr_schedulers_G, self.lr_schedulers_D, cfg.optimizer)
        else:
            if 'lr_scheduler' in cfg and 'iters_per_epoch' in cfg.lr_scheduler:
                cfg.lr_scheduler.iters_per_epoch = self.iters_per_epoch
            self.lr_schedulers = self.model.setup_lr_schedulers(cfg.lr_scheduler)

            # build optimizers
            self.optimizers = self.model.setup_optimizers(self.lr_schedulers,
                                                          cfg.optimizer)

        self.epochs = cfg.get('epochs', None)
        if self.epochs:
            self.total_iters = self.epochs * self.iters_per_epoch
            self.by_epoch = True
        else:
            self.by_epoch = False
            self.total_iters = cfg.get('total_iters', None)
            if self.total_iters is None:
                kimgs = cfg.get('kimgs', None)
                kimgs = kimgs * 1000
                batch_size = self.cfg.dataset.train.batch_size
                self.total_iters = kimgs // batch_size

        if self.by_epoch:
            self.weight_interval *= self.iters_per_epoch

        self.validate_interval = -1
        if cfg.get('validate', None) is not None:
            self.validate_interval = cfg.validate.get('interval', -1)

        self.time_count = {}
        self.best_metric = {}
        self.model.set_total_iter(self.total_iters)
        self.profiler_options = cfg.profiler_options

        # miemie2013: 调试的代码
        # state_dicts222 = paddle.load("../G_temp_256.pdparams")
        # for net_name, net in self.model.nets.items():
        #     if net_name == 'discriminator':
        #         continue
        #     net.set_state_dict(state_dicts222[net_name])
        # self.model.nets['discriminator'].set_state_dict(paddle.load("../D_temp_256.pdparams"))
        # print()

    def distributed_data_parallel(self):
        paddle.distributed.init_parallel_env()
        find_unused_parameters = self.cfg.get('find_unused_parameters', False)
        for net_name, net in self.model.nets.items():
            self.model.nets[net_name] = paddle.DataParallel(
                net, find_unused_parameters=find_unused_parameters)

    def distributed_data_parallel_ema(self):
        find_unused_parameters = self.cfg.get('find_unused_parameters', False)
        for net_name, net in self.model.nets_ema.items():
            self.model.nets_ema[net_name] = paddle.DataParallel(
                net, find_unused_parameters=find_unused_parameters)

    def learning_rate_scheduler_step(self):
        if isinstance(self.model.lr_scheduler, dict):
            for lr_scheduler in self.model.lr_scheduler.values():
                lr_scheduler.step()
        elif isinstance(self.model.lr_scheduler,
                        paddle.optimizer.lr.LRScheduler):
            self.model.lr_scheduler.step()
        else:
            raise ValueError(
                'lr schedulter must be a dict or an instance of LRScheduler')

    def train(self):
        reader_cost_averager = TimeAverager()
        batch_cost_averager = TimeAverager()

        iter_loader = IterLoader(self.train_dataloader)

        # set model.is_train = True
        self.model.setup_train_mode(is_train=True)
        while self.current_iter < (self.total_iters + 1):
            self.current_epoch = iter_loader.epoch
            self.inner_iter = self.current_iter % self.iters_per_epoch

            add_profiler_step(self.profiler_options)

            start_time = step_start_time = time.time()
            data = next(iter_loader)
            reader_cost_averager.record(time.time() - step_start_time)
            # unpack data from dataset and apply preprocessing
            # data input should be dict
            self.model.setup_input(data)
            self.model.train_iter(self.optimizers)

            batch_cost_averager.record(
                time.time() - step_start_time,
                num_samples=self.cfg['dataset']['train'].get('batch_size', 1))

            step_start_time = time.time()

            if self.current_iter % self.log_interval == 0:
                self.data_time = reader_cost_averager.get_average()
                self.step_time = batch_cost_averager.get_average()
                self.ips = batch_cost_averager.get_ips_average()
                self.print_log()

                reader_cost_averager.reset()
                batch_cost_averager.reset()

            if self.current_iter % self.visual_interval == 0 and self.local_rank == 0:
                self.visual('visual_train')

            self.learning_rate_scheduler_step()

            if self.validate_interval > -1 and self.current_iter % self.validate_interval == 0:
                self.test()

            if self.current_iter % self.weight_interval == 0:
                # self.save(self.current_iter, 'weight', keep=-1)
                # self.save(self.current_iter, keep=-1)
                self.save(self.current_iter, keep=1)

            self.current_iter += 1

    def test(self):
        if not hasattr(self, 'test_dataloader'):
            self.test_dataloader = build_dataloader(self.cfg.dataset.test,
                                                    is_train=False)
        iter_loader = IterLoader(self.test_dataloader)
        if self.max_eval_steps is None:
            self.max_eval_steps = len(self.test_dataloader)

        if self.metrics:
            for metric in self.metrics.values():
                metric.reset()

        # set model.is_train = False
        self.model.setup_train_mode(is_train=False)

        for i in range(self.max_eval_steps):
            if self.max_eval_steps < self.log_interval or i % self.log_interval == 0:
                self.logger.info('Test iter: [%d/%d]' %
                                 (i * self.world_size,
                                  self.max_eval_steps * self.world_size))

            data = next(iter_loader)
            self.model.setup_input(data)
            self.model.test_iter(metrics=self.metrics)

            if self.is_save_img:
                visual_results = {}
                current_paths = self.model.get_image_paths()
                current_visuals = self.model.get_current_visuals()

                if len(current_visuals) > 0 and list(
                        current_visuals.values())[0].shape == 4:
                    num_samples = list(current_visuals.values())[0].shape[0]
                else:
                    num_samples = 1

                for j in range(num_samples):
                    if j < len(current_paths):
                        short_path = os.path.basename(current_paths[j])
                        basename = os.path.splitext(short_path)[0]
                    else:
                        basename = '{:04d}_{:04d}'.format(i, j)
                    for k, img_tensor in current_visuals.items():
                        name = '%s_%s' % (basename, k)
                        if len(img_tensor.shape) == 4:
                            visual_results.update({name: img_tensor[j]})
                        else:
                            visual_results.update({name: img_tensor})

                self.visual('visual_test',
                            visual_results=visual_results,
                            step=self.batch_id,
                            is_save_image=True)

        if self.metrics:
            for metric_name, metric in self.metrics.items():
                self.logger.info("Metric {}: {:.4f}".format(
                    metric_name, metric.accumulate()))

    @paddle.no_grad()
    def calc_stylegan2ada_metric(self, inceptionv3_model, dataset_batch_size, batch_size, num_gen, G_kwargs={}):
        cfg_ = self.cfg.dataset.train.copy()
        _ = cfg_.pop('batch_size', 1)
        num_workers = cfg_.pop('num_workers', 0)
        use_shared_memory = cfg_.pop('use_shared_memory', True)
        dataset = build_dataset(cfg_)
        n_dataset = len(dataset)
        return_features = True

        if not hasattr(self, 'train_dataloader'):
            self.cfg.dataset.train.batch_size = dataset_batch_size
            self.test_dataloader = build_dataloader(self.cfg.dataset.train,
                                                    is_train=False)
        iter_loader = IterLoader(self.test_dataloader)
        if self.max_eval_steps is None:
            self.max_eval_steps = len(self.test_dataloader)

        # set model.is_train = False
        self.model.setup_train_mode(is_train=False)

        num_items = len(self.test_dataloader)
        real_stats_kwargs = dict(capture_mean_cov=True,)
        real_stats = FeatureStats(max_items=n_dataset, **real_stats_kwargs)

        log_interval = 1024
        for i in range(self.max_eval_steps):
            n_imgs = i * dataset_batch_size
            if n_dataset < log_interval or n_imgs % log_interval == 0:
                self.logger.info('dataset features: [%d/%d]' % (n_imgs, n_dataset))

            data = next(iter_loader)
            real_image, label, image_gen_c = data
            real_image = paddle.cast(real_image, dtype=paddle.float32)  # BGR格式
            real_features = inceptionv3_model(real_image, return_features=return_features)
            real_stats.append_tensor(real_features, num_gpus=1, rank=0)
        mu_real, sigma_real = real_stats.get_mean_cov()

        batch_gen = min(batch_size, 4)
        assert batch_size % batch_gen == 0

        fake_stats_kwargs = dict(capture_mean_cov=True,)
        fake_stats = FeatureStats(max_items=num_gen, **fake_stats_kwargs)

        from collections import deque
        time_stat = deque(maxlen=20)
        start_time = time.time()
        end_time = time.time()
        num_imgs = num_gen
        start = time.time()
        i = 0

        # Main loop.
        while not fake_stats.is_full():
            images = []
            for _i in range(batch_size // batch_gen):
                z = paddle.randn([batch_gen, self.model.z_dim], dtype=paddle.float32)
                c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_gen)]
                c = paddle.to_tensor(np.stack(c))
                img = self.model.gen_images(z=z, c=c, **G_kwargs)
                img = (img * 127.5 + 128)
                img = paddle.clip(img, 0, 255)
                images.append(img)
            images = paddle.concat(images)  # BGR格式
            if images.shape[1] == 1:
                images = images.tile([1, 3, 1, 1])
            fake_features = inceptionv3_model(images, return_features=return_features)
            fake_stats.append_tensor(fake_features, num_gpus=1, rank=0)

            # 估计剩余时间
            start_time = end_time
            end_time = time.time()
            time_stat.append(end_time - start_time)
            time_cost = np.mean(time_stat)
            eta_sec = (num_imgs - i * batch_size) * time_cost
            eta = str(datetime.timedelta(seconds=int(eta_sec)))
            n_imgs = i * batch_size
            if num_gen < log_interval or n_imgs % log_interval == 0:
                self.logger.info('generator features: [%d/%d], eta=%s.' % (n_imgs, num_gen, eta))

            i += 1
        cost = time.time() - start
        self.logger.info('total time: {0:.6f}s'.format(cost))
        self.logger.info('Speed: %.6fs per image,  %.1f FPS.' % ((cost / num_imgs), (num_imgs / cost)))
        mu_gen, sigma_gen = fake_stats.get_mean_cov()

        m = np.square(mu_gen - mu_real).sum()
        import scipy.linalg
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        fid = float(fid)
        self.logger.info('FID: %.6f' % (fid, ))


    def style_mixing(self, row_seeds, col_seeds, col_styles):
        # set model.is_train = False
        self.model.setup_train_mode(is_train=False)

        all_seeds = list(set(row_seeds + col_seeds))
        all_z = np.stack([np.random.RandomState(seed).randn(self.model.z_dim) for seed in all_seeds])
        all_z = paddle.to_tensor(all_z)
        all_z = paddle.cast(all_z, dtype=paddle.float32)
        data = {
            'z': all_z,
        }
        self.model.setup_input(data)
        self.model.style_mixing(row_seeds, col_seeds, all_seeds, col_styles)

        if self.is_save_img:
            visual_results = {}
            current_paths = self.model.get_image_paths()
            current_visuals = self.model.get_current_visuals()

            if len(current_visuals) > 0 and list(
                    current_visuals.values())[0].shape == 4:
                num_samples = list(current_visuals.values())[0].shape[0]
            else:
                num_samples = 1
            i = 0
            for j in range(num_samples):
                if j < len(current_paths):
                    short_path = os.path.basename(current_paths[j])
                    basename = os.path.splitext(short_path)[0]
                else:
                    basename = '{:04d}_{:04d}'.format(i, j)
                for k, img_tensor in current_visuals.items():
                    name = '%s_%s' % (basename, k)
                    if len(img_tensor.shape) == 4:
                        visual_results.update({name: img_tensor[j]})
                    else:
                        visual_results.update({name: img_tensor})

            self.visual('visual_test',
                        visual_results=visual_results,
                        step=self.batch_id,
                        is_save_image=True)

    def print_log(self):
        losses = self.model.get_current_losses()

        message = ''
        if self.by_epoch:
            message += 'Epoch: %d/%d, iter: %d/%d ' % (
                self.current_epoch, self.epochs, self.inner_iter,
                self.iters_per_epoch)
        else:
            message += 'Iter: %d/%d ' % (self.current_iter, self.total_iters)

        message += f'lr: {self.current_learning_rate:.3e} '

        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
            if self.enable_visualdl:
                self.vdl_logger.add_scalar(k, v, step=self.global_steps)

        if hasattr(self, 'step_time'):
            message += 'batch_cost: %.5f sec ' % self.step_time

        if hasattr(self, 'data_time'):
            message += 'reader_cost: %.5f sec ' % self.data_time

        if hasattr(self, 'ips'):
            message += 'ips: %.5f images/s ' % self.ips

        if hasattr(self, 'step_time'):
            eta = self.step_time * (self.total_iters - self.current_iter)
            eta = eta if eta > 0 else 0

            eta_str = str(datetime.timedelta(seconds=int(eta)))
            message += f'eta: {eta_str}'

        # print the message
        self.logger.info(message)

    @property
    def current_learning_rate(self):
        for optimizer in self.model.optimizers.values():
            return optimizer.get_lr()

    def visual(self,
               results_dir,
               visual_results=None,
               step=None,
               is_save_image=False):
        """
        visual the images, use visualdl or directly write to the directory
        Parameters:
            results_dir (str)     --  directory name which contains saved images
            visual_results (dict) --  the results images dict
            step (int)            --  global steps, used in visualdl
            is_save_image (bool)  --  weather write to the directory or visualdl
        """
        self.model.compute_visuals()

        if visual_results is None:
            visual_results = self.model.get_current_visuals()

        min_max = self.cfg.get('min_max', None)
        if min_max is None:
            min_max = (-1., 1.)

        image_num = self.cfg.get('image_num', None)
        if (image_num is None) or (not self.enable_visualdl):
            image_num = 1
        for label, image in visual_results.items():
            image_numpy = tensor2img(image, min_max, image_num)
            if (not is_save_image) and self.enable_visualdl:
                self.vdl_logger.add_image(
                    results_dir + '/' + label,
                    image_numpy,
                    step=step if step else self.global_steps,
                    dataformats="HWC" if image_num == 1 else "NCHW")
            else:
                if self.cfg.is_train:
                    if self.by_epoch:
                        msg = 'epoch%.3d_' % self.current_epoch
                    else:
                        msg = 'iter%.9d_' % self.current_iter
                else:
                    msg = ''
                makedirs(os.path.join(self.output_dir, results_dir))
                img_path = os.path.join(self.output_dir, results_dir,
                                        msg + '%s.png' % (label))
                save_image(image_numpy, img_path)

    def save(self, steps, name='checkpoint', keep=1):
        if self.local_rank != 0:
            return

        assert name in ['checkpoint', 'weight']

        state_dicts = {}
        if self.by_epoch:
            save_filename = 'epoch_%s_%s.pdparams' % (
                steps // self.iters_per_epoch, name)
        else:
            save_filename = 'iter_%s_%s.pdparams' % (steps, name)

        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, save_filename)
        for net_name, net in self.model.nets.items():
            state_dicts[net_name] = net.state_dict()

        if hasattr(self.model, 'nets_ema'):
            for net_name, net in self.model.nets_ema.items():
                net_ema_name = net_name + '_ema'
                state_dicts[net_ema_name] = net.state_dict()

        if name == 'weight':
            save(state_dicts, save_path)
            return

        state_dicts['steps'] = steps

        for opt_name, opt in self.model.optimizers.items():
            opt_name_d = opt_name + '_d'
            state_dicts[opt_name_d] = opt.state_dict()

        save(state_dicts, save_path)

        if keep > 0:
            try:
                if self.by_epoch:
                    checkpoint_name_to_be_removed = os.path.join(
                        self.output_dir, 'epoch_%s_%s.pdparams' %
                        ((steps - keep * self.weight_interval) //
                         self.iters_per_epoch, name))
                else:
                    checkpoint_name_to_be_removed = os.path.join(
                        self.output_dir, 'iter_%s_%s.pdparams' %
                        (steps - keep * self.weight_interval, name))

                if os.path.exists(checkpoint_name_to_be_removed):
                    os.remove(checkpoint_name_to_be_removed)

            except Exception as e:
                self.logger.info('remove old checkpoints error: {}'.format(e))

    def resume(self, checkpoint_path):
        state_dicts = load(checkpoint_path)
        if state_dicts.get('steps', None) is not None:
            if self.by_epoch:
                self.start_epoch = state_dicts['steps'] // self.iters_per_epoch + 1
                self.global_steps = state_dicts['steps']
                self.current_iter = state_dicts['steps'] + 1
            else:
                self.start_epoch = state_dicts['steps'] + 1
                self.global_steps = self.iters_per_epoch * state_dicts['steps']
                self.current_iter = state_dicts['steps'] + 1

        for net_name, net in self.model.nets.items():
            net.set_state_dict(state_dicts[net_name])

        if hasattr(self.model, 'nets_ema'):
            for net_name, net in self.model.nets_ema.items():
                net_ema_name = net_name + '_ema'
                if net_ema_name in state_dicts:
                    net.set_state_dict(state_dicts[net_ema_name])

        for opt_name, opt in self.model.optimizers.items():
            opt_name_d = opt_name + '_d'
            opt.set_state_dict(state_dicts[opt_name_d])

    def load(self, weight_path):
        state_dicts = load(weight_path)

        def is_dict_in_dict_weight(state_dict):
            if isinstance(state_dict, dict) and len(state_dict) > 0:
                val = list(state_dict.values())[0]
                if isinstance(val, dict):
                    return True
                else:
                    return False
            else:
                return False

        if is_dict_in_dict_weight(state_dicts):
            for net_name, net in self.model.nets.items():
                if net_name in state_dicts:
                    net.set_state_dict(state_dicts[net_name])
                    self.logger.info(
                        'Loaded pretrained weight for net {}'.format(net_name))
                else:
                    self.logger.warning(
                        'Can not find state dict of net {}. Skip load pretrained weight for net {}'
                        .format(net_name, net_name))

            if hasattr(self.model, 'nets_ema'):
                for net_name, net in self.model.nets_ema.items():
                    net_ema_name = net_name + '_ema'
                    if net_ema_name in state_dicts:
                        net.set_state_dict(state_dicts[net_ema_name])
                        self.logger.info(
                            'Loaded pretrained weight for ema_net {}'.format(net_ema_name))
                    else:
                        self.logger.warning(
                            'Can not find state dict of net {}. Skip load pretrained weight for net {}'
                                .format(net_ema_name, net_ema_name))
        else:
            assert len(self.model.nets
                       ) == 1, 'checkpoint only contain weight of one net, \
                                                but model contains more than one net!'

            net_name, net = list(self.model.nets.items())[0]
            net.set_state_dict(state_dicts)
            self.logger.info(
                'Loaded pretrained weight for net {}'.format(net_name))

    def close(self):
        """
        when finish the training need close file handler or other.
        """
        if self.enable_visualdl:
            self.vdl_logger.close()
