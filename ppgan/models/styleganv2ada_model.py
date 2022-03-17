# code was heavily based on https://github.com/clovaai/stargan-v2
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/clovaai/stargan-v2#license
import cv2
from paddle.fluid.layers.nn import soft_relu
from .base_model import BaseModel

from paddle import nn
import paddle
import paddle.nn.functional as F
from .builder import MODELS
from .generators.builder import build_generator
from .discriminators.builder import build_discriminator
from ..modules.init import kaiming_normal_, constant_
from ppgan.utils.visual import make_grid, tensor2img

import numpy as np
import sys







def soft_update(source, ema_model, beta=1.0):
    '''
    ema:
    ema = beta * ema + (1. - beta) * source

    '''
    assert 0.0 <= beta <= 1.0

    if isinstance(source, paddle.DataParallel):
        source = source._layers

    ema_model_map = dict(ema_model.named_parameters())
    for param_name, source_param in source.named_parameters():
        ema_param = ema_model_map[param_name]
        ema_param.set_value(beta * ema_param + (1.0 - beta) * source_param)


def dump_model(model):
    params = {}
    for k in model.state_dict().keys():
        if k.endswith('.scale'):
            params[k] = model.state_dict()[k].shape
    return params



def he_init(module):
    if isinstance(module, nn.Conv2D):
        kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            constant_(module.bias, 0)


@MODELS.register()
class StyleGANv2ADAModel(BaseModel):
    def __init__(
        self,
        synthesis,
        mapping=None,
        discriminator=None,
        G_reg_interval=4,
        D_reg_interval=16,
        augment_pipe=None,
        style_mixing_prob=0.9,
        r1_gamma=10,
        pl_batch_shrink=2,
        pl_decay=0.01,
        pl_weight=2.0,
        ema_kimg=10,
        ema_rampup=None,
        augment_p=0.0,
        ada_kimg=500,
        ada_interval=4,
        ada_target=None,
    ):
        super(StyleGANv2ADAModel, self).__init__()
        self.nets_ema = {}
        self.nets['synthesis'] = build_generator(synthesis)
        self.nets_ema['synthesis'] = build_generator(synthesis)
        self.num_ws = self.nets['synthesis'].num_ws
        mapping['num_ws'] = self.num_ws
        self.nets['mapping'] = build_generator(mapping)
        self.nets_ema['mapping'] = build_generator(mapping)
        if discriminator:
            self.nets['discriminator'] = build_discriminator(discriminator)
        self.nets['synthesis'].train()
        self.nets['mapping'].train()
        self.nets['discriminator'].train()
        self.nets_ema['synthesis'].eval()
        self.nets_ema['mapping'].eval()
        self.c_dim = mapping.c_dim
        self.z_dim = mapping.z_dim
        self.w_dim = mapping.w_dim

        self.phases = []
        for name, reg_interval in [('G', G_reg_interval), ('D', D_reg_interval)]:
            if reg_interval is None:
                # opt = dnnlib.util.construct_class_by_name(params=module.parameters(),
                #                                           **opt_kwargs)  # subclass of torch.optim.Optimizer
                # phases += [dnnlib.EasyDict(name=name + 'both', module=module, opt=opt, interval=1)]
                pass
            else:  # Lazy regularization.
                self.phases += [dict(name=name + 'main', interval=1)]
                self.phases += [dict(name=name + 'reg', interval=reg_interval)]

        self.z_dim = self.nets['mapping'].z_dim
        self.cur_nimg = 0
        self.batch_idx = 0

        # loss config.
        self.augment_pipe = build_generator(augment_pipe)
        self.style_mixing_prob = style_mixing_prob
        # self.augment_pipe = None
        # self.style_mixing_prob = -1.0
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight

        self.pl_mean = paddle.zeros([1, ], dtype=paddle.float32)
        self.ema_kimg = ema_kimg
        self.ema_rampup = ema_rampup

        self.augment_p = augment_p
        self.ada_kimg = ada_kimg
        self.ada_target = ada_target
        self.ada_interval = ada_interval
        self.adjust_p = False   # 是否调整augment_pipe的p
        if self.augment_pipe is not None and (self.augment_p > 0 or self.ada_target is not None):
            self.augment_pipe.train()
            for param_name, param in self.augment_pipe.named_parameters():
                param.stop_gradient = True
            self.augment_pipe.p.set_value(paddle.to_tensor(self.augment_p))
            if self.ada_target is not None:
                self.adjust_p = True
        self.Loss_signs_real = []

        self.align_grad = False
        # self.align_grad = True



    def setup_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Args:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.input = input

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    def run_G(self, z, c, sync):
        ws = self.nets['mapping'](z, c)
        if self.style_mixing_prob > 0:
            num_vector = self.num_ws
            cutoff_ = paddle.randint(low=1, high=num_vector, shape=[1, ], dtype='int64')
            cond = paddle.rand([1, ], dtype='float32') < self.style_mixing_prob
            cutoff = paddle.where(cond, cutoff_, paddle.full_like(cutoff_, num_vector))
            cutoff.stop_gradient = True
            if cutoff == num_vector:
                pass
            else:
                temp = self.nets['mapping'](paddle.randn(z.shape), c, skip_w_avg_update=True)[:, cutoff:]
                temp2 = ws[:, :cutoff]
                ws = paddle.concat([temp2, temp], 1)
        img = self.nets['synthesis'](ws)
        return img, ws

    def run_G_grad(self, dloss_dout):
        dloss_dws = self.nets['synthesis'].grad_layer(dloss_dout)
        dloss_dws = paddle.stack(dloss_dws, 1)
        return dloss_dws

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits = self.nets['discriminator'](img, c)
        return logits

    def run_D_grad(self, dloss_dout):
        dloss_daug_x = self.nets['discriminator'].grad_layer(dloss_dout)
        if self.augment_pipe is not None:
            dloss_dx = self.augment_pipe.grad_layer(dloss_daug_x)
        else:
            dloss_dx = dloss_daug_x
        return dloss_dx

    # 梯度累加（变相增大批大小）。
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain, dic2=None):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        loss_numpy = {}

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
            # if self.align_grad:
            #     ddd = np.sum((dic2[phase + 'gen_img'] - gen_img.numpy()) ** 2)
            #     print('do_Gmain gen_img=%.6f' % ddd)
            #     ddd = np.sum((dic2[phase + '_gen_ws'] - _gen_ws.numpy()) ** 2)
            #     print('do_Gmain _gen_ws=%.6f' % ddd)

            gen_logits = self.run_D(gen_img, gen_c, sync=False)
            # if self.align_grad:
            #     ddd = np.sum((dic2[phase + 'gen_logits'] - gen_logits.numpy()) ** 2)
            #     print('do_Gmain gen_logits=%.6f' % ddd)

            loss_Gmain = paddle.nn.functional.softplus(-gen_logits)  # -log(sigmoid(gen_logits))
            loss_Gmain = loss_Gmain.mean()
            loss_numpy['loss_Gmain'] = loss_Gmain.numpy()

            loss_G = loss_Gmain
            loss_G = loss_G * float(gain)
            loss_G.backward()  # 咩酱：gain即上文提到的这个阶段的训练间隔。

        # Gpl: Apply path length regularization.
        if do_Gpl:
            batch_size = gen_z.shape[0] // self.pl_batch_shrink
            batch_size = max(batch_size, 1)

            gen_c_ = None
            if gen_c is not None:
                gen_c_ = gen_c[:batch_size]

            gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c_, sync=sync)
            # if self.align_grad:
            #     ddd = np.sum((dic2[phase + 'gen_img'] - gen_img.numpy()) ** 2)
            #     print('do_Gpl gen_img=%.6f' % ddd)
            #     ddd = np.sum((dic2[phase + 'gen_ws'] - gen_ws.numpy()) ** 2)
            #     print('do_Gpl gen_ws=%.6f' % ddd)
            pl_noise = paddle.randn(gen_img.shape) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
            # pl_noise = paddle.ones(gen_img.shape) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
            dgen_img_dgen_img = paddle.ones(gen_img.shape, dtype=paddle.float32)
            dgen_img_dgen_img = dgen_img_dgen_img * pl_noise
            pl_grads = self.run_G_grad(dgen_img_dgen_img)

            pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
            # if self.align_grad:
            #     aaaaaaaaa1 = dic2[phase + 'pl_grads']
            #     aaaaaaaaa2 = pl_grads.numpy()
            #     aaaaaaaaa3 = dic2[phase + 'pl_lengths']
            #     aaaaaaaaa4 = pl_lengths.numpy()
            #     ddd = np.sum((dic2[phase + 'pl_grads'] - pl_grads.numpy()) ** 2)
            #     print('do_Gpl pl_grads=%.6f' % ddd)
            #     ddd = np.sum((dic2[phase + 'pl_lengths'] - pl_lengths.numpy()) ** 2)
            #     print('do_Gpl pl_lengths=%.6f' % ddd)

            pl_mean = self.pl_mean + self.pl_decay * (pl_lengths.mean() - self.pl_mean)
            self.pl_mean.set_value(pl_mean.detach())

            pl_penalty = (pl_lengths - pl_mean).square()
            loss_Gpl = pl_penalty * self.pl_weight

            loss_Gpl = (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean() * float(gain)
            loss_numpy['loss_Gpl'] = loss_Gpl.numpy()
            loss_Gpl.backward()  # 咩酱：gain即上文提到的这个阶段的训练间隔。

        # Dmain: Minimize logits for generated images.
        loss3 = 0.0
        if do_Dmain:
            gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
            gen_img.stop_gradient = True   # 训练判别器时，假图片要停止梯度，不能更新生成器的参数。
            # if self.align_grad:
            #     ddd = np.sum((dic2[phase + 'gen_img'] - gen_img.numpy()) ** 2)
            #     print('do_Dmain gen_img=%.6f' % ddd)
            #     ddd = np.sum((dic2[phase + '_gen_ws'] - _gen_ws.numpy()) ** 2)
            #     print('do_Dmain _gen_ws=%.6f' % ddd)
            gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
            # if self.align_grad:
            #     ddd = np.sum((dic2[phase + 'gen_logits'] - gen_logits.numpy()) ** 2)
            #     print('do_Dmain gen_logits=%.6f' % ddd)

            loss_Dgen = paddle.nn.functional.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))
            loss_Dgen = loss_Dgen.mean()
            loss_numpy['loss_Dgen'] = loss_Dgen.numpy()

            loss3 = loss_Dgen * float(gain)

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            real_img_tmp = real_img.detach()
            real_img_tmp.stop_gradient = not do_Dr1
            real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
            if self.adjust_p and self.augment_pipe is not None:
                self.Loss_signs_real.append(real_logits.sign().numpy())
            # if self.align_grad:
            #     ddd = np.sum((dic2[phase + 'real_logits'] - real_logits.numpy()) ** 2)
            #     print('do_Dmain or do_Dr1 real_logits=%.6f' % ddd)

            loss_Dreal = 0
            if do_Dmain:
                loss_Dreal = paddle.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                # if self.align_grad:
                #     ddd = np.sum((dic2[phase + 'loss_Dreal'] - loss_Dreal.numpy()) ** 2)
                #     print('do_Dmain or do_Dr1 do_Dmain loss_Dreal=%.6f' % ddd)
                loss_numpy['loss_Dreal'] = loss_Dreal.numpy().mean()

            loss_Dr1 = 0
            if do_Dr1:
                dreal_logitssum_dreal_logits = paddle.ones(real_logits.shape, dtype=paddle.float32)
                r1_grads = self.run_D_grad(dreal_logitssum_dreal_logits)
                # if self.align_grad:
                #     ddd = np.sum((dic2[phase + 'r1_grads'] - r1_grads.numpy()) ** 2)
                #     print('do_Dmain or do_Dr1 do_Dr1 r1_grads=%.6f' % ddd)

                r1_penalty = r1_grads.square().sum([1, 2, 3])
                # if self.align_grad:
                #     ddd = np.sum((dic2[phase + 'r1_penalty'] - r1_penalty.numpy()) ** 2)
                #     print('do_Dmain or do_Dr1 do_Dr1 r1_penalty=%.6f' % ddd)
                loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                loss_numpy['loss_Dr1'] = loss_Dr1.numpy().mean()
            loss4 = (loss_Dreal + loss_Dr1).mean() * float(gain)
            if do_Dmain:
                loss4 += loss3
            loss4.backward()  # 咩酱：gain即上文提到的这个阶段的训练间隔。
        return loss_numpy

    def train_iter(self, optimizers=None):
        phase_real_img = self.input[0]
        phase_real_c = self.input[1]
        phases_all_gen_c = self.input[2]

        # 对齐梯度用
        dic2 = None
        # if self.align_grad:
        #     print('======================== batch%.5d.npz ========================'%self.batch_idx)
        #     npz_path = 'batch%.5d.npz'%self.batch_idx
        #     isDebug = True if sys.gettrace() else False
        #     if isDebug:
        #         npz_path = '../batch%.5d.npz'%self.batch_idx
        #     dic2 = np.load(npz_path)
        #     aaaaaaaaa = dic2['phase_real_img']
        #     phase_real_img = paddle.to_tensor(aaaaaaaaa)


        phase_real_img = paddle.cast(phase_real_img, dtype=paddle.float32) / 127.5 - 1


        phases = self.phases
        batch_size = phase_real_img.shape[0]

        all_gen_z = None
        num_gpus = 1  # 显卡数量
        batch_gpu = batch_size // num_gpus  # 一张显卡上的批大小
        if self.z_dim > 0:
            all_gen_z = paddle.randn([len(phases) * batch_size, self.z_dim])  # 咩酱：训练的4个阶段每个gpu的噪声
            # if self.align_grad:
            #     all_gen_z = paddle.to_tensor(dic2['all_gen_z'], dtype=all_gen_z.dtype)
        else:
            all_gen_z = paddle.randn([len(phases) * batch_size, 1])  # 咩酱：训练的4个阶段每个gpu的噪声
        phases_all_gen_z = paddle.split(all_gen_z, num_or_sections=len(phases))  # 咩酱：训练的4个阶段的噪声
        all_gen_z = [paddle.split(phase_gen_z, num_or_sections=num_gpus) for phase_gen_z in phases_all_gen_z]  # 咩酱：训练的4个阶段每个gpu的噪声

        c_dim = phases_all_gen_c[0].shape[1]
        all_gen_c = None
        if c_dim > 0:
            all_gen_c = [paddle.split(phase_gen_c, num_or_sections=num_gpus) for phase_gen_c in phases_all_gen_c]  # 咩酱：训练的4个阶段每个gpu的类别
        else:
            all_gen_c = [[None for _2 in range(num_gpus)] for _1 in range(len(phases))]

        phase_real_img = paddle.split(phase_real_img, num_or_sections=num_gpus)

        c_dim = phase_real_c.shape[1]
        if c_dim > 0:
            phase_real_c = paddle.split(phase_real_c, num_or_sections=num_gpus)
        else:
            phase_real_c = [[None for _2 in range(num_gpus)] for _1 in range(len(phases))]

        # Execute training phases.  咩酱：训练的4个阶段。一个批次的图片训练4个阶段。
        loss_numpys = []
        loss_phase_name = []
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):  # 咩酱：phase_gen_z是这个阶段每个gpu的噪声，是一个元组，元组长度等于gpu数量。
            if self.batch_idx % phase['interval'] != 0:  # 咩酱：每一个阶段phase有一个属性interval，即训练间隔，每隔几个批次图片才会执行1次这个阶段！
                continue

            # Initialize gradient accumulation.  咩酱：初始化梯度累加（变相增大批大小）。
            if 'G' in phase['name']:
                optimizers['generator'].clear_gradients()
                for name, param in self.nets['synthesis'].named_parameters():
                    param.stop_gradient = False
                for name, param in self.nets['mapping'].named_parameters():
                    param.stop_gradient = False
                for name, param in self.nets['discriminator'].named_parameters():
                    param.stop_gradient = True
            elif 'D' in phase['name']:
                optimizers['discriminator'].clear_gradients()
                for name, param in self.nets['synthesis'].named_parameters():
                    param.stop_gradient = True
                for name, param in self.nets['mapping'].named_parameters():
                    param.stop_gradient = True
                for name, param in self.nets['discriminator'].named_parameters():
                    param.stop_gradient = False

            # 梯度累加。一个总的批次的图片分开{显卡数量}次遍历。
            # Accumulate gradients over multiple rounds.  咩酱：遍历每一个gpu上的批次图片。这样写好奇葩啊！round_idx是gpu_id
            for round_idx, (real_img, real_c, gen_z, gen_c) in enumerate(zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c)):
                sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)   # 咩酱：右边的式子结果一定是0。即只有0号gpu做同步。这是梯度累加的固定写法。
                gain = phase['interval']     # 咩酱：即上文提到的训练间隔。

                # 梯度累加（变相增大批大小）。
                loss_numpy = self.accumulate_gradients(phase=phase['name'], real_img=real_img, real_c=real_c,
                                                       gen_z=gen_z, gen_c=gen_c, sync=sync, gain=gain, dic2=dic2)
                loss_numpys.append(loss_numpy)
                loss_phase_name.append(phase['name'])

            # Update weights.
            # phase.module.requires_grad_(False)
            # 梯度裁剪
            # for param in phase.module.parameters():
            #     if param.grad is not None:
            #         misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            if 'G' in phase['name']:
                optimizers['generator'].step()  # 更新参数
            elif 'D' in phase['name']:
                optimizers['discriminator'].step()  # 更新参数


        # compute moving average of network parameters。指数滑动平均
        # self.mapping_ema.requires_grad_(False)
        # self.synthesis_ema.requires_grad_(False)
        ema_kimg = self.ema_kimg
        ema_nimg = ema_kimg * 1000
        ema_rampup = self.ema_rampup
        cur_nimg = self.cur_nimg
        if ema_rampup is not None:
            ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
        ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
        soft_update(self.nets['synthesis'],
                    self.nets_ema['synthesis'],
                    beta=ema_beta)
        soft_update(self.nets['mapping'],
                    self.nets_ema['mapping'],
                    beta=ema_beta)

        # if self.align_grad:
        #     if self.batch_idx == 19:
        #         paddle.save(self.nets['synthesis'].state_dict(), 'synthesis_19.pdparams')
        #         paddle.save(self.nets_ema['synthesis'].state_dict(), 'synthesis_ema_19.pdparams')
        #         paddle.save(self.nets['mapping'].state_dict(), 'mapping_19.pdparams')
        #         paddle.save(self.nets_ema['mapping'].state_dict(), 'mapping_ema_19.pdparams')
        #         paddle.save(self.nets['discriminator'].state_dict(), 'discriminator_19.pdparams')

        self.cur_nimg += batch_size
        self.batch_idx += 1

        # Execute ADA heuristic.
        if self.adjust_p and self.augment_pipe is not None and (self.batch_idx % self.ada_interval == 0):
            # self.ada_interval个迭代中，real_logits.sign()的平均值。
            Loss_signs_real_mean = np.mean(np.concatenate(self.Loss_signs_real, 0))
            diff = Loss_signs_real_mean - self.ada_target
            adjust = np.sign(diff)
            # print(Loss_signs_real_mean)
            # print('==========================')
            adjust = adjust * (batch_size * self.ada_interval) / (self.ada_kimg * 1000)
            new_p = self.augment_pipe.p + adjust
            new_p = paddle.clip(new_p, min=0)
            self.augment_pipe.p.set_value(new_p)
            self.Loss_signs_real = []

        return loss_numpys

    def test_iter(self, metrics=None):
        z = self.input['z']

        class_idx = None
        label = paddle.zeros([1, self.c_dim])
        if self.c_dim != 0:
            if class_idx is None:
                print('Must specify class label with --class when using a conditional network')
            label[:, class_idx] = 1
        else:
            if class_idx is not None:
                print('warn: --class=lbl ignored when running on an unconditional network')
        # noise_mode = ['const', 'random', 'none']
        noise_mode = 'const'
        truncation_psi = 1.0

        ws = self.nets_ema['mapping'](z, label, truncation_psi=truncation_psi, truncation_cutoff=None)
        img = self.nets_ema['synthesis'](ws, noise_mode=noise_mode)

        img = (paddle.transpose(img, (0, 2, 3, 1)) * 127.5 + 128)
        img = paddle.clip(img, 0, 255)
        img = paddle.cast(img, dtype=paddle.uint8)
        img_rgb = img.numpy()[0]  # pgan是将RGB格式的图片进行保存的。

        self.visual_items['reference'] = img_rgb


    def style_mixing(self, row_seeds, col_seeds, all_seeds, col_styles):
        all_z = self.input['z']
        # noise_mode = ['const', 'random', 'none']
        noise_mode = 'const'
        truncation_psi = 1.0
        all_w = self.nets_ema['mapping'](all_z, None)
        w_avg = self.nets_ema['mapping'].w_avg
        all_w = w_avg + (all_w - w_avg) * truncation_psi
        w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}

        # print('Generating images...')
        all_images = self.nets_ema['synthesis'](all_w, noise_mode=noise_mode)
        all_images = (paddle.transpose(all_images, (0, 2, 3, 1)) * 127.5 + 128)
        all_images = paddle.clip(all_images, 0, 255)
        all_images = paddle.cast(all_images, dtype=paddle.uint8)
        all_images = all_images.numpy()
        image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

        # print('Generating style-mixed images...')
        for row_seed in row_seeds:
            for col_seed in col_seeds:
                w = w_dict[row_seed].clone()
                w[col_styles] = w_dict[col_seed][col_styles]
                w = paddle.unsqueeze(w, 0)
                image = self.nets_ema['synthesis'](w, noise_mode=noise_mode)
                image = (paddle.transpose(image, (0, 2, 3, 1)) * 127.5 + 128)
                image = paddle.clip(image, 0, 255)
                image = paddle.cast(image, dtype=paddle.uint8)
                image_dict[(row_seed, col_seed)] = image.numpy()[0]

        # print('Saving image grid...')
        ROW = len(row_seeds)
        COL = len(col_seeds)
        res = self.nets_ema['synthesis'].img_resolution
        grid_img_rgb = np.zeros(((ROW+1)*res, (COL+1)*res, 3), dtype=np.uint8)
        for j, row_seed in enumerate(row_seeds):
            for i, col_seed in enumerate(col_seeds):
                grid_img_rgb[(j+1)*res:(j+2)*res, (i+1)*res:(i+2)*res, :] = image_dict[(row_seed, col_seed)]
        for j, row_seed in enumerate(row_seeds):
            grid_img_rgb[(j+1)*res:(j+2)*res, 0:res, :] = image_dict[(row_seed, row_seed)]
        for i, col_seed in enumerate(col_seeds):
            grid_img_rgb[0:res, (i+1)*res:(i+2)*res, :] = image_dict[(col_seed, col_seed)]

        self.visual_items['reference'] = grid_img_rgb
