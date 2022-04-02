# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import sys
import paddle
import time


cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from ppgan.utils.config import get_config
from ppgan.utils.setup import setup
from ppgan.engine.trainer import Trainer
from ppgan.utils.logger import setup_logger
from ppgan.metrics.inception import InceptionV3

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='PaddleGAN')
    parser.add_argument('-c',
                        '--config-file',
                        metavar="FILE",
                        help='config file path')
    # cuda setting
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    # checkpoint and log
    parser.add_argument('--load',
                        type=str,
                        default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument("-db", "--dataset_batch_size", type=int, default=64, help="dataset batch size")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("-n", "--num_gen", type=int, default=50000, help="num gen")
    parser.add_argument(
        "--inceptionv3_path",
        default="",
        type=str,
        help="inceptionv3_path",
    )
    parser.add_argument(
        "--col_styles",
        default="0,1,2,3,4,5,6",
        type=str,
        help="stylegan2ada style-mixing col_styles",
    )

    # config options
    parser.add_argument("-o",
                        "--opt",
                        nargs='+',
                        help="set configuration options")
    args = parser.parse_args()

    return args


def setup(args, cfg):
    cfg.is_train = False
    cfg.profiler_options = None

    cfg.timestamp = time.strftime('-%Y-%m-%d-%H-%M', time.localtime())
    cfg.output_dir = os.path.join(
        cfg.output_dir,
        os.path.splitext(os.path.basename(str(args.config_file)))[0] +
        cfg.timestamp)

    logger = setup_logger(cfg.output_dir)

    logger.info('Configs: {}'.format(cfg))

    if paddle.is_compiled_with_cuda():
        paddle.set_device('gpu')
    else:
        paddle.set_device('cpu')


def main(args, cfg):
    # init environment, include logger, dynamic graph, seed, device, train or test mode...
    setup(args, cfg)
    # build trainer
    trainer = Trainer(cfg)

    if args.load:
        trainer.load(args.load)

    # build inceptionv3
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inceptionv3_model = InceptionV3([block_idx], normalize_input=False)
    param_dict = paddle.load(args.inceptionv3_path)
    inceptionv3_model.load_dict(param_dict)
    inceptionv3_model.eval()

    # calc stylegan2ada metric
    trainer.calc_stylegan2ada_metric(inceptionv3_model, args.dataset_batch_size, args.batch_size, args.num_gen)


if __name__ == '__main__':
    args = parse_args()
    # 判断是否是调试状态
    isDebug = True if sys.gettrace() else False
    if isDebug:
        print('Debug Mode.')
        args.config_file = '../' + args.config_file
        if args.load is not None:
            args.load = '../' + args.load   # 如果是绝对路径，把这一行注释掉
        if args.inceptionv3_path is not None:
            args.inceptionv3_path = '../' + args.inceptionv3_path   # 如果是绝对路径，把这一行注释掉
    cfg = get_config(args.config_file, args.opt)
    if isDebug:
        cfg['dataset']['train']['dataroot'] = '../' + cfg['dataset']['train']['dataroot']   # 如果是绝对路径，把这一行注释掉
        if 'dataroot' in cfg['dataset']['test'].keys():
            cfg['dataset']['test']['dataroot'] = '../' + cfg['dataset']['test']['dataroot']   # 如果是绝对路径，把这一行注释掉
        cfg['output_dir'] = '../' + cfg['output_dir']   # 如果是绝对路径，把这一行注释掉
        if 'vgg19_ckpt1' in cfg['model'].keys():
            cfg['model']['vgg19_ckpt1'] = '../' + cfg['model']['vgg19_ckpt1']   # 如果是绝对路径，把这一行注释掉
        if 'vgg19_ckpt2' in cfg['model'].keys():
            cfg['model']['vgg19_ckpt2'] = '../' + cfg['model']['vgg19_ckpt2']   # 如果是绝对路径，把这一行注释掉

    main(args, cfg)
