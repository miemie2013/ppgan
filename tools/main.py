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

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from ppgan.utils.options import parse_args
from ppgan.utils.config import get_config
from ppgan.utils.setup import setup
from ppgan.engine.trainer import Trainer


def main(args, cfg):
    # init environment, include logger, dynamic graph, seed, device, train or test mode...
    setup(args, cfg)
    # build trainer
    trainer = Trainer(cfg)

    # continue train or evaluate, checkpoint need contain epoch and optimizer info
    if args.resume:
        trainer.resume(args.resume)
    # evaluate or finute, only load generator weights
    elif args.load:
        trainer.load(args.load)

    if args.evaluate_only:
        trainer.test()
        return
    # training, when keyboard interrupt save weights
    try:
        trainer.train()
    except KeyboardInterrupt as e:
        trainer.save(trainer.current_epoch)
    trainer.close()


if __name__ == '__main__':
    args = parse_args()
    # 判断是否是调试状态
    isDebug = True if sys.gettrace() else False
    if isDebug:
        print('Debug Mode.')
        args.config_file = '../' + args.config_file
        if args.load is not None:
            args.load = '../' + args.load   # 如果是绝对路径，把这一行注释掉
        if args.resume is not None:
            args.resume = '../' + args.resume   # 如果是绝对路径，把这一行注释掉
        if args.model_path is not None:
            args.model_path = '../' + args.model_path   # 如果是绝对路径，把这一行注释掉
        args.source_path = '../' + args.source_path   # 如果是绝对路径，把这一行注释掉
        args.reference_dir = '../' + args.reference_dir   # 如果是绝对路径，把这一行注释掉
    cfg = get_config(args.config_file, args.opt)
    if isDebug:
        cfg['dataset']['train']['dataroot'] = '../' + cfg['dataset']['train']['dataroot']   # 如果是绝对路径，把这一行注释掉
        cfg['dataset']['test']['dataroot'] = '../' + cfg['dataset']['test']['dataroot']   # 如果是绝对路径，把这一行注释掉

    main(args, cfg)
