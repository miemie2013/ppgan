#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import argparse
import os
import time
import pickle
import six
import paddle
import cv2
import numpy as np
import torch
import os

# add python path of this repo to sys.path
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from ppgan.utils.config import get_config
from ppgan.utils.filesystem import save
from ppgan.models.generators.generator_styleganv2ada import StyleGANv2ADA_MappingNetwork, StyleGANv2ADA_SynthesisNetwork
from ppgan.models.discriminators.discriminator_styleganv2ada import StyleGANv2ADA_Discriminator


use_gpu = False

gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()


def copy(name, w, std):
    value2 = paddle.to_tensor(w, place=place)
    value = std[name]
    value = value * 0 + value2
    std[name] = value


def make_parser():
    parser = argparse.ArgumentParser("StyleGAN convert weights")
    parser.add_argument('-c',
                        '--config-file',
                        metavar="FILE",
                        help='config file path')
    parser.add_argument("-c_G", "--c_G", default=None, type=str, help="generator checkpoint")
    parser.add_argument("-c_Gema", "--c_Gema", default=None, type=str, help="generator_ema checkpoint")
    parser.add_argument("-c_D", "--c_D", default=None, type=str, help="discriminator checkpoint")
    parser.add_argument("-oc", "--output_ckpt", default=None, type=str, help="output checkpoint")
    parser.add_argument("-o",
                        "--opt",
                        nargs='+',
                        help="set configuration options")
    return parser


def main(cfg, args):
    print("Args: {}".format(args))
    archi_name = cfg.model.name

    # 新增算法时这里也要增加elif
    if archi_name == 'StyleGANv2ADAModel':
        img_resolution = cfg.model.synthesis.img_resolution
        img_channels = cfg.model.synthesis.img_channels

        synthesis_kwargs = dict(
            channel_base=cfg.model.synthesis.channel_base,
            channel_max=cfg.model.synthesis.channel_max,
            num_fp16_res=cfg.model.synthesis.num_fp16_res,
            conv_clamp=cfg.model.synthesis.conv_clamp,
        )

        z_dim = cfg.model.mapping.z_dim
        c_dim = cfg.model.mapping.c_dim
        w_dim = cfg.model.mapping.w_dim

        mapping_kwargs = dict(
            num_layers=cfg.model.mapping.num_layers,
        )

        synthesis = StyleGANv2ADA_SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution,
                                                   img_channels=img_channels, **synthesis_kwargs)
        num_ws = synthesis.num_ws
        mapping = StyleGANv2ADA_MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=num_ws, **mapping_kwargs)

        synthesis_ema = StyleGANv2ADA_SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution,
                                                       img_channels=img_channels, **synthesis_kwargs)
        mapping_ema = StyleGANv2ADA_MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=num_ws,
                                                   **mapping_kwargs)

        channel_base = cfg.model.discriminator.channel_base
        channel_max = cfg.model.discriminator.channel_max
        num_fp16_res = cfg.model.discriminator.num_fp16_res
        conv_clamp = cfg.model.discriminator.conv_clamp
        epilogue_kwargs = dict(
            mbstd_group_size=cfg.model.discriminator.epilogue_kwargs.mbstd_group_size,
        )
        discriminator = StyleGANv2ADA_Discriminator(c_dim=c_dim,
                                                    img_resolution=img_resolution,
                                                    img_channels=img_channels,
                                                    channel_base=channel_base,
                                                    channel_max=channel_max,
                                                    num_fp16_res=num_fp16_res,
                                                    conv_clamp=conv_clamp,
                                                    block_kwargs={},
                                                    mapping_kwargs={},
                                                    epilogue_kwargs=epilogue_kwargs,
                                                    )

        mapping_std = mapping.state_dict()
        synthesis_std = synthesis.state_dict()
        mapping_ema_std = mapping_ema.state_dict()
        synthesis_ema_std = synthesis_ema.state_dict()
        discriminator_std = discriminator.state_dict()

        mapping.eval()
        synthesis.eval()
        mapping_ema.eval()
        synthesis_ema.eval()
        discriminator.eval()

        # save_name = '../styleganv2ada_32_afhqcat_step19_pytorch.pdparams'
        save_name = args.output_ckpt
        G_ema_state_dict = torch.load(args.c_Gema, map_location=torch.device('cpu'))
        G_state_dict = torch.load(args.c_G, map_location=torch.device('cpu'))
        D_state_dict = torch.load(args.c_D, map_location=torch.device('cpu'))

        synthesis_ema_dic = {}
        mapping_ema_dic = {}
        others = {}
        for key, value in G_ema_state_dict.items():
            # if 'tracked' in key:
            #     continue
            if 'synthesis' in key:
                synthesis_ema_dic[key] = value.data.numpy()
            elif 'mapping' in key:
                mapping_ema_dic[key] = value.data.numpy()
            else:
                others[key] = value.data.numpy()

        synthesis_dic = {}
        mapping_dic = {}
        for key, value in G_state_dict.items():
            # if 'tracked' in key:
            #     continue
            if 'synthesis' in key:
                synthesis_dic[key] = value.data.numpy()
            elif 'mapping' in key:
                mapping_dic[key] = value.data.numpy()
            else:
                others[key] = value.data.numpy()

        discriminator_dic = {}
        for key, value in D_state_dict.items():
            discriminator_dic[key] = value.data.numpy()

        # print()

        for key in mapping_dic.keys():
            name2 = key
            w = mapping_dic[key]
            name2 = name2.replace('mapping.', '')
            if '.linear.weight' in key:
                w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
            # print(key)
            copy(name2, w, mapping_std)
        mapping.set_state_dict(mapping_std)

        for key in mapping_ema_dic.keys():
            name2 = key
            w = mapping_ema_dic[key]
            name2 = name2.replace('mapping.', '')
            if '.linear.weight' in key:
                w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
            # print(key)
            copy(name2, w, mapping_ema_std)
        mapping_ema.set_state_dict(mapping_ema_std)

        map = {}

        conv_i = 0
        torgb_i = 0
        for block_idx, res in enumerate(synthesis.block_resolutions):
            in_channels = synthesis.channels_dict[res // 2] if res > 4 else 0
            is_last = synthesis.is_lasts[block_idx]
            architecture = synthesis.architectures[block_idx]

            if in_channels == 0:
                map[f'b{res}.const'] = 'const'
            else:
                pass

            # Main layers.
            if in_channels == 0:
                map[f'b{res}.conv1'] = 'convs.%d' % (conv_i)
                conv_i += 1
            # elif self.architecture == 'resnet':
            #     y = self.skip(x, gain=np.sqrt(0.5))
            #     x = self.conv0(x, ws[:, i + 1], fused_modconv=fused_modconv, **layer_kwargs)
            #     x = self.conv1(x, ws[:, i + 1], fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            #     x = y.add_(x)
            else:
                map[f'b{res}.conv0'] = 'convs.%d' % (conv_i)
                map[f'b{res}.conv1'] = 'convs.%d' % (conv_i + 1)
                conv_i += 2

            # ToRGB.
            map[f'b{res}.resample_filter'] = f"resample_filter_{block_idx}"
            if is_last or architecture == 'skip':
                map[f'b{res}.torgb'] = 'torgbs.%d' % (torgb_i)
                torgb_i += 1

        for key in synthesis_dic.keys():
            name2 = None
            for key2 in map.keys():
                if key2 in key:
                    name2 = key.replace(key2, map[key2])
                    name2 = name2.split('ynthesis.')[1]
                    break
            w = synthesis_dic[key]
            if '.linear.weight' in key:
                # print('.linear.weight...')
                w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
            if '.noise_strength' in key:
                # print('noise_strength...')
                w = np.reshape(w, [1, ])
            # print(key)
            copy(name2, w, synthesis_std)
        synthesis.set_state_dict(synthesis_std)

        for key in synthesis_ema_dic.keys():
            name2 = None
            for key2 in map.keys():
                if key2 in key:
                    name2 = key.replace(key2, map[key2])
                    name2 = name2.split('ynthesis.')[1]
                    break
            w = synthesis_ema_dic[key]
            if '.linear.weight' in key:
                # print('.linear.weight...')
                w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
            if '.noise_strength' in key:
                # print('noise_strength...')
                w = np.reshape(w, [1, ])
            # print(key)
            copy(name2, w, synthesis_ema_std)
        synthesis_ema.set_state_dict(synthesis_ema_std)

        for key in discriminator_dic.keys():
            name2 = key
            w = discriminator_dic[key]
            name2 = name2.replace('discriminator.', '')
            if '.linear.weight' in key:
                w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
            if '.noise_strength' in key:
                # print()
                w = np.reshape(w, [1, ])
            # print(key)
            copy(name2, w, discriminator_std)
        discriminator.set_state_dict(discriminator_std)

        class_idx = None
        label = paddle.zeros([1, mapping.c_dim])
        if mapping.c_dim != 0:
            if class_idx is None:
                print('Must specify class label with --class when using a conditional network')
            label[:, class_idx] = 1
        else:
            if class_idx is not None:
                print('warn: --class=lbl ignored when running on an unconditional network')

        class Model(paddle.nn.Layer):
            def __init__(self, synthesis, mapping, synthesis_ema, mapping_ema, discriminator):
                super().__init__()
                self.nets = {}
                self.nets['synthesis'] = synthesis
                self.nets['mapping'] = mapping
                self.nets['synthesis_ema'] = synthesis_ema
                self.nets['mapping_ema'] = mapping_ema
                if discriminator is not None:
                    self.nets['discriminator'] = discriminator

            def forward(self, x):
                return x

        model = Model(synthesis, mapping, synthesis_ema, mapping_ema, discriminator=discriminator)

        state_dicts = {}
        for net_name, net in model.nets.items():
            state_dicts[net_name] = net.state_dict()
        save(state_dicts, save_name)
    elif archi_name == 'StyleGANv3Model':
        pass
    else:
        raise NotImplementedError("Architectures \'{}\' is not implemented.".format(model_class_name))
    print("Done.")


if __name__ == "__main__":
    args = make_parser().parse_args()
    # 判断是否是调试状态
    isDebug = True if sys.gettrace() else False
    if isDebug:
        print('Debug Mode.')
        args.config_file = '../' + args.config_file
        args.c_G = '../' + args.c_G
        args.c_Gema = '../' + args.c_Gema
        args.c_D = '../' + args.c_D
        args.output_ckpt = '../' + args.output_ckpt
    cfg = get_config(args.config_file, args.opt)

    main(cfg, args)
