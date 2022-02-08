import paddle
import cv2

from ppgan.models.pastagan_model import vgg19, VGG19_feature_color_torchversion
from ppgan.utils.filesystem import save
from ppgan.models.generators.generator_pastagan import ConstEncoderNetwork, StyleEncoderNetwork, MappingNetwork, SynthesisNetwork
from ppgan.models.discriminators.discriminator_pastagan import PastaGANDiscriminator
# from ppgan.models.pastagan_model import PastaGANModel

import numpy as np
import torch
import os


use_gpu = True

gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()


def copy(name, w, std):
    value2 = paddle.to_tensor(w, place=place)
    value = std[name]
    value = value * 0 + value2
    std[name] = value


vgg = vgg19(pretrained=False, progress=False, ckpt_path=None)
vgg_std = vgg.state_dict()
vgg.eval()


print('\nCopying...')
ckpt_file = "./checkpoints/vgg19-dcbb9e9d.pth"
save_name = './checkpoints/vgg19-dcbb9e9d.pdparams'
state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))


all_dic = {}
for key, value in state_dict.items():
    all_dic[key] = value.data.numpy()

print()

for key in all_dic.keys():
    name2 = key
    w = all_dic[key]
    if 'classifier' in key and 'weight' in key:
        print('wwwwwwwwwwwwwwwwwwwwwwwwwwww')
        w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
    print(key)
    copy(name2, w, vgg_std)
vgg.set_state_dict(vgg_std)

paddle.save(vgg_std, save_name)


# ==========================================================

contextual_vgg = VGG19_feature_color_torchversion()
contextual_vgg_std = contextual_vgg.state_dict()
contextual_vgg.eval()


print('\nCopying...')
ckpt_file = "./checkpoints/vgg19_conv.pth"
save_name = './checkpoints/vgg19_conv.pdparams'
state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))


all_dic = {}
for key, value in state_dict.items():
    all_dic[key] = value.data.numpy()

print()

for key in all_dic.keys():
    name2 = key
    w = all_dic[key]
    print(key)
    copy(name2, w, contextual_vgg_std)
contextual_vgg.set_state_dict(contextual_vgg_std)

paddle.save(contextual_vgg_std, save_name)







