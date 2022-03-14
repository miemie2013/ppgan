import torch
import paddle
import os
import numpy as np
from ppgan.models.generators.generator_styleganv2ada import ToRGBLayer

x_shape = [1, 512, 4, 4]
w_shape = [1, 512]
in_channels = 512
out_channels = 3
w_dim = 512
kernel_size = 1
conv_clamp = 256
channels_last = False

x_shape = [1, 512, 8, 8]
w_shape = [1, 512]
in_channels = 512
out_channels = 3
w_dim = 512
kernel_size = 1
conv_clamp = 256
channels_last = False

x_shape = [1, 512, 16, 16]
w_shape = [1, 512]
in_channels = 512
out_channels = 3
w_dim = 512
kernel_size = 1
conv_clamp = 256
channels_last = False

x_shape = [1, 512, 32, 32]
w_shape = [1, 512]
in_channels = 512
out_channels = 3
w_dim = 512
kernel_size = 1
conv_clamp = 256
channels_last = False

x_shape = [1, 512, 64, 64]
w_shape = [1, 512]
in_channels = 512
out_channels = 3
w_dim = 512
kernel_size = 1
conv_clamp = 256
channels_last = False

x_shape = [1, 256, 128, 128]
w_shape = [1, 512]
in_channels = 256
out_channels = 3
w_dim = 512
kernel_size = 1
conv_clamp = 256
channels_last = False

x_shape = [1, 128, 256, 256]
w_shape = [1, 512]
in_channels = 128
out_channels = 3
w_dim = 512
kernel_size = 1
conv_clamp = 256
channels_last = False

x_shape = [1, 64, 512, 512]
w_shape = [1, 512]
in_channels = 64
out_channels = 3
w_dim = 512
kernel_size = 1
conv_clamp = None
channels_last = False




toRGBLayer = ToRGBLayer(in_channels, out_channels, w_dim, kernel_size, conv_clamp, channels_last)
toRGBLayer.train()

use_gpu = True
gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()

def copy(name, w, std):
    value2 = paddle.to_tensor(w, place=place)
    value = std[name]
    value = value * 0 + value2
    std[name] = value

toRGBLayer_std = toRGBLayer.state_dict()

ckpt_file = '51.pth'
save_name = '51.pdparams'
state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))


toRGBLayer_dic = {}
for key, value in state_dict.items():
    toRGBLayer_dic[key] = value.data.numpy()

for key in toRGBLayer_dic.keys():
    name2 = key
    w = toRGBLayer_dic[key]
    if '.linear.weight' in key:
        w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
    if '.noise_strength' in key:
        print()
        w = np.reshape(w, [1, ])
    print(key)
    copy(name2, w, toRGBLayer_std)
toRGBLayer.set_state_dict(toRGBLayer_std)

paddle.save(toRGBLayer_std, save_name)

