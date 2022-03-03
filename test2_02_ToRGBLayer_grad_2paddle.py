
import torch
import paddle
import os
import numpy as np
from paddle_networks2 import ToRGBLayer

w_dim = 512
out_channels = 512
img_channels = 3
conv_clamp = 256
channels_last = False
fused_modconv = False

toRGBLayer = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                        conv_clamp=conv_clamp, channels_last=channels_last)
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

ckpt_file = 'pytorch_toRGBLayer.pth'
save_name = 'pytorch_toRGBLayer.pdparams'
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

