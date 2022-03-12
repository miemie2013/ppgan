import torch
import paddle
import os
import numpy as np
from ppgan.models.generators.generator_styleganv2ada import Conv2dLayer

in_channels = 256
out_channels = 256
kernel_size = 3
bias = True
activation = 'lrelu'
up = 1
down = 1
resample_filter = [1, 3, 3, 1]
conv_clamp = 256
channels_last = False
trainable = True




x_shape = [2, in_channels, 64, 64]
lr = 0.0001

model = Conv2dLayer(in_channels, out_channels, kernel_size, bias, activation, up, down, resample_filter, conv_clamp, channels_last, trainable)
model.train()

use_gpu = True
gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()

def copy(name, w, std):
    value2 = paddle.to_tensor(w, place=place)
    value = std[name]
    value = value * 0 + value2
    std[name] = value

fullyConnectedLayer_std = model.state_dict()

ckpt_file = '11.pth'
save_name = '11.pdparams'
state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))


fullyConnectedLayer_dic = {}
for key, value in state_dict.items():
    fullyConnectedLayer_dic[key] = value.data.numpy()

for key in fullyConnectedLayer_dic.keys():
    name2 = key
    w = fullyConnectedLayer_dic[key]
    if '.linear.weight' in key:
        w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
    if '.noise_strength' in key:
        print()
        w = np.reshape(w, [1, ])
    print(key)
    copy(name2, w, fullyConnectedLayer_std)
model.set_state_dict(fullyConnectedLayer_std)

paddle.save(fullyConnectedLayer_std, save_name)

