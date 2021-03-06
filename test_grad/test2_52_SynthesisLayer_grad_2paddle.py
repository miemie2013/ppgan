import torch
import paddle
import os
import numpy as np
from ppgan.models.generators.generator_styleganv2ada import SynthesisLayer

x_shape = [1, 512, 4, 4]
w_shape = [1, 512]
in_channels = 512
out_channels = 512
w_dim = 512
resolution = 4
kernel_size = 3
up = 1
use_noise = True
activation = 'lrelu'
resample_filter = [1, 3, 3, 1]
conv_clamp = 256
channels_last = False
fused_modconv = True
gain = 1

# x_shape = [1, 512, 4, 4]
# w_shape = [1, 512]
# in_channels = 512
# out_channels = 512
# w_dim = 512
# resolution = 8
# kernel_size = 3
# up = 2
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1
#
# x_shape = [1, 512, 8, 8]
# w_shape = [1, 512]
# in_channels = 512
# out_channels = 512
# w_dim = 512
# resolution = 8
# kernel_size = 3
# up = 1
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1
#
# x_shape = [1, 512, 8, 8]
# w_shape = [1, 512]
# in_channels = 512
# out_channels = 512
# w_dim = 512
# resolution = 16
# kernel_size = 3
# up = 2
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1
#
# x_shape = [1, 512, 16, 16]
# w_shape = [1, 512]
# in_channels = 512
# out_channels = 512
# w_dim = 512
# resolution = 16
# kernel_size = 3
# up = 1
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1
#
# x_shape = [1, 512, 16, 16]
# w_shape = [1, 512]
# in_channels = 512
# out_channels = 512
# w_dim = 512
# resolution = 32
# kernel_size = 3
# up = 2
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1
#
# x_shape = [1, 512, 32, 32]
# w_shape = [1, 512]
# in_channels = 512
# out_channels = 512
# w_dim = 512
# resolution = 32
# kernel_size = 3
# up = 1
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1
#
# x_shape = [1, 512, 32, 32]
# w_shape = [1, 512]
# in_channels = 512
# out_channels = 512
# w_dim = 512
# resolution = 64
# kernel_size = 3
# up = 2
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1
#
# x_shape = [1, 512, 64, 64]
# w_shape = [1, 512]
# in_channels = 512
# out_channels = 512
# w_dim = 512
# resolution = 64
# kernel_size = 3
# up = 1
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1
#
# x_shape = [1, 512, 64, 64]
# w_shape = [1, 512]
# in_channels = 512
# out_channels = 256
# w_dim = 512
# resolution = 128
# kernel_size = 3
# up = 2
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1
#
# x_shape = [1, 256, 128, 128]
# w_shape = [1, 512]
# in_channels = 256
# out_channels = 256
# w_dim = 512
# resolution = 128
# kernel_size = 3
# up = 1
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1
#
# x_shape = [1, 256, 128, 128]
# w_shape = [1, 512]
# in_channels = 256
# out_channels = 128
# w_dim = 512
# resolution = 256
# kernel_size = 3
# up = 2
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1
#
# x_shape = [1, 128, 256, 256]
# w_shape = [1, 512]
# in_channels = 128
# out_channels = 128
# w_dim = 512
# resolution = 256
# kernel_size = 3
# up = 1
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1
#
# x_shape = [1, 128, 256, 256]
# w_shape = [1, 512]
# in_channels = 128
# out_channels = 64
# w_dim = 512
# resolution = 512
# kernel_size = 3
# up = 2
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1

# x_shape = [1, 64, 512, 512]
# w_shape = [1, 512]
# in_channels = 64
# out_channels = 64
# w_dim = 512
# resolution = 512
# kernel_size = 3
# up = 1
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1




# ??????????????????????????????
use_noise = False
model = SynthesisLayer(in_channels, out_channels, w_dim, resolution,
                       kernel_size, up, use_noise, activation, resample_filter, conv_clamp, channels_last)
model.train()

use_gpu = True
gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()

def copy(name, w, std):
    value2 = paddle.to_tensor(w, place=place)
    value = std[name]
    value = value * 0 + value2
    std[name] = value

model_std = model.state_dict()

ckpt_file = '52.pth'
save_name = '52.pdparams'
state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))


model_dic = {}
for key, value in state_dict.items():
    model_dic[key] = value.data.numpy()

for key in model_dic.keys():
    name2 = key
    w = model_dic[key]
    if '.linear.weight' in key:
        w = w.transpose(1, 0)  # pytorch???nn.Linear()???weight??????????????????????????????paddle???nn.Linear()
    if '.noise_strength' in key:
        print()
        w = np.reshape(w, [1, ])
    print(key)
    copy(name2, w, model_std)
model.set_state_dict(model_std)

paddle.save(model_std, save_name)

