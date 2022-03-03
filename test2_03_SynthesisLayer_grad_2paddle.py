
import torch
import paddle
import os
import numpy as np
from paddle_networks2 import SynthesisLayer

w_dim = 512
in_channels = 512
out_channels = 512
conv_clamp = 256
channels_last = False
fused_modconv = False
layer_kwargs = {}


resolution = 4
synthesisLayer = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
                                conv_clamp=conv_clamp, channels_last=channels_last, **layer_kwargs)

# resample_filter = [1, 3, 3, 1]
# resolution = 8
# synthesisLayer = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution,
#                                 up=2, resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=channels_last, **layer_kwargs)

synthesisLayer.train()

use_gpu = True
gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()

def copy(name, w, std):
    value2 = paddle.to_tensor(w, place=place)
    value = std[name]
    value = value * 0 + value2
    std[name] = value

synthesisLayer_std = synthesisLayer.state_dict()

ckpt_file = 'pytorch_synthesisLayer.pth'
save_name = 'pytorch_synthesisLayer.pdparams'
state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))


synthesisLayer_dic = {}
for key, value in state_dict.items():
    synthesisLayer_dic[key] = value.data.numpy()

for key in synthesisLayer_dic.keys():
    name2 = key
    w = synthesisLayer_dic[key]
    if '.linear.weight' in key:
        w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
    if 'noise_strength' in key:
        print()
        w = np.reshape(w, [1, ])
    print(key)
    copy(name2, w, synthesisLayer_std)
synthesisLayer.set_state_dict(synthesisLayer_std)

paddle.save(synthesisLayer_std, save_name)

