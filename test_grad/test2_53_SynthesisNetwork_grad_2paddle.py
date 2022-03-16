import torch
import paddle
import os
import numpy as np
from ppgan.models.generators.generator_styleganv2ada import StyleGANv2ADA_SynthesisNetwork

w_dim = 512
# img_resolution = 512
img_resolution = 128
# img_resolution = 32
img_channels = 3
channel_base = 32768
channel_max = 512
num_fp16_res = 4
conv_clamp = 256




# 需要强制设置SynthesisLayer的self.use_noise = False


synthesis_cfg = dict(
    w_dim=w_dim,
    img_resolution=img_resolution,
    img_channels=img_channels,
    channel_base=channel_base,
    channel_max=channel_max,
    num_fp16_res=num_fp16_res,
    conv_clamp=conv_clamp,
)

batch_size = 2
w_shape = [batch_size, w_dim]
lr = 0.0001

# 强制设置为不使用噪声
use_noise = False
model = StyleGANv2ADA_SynthesisNetwork(**synthesis_cfg)
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

ckpt_file = '53.pth'
save_name = '53.pdparams'
state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))


model_dic = {}
for key, value in state_dict.items():
    model_dic[key] = value.data.numpy()



map = {}

conv_i = 0
torgb_i = 0
for block_idx, res in enumerate(model.block_resolutions):
    in_channels = model.channels_dict[res // 2] if res > 4 else 0
    is_last = model.is_lasts[block_idx]
    architecture = model.architectures[block_idx]

    if in_channels == 0:
        map[f'b{res}.const'] = 'const'
    else:
        pass

    # Main layers.
    if in_channels == 0:
        map[f'b{res}.conv1'] = 'convs.%d'%(conv_i)
        conv_i += 1
    # elif self.architecture == 'resnet':
    #     y = self.skip(x, gain=np.sqrt(0.5))
    #     x = self.conv0(x, ws[:, i + 1], fused_modconv=fused_modconv, **layer_kwargs)
    #     x = self.conv1(x, ws[:, i + 1], fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
    #     x = y.add_(x)
    else:
        map[f'b{res}.conv0'] = 'convs.%d'%(conv_i)
        map[f'b{res}.conv1'] = 'convs.%d'%(conv_i + 1)
        conv_i += 2

    # ToRGB.
    map[f'b{res}.resample_filter'] = f"resample_filter_{block_idx}"
    if is_last or architecture == 'skip':
        map[f'b{res}.torgb'] = 'torgbs.%d'%(torgb_i)
        torgb_i += 1


for key in model_dic.keys():
    name2 = None
    for key2 in map.keys():
        if key2 in key:
            name2 = key.replace(key2, map[key2])
            # name2 = name2.split('ynthesis.')[1]
            break
    w = model_dic[key]
    if '.linear.weight' in key:
        print('.linear.weight...')
        w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
    if '.noise_strength' in key:
        print('noise_strength...')
        w = np.reshape(w, [1, ])
    # print(key)
    copy(name2, w, model_std)
model.set_state_dict(model_std)
paddle.save(model_std, save_name)

