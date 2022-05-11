import torch
import megengine as mge
import os
import numpy as np
from meg_networks import StyleGANv2ADA_SynthesisNetwork

w_dim = 512
# img_resolution = 512
# img_resolution = 128
img_resolution = 32
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


def copy(name, w, std):
    value2 = w
    value = std[name]
    value = value * 0 + value2
    std[name] = value

model_std = model.state_dict()

ckpt_file = '53.pth'
save_name = '53.pkl'
state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))


model_dic = {}
for key, value in state_dict.items():
    model_dic[key] = value.data.numpy()

for key in model_dic.keys():
    name2 = key
    w = model_dic[key]
    if '.linear.weight' in key:
        w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
    if '.noise_strength' in key:
        print()
        w = np.reshape(w, [1, ])
    print(key)
    copy(name2, w, model_std)
model.load_state_dict(model_std)

mge.save(model_std, save_name)
print(mge.__version__)

