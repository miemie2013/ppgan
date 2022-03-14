import torch
import paddle
import os
import numpy as np
from ppgan.models.discriminators.discriminator_styleganv2ada import StyleGANv2ADA_Discriminator

c_dim = 0
w_dim = 512
# img_resolution = 512
# img_resolution = 128
img_resolution = 32
img_channels = 3
channel_base = 32768
channel_max = 512
num_fp16_res = 4
conv_clamp = 256
epilogue_kwargs = dict(
    mbstd_group_size=8,
)


batch_size = 2
x_shape = [batch_size, img_channels, img_resolution, img_resolution]
lr = 0.0001

model = StyleGANv2ADA_Discriminator(c_dim=c_dim,
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

ckpt_file = '54.pth'
save_name = '54.pdparams'
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

