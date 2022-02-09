import paddle
import cv2

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


'''
在 https://github.com/xiezhy6/PASTA-GAN 的legacy.py的load_network_pkl()方法里data = _LegacyUnpickler(f).load()
后面的代码打断点，超参数全部在data里。

def load_network_pkl(f, force_fp16=False):
    data = _LegacyUnpickler(f).load()

'''
img_resolution = 256
img_channels = 3

c_dim = 512

block_kwargs = dict(
)

mapping_kwargs = dict(
)

epilogue_kwargs = dict(
    mbstd_group_size=4,
)
channel_base = 16384
channel_max = 512
num_fp16_res = 3
conv_clamp = 256


'''
这里的self.noise_strength和原版形状不同，原版是空形状，不知道有没有影响。
            self.register_buffer('noise_const', paddle.randn([resolution, resolution]))
            # self.noise_strength = torch.nn.Parameter(torch.zeros([]))
            # 噪声强度（振幅）
            self.noise_strength = self.create_parameter([resolution, resolution],
                                                        default_initializer=paddle.nn.initializer.Constant(0.0))

'''


# class GeneratorV18(torch.nn.Module):
discriminator = PastaGANDiscriminator(c_dim, img_resolution, img_channels,
                                      channel_base=channel_base, channel_max=channel_max,
                                      num_fp16_res=num_fp16_res, conv_clamp=conv_clamp,
                                      block_kwargs=block_kwargs, mapping_kwargs=mapping_kwargs,
                                      epilogue_kwargs=epilogue_kwargs)


discriminator_std = discriminator.state_dict()

discriminator.eval()


print('\nCopying...')



'''
在 https://github.com/xiezhy6/PASTA-GAN 的legacy.py的load_network_pkl()方法里插入torch.save()的代码获得权重，如下所示：

def load_network_pkl(f, force_fp16=False):
    data = _LegacyUnpickler(f).load()

    torch.save(data['G_ema'].state_dict(), "G_ema_256.pth")
    torch.save(data['D'].state_dict(), "D_256.pth")

'''
ckpt_file = 'D_temp_256.pth'
save_name = 'D_temp_256.pdparams'
state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))


all_dic = {}
for key, value in state_dict.items():
    all_dic[key] = value.data.numpy()

print()

for key in all_dic.keys():
    name2 = key
    w = all_dic[key]
    # name2 = name2.replace('synthesis.', '')
    if '.linear.weight' in key:
        print('wwwwwwwwwwwwwwwwwwwwwwwwwwww')
        w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
    if '.noise_strength' in key:
        print('qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq')
        w = np.reshape(w, [1, ])
    # print(key)
    copy(name2, w, discriminator_std)
discriminator.set_state_dict(discriminator_std)

paddle.save(discriminator_std, save_name)






