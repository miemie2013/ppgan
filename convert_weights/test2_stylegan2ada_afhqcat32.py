import paddle
import cv2

from ppgan.utils.filesystem import save
# from ppgan.models.generators.generator_styleganv2ada import ConstEncoderNetwork, StyleEncoderNetwork, StyleGANv2ADA_MappingNetwork, StyleGANv2ADA_SynthesisNetwork
from ppgan.models.generators.generator_styleganv2ada import StyleGANv2ADA_MappingNetwork, StyleGANv2ADA_SynthesisNetwork
from ppgan.models.discriminators.discriminator_styleganv2ada import StyleGANv2ADA_Discriminator

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
在 https://github.com/NVlabs/stylegan2-ada-pytorch 的training_loop.py的
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

处打断点，超参数即可看见。

'''
img_resolution = 32
img_channels = 3

synthesis_kwargs = dict(
    channel_base=32768,
    channel_max=512,
    num_fp16_res=4,
    conv_clamp=256,
)

z_dim = 512
c_dim = 0
w_dim = 512

mapping_kwargs = dict(
    num_layers=8,
)


'''
这里的self.noise_strength和原版形状不同，原版是空形状，不知道有没有影响。
            self.register_buffer('noise_const', paddle.randn([resolution, resolution]))
            # self.noise_strength = torch.nn.Parameter(torch.zeros([]))
            # 噪声强度（振幅）
            self.noise_strength = self.create_parameter([resolution, resolution],
                                                        default_initializer=paddle.nn.initializer.Constant(0.0))

'''


# class GeneratorV18(torch.nn.Module):
synthesis = StyleGANv2ADA_SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
num_ws = synthesis.num_ws
mapping = StyleGANv2ADA_MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=num_ws, **mapping_kwargs)

synthesis_ema = StyleGANv2ADA_SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
mapping_ema = StyleGANv2ADA_MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=num_ws, **mapping_kwargs)

channel_base = 32768
channel_max = 512
num_fp16_res = 4
conv_clamp = 256
epilogue_kwargs = dict(
    mbstd_group_size=8,
)
discriminator = StyleGANv2ADA_Discriminator(c_dim=c_dim,
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

mapping_std = mapping.state_dict()
synthesis_std = synthesis.state_dict()
mapping_ema_std = mapping_ema.state_dict()
synthesis_ema_std = synthesis_ema.state_dict()
discriminator_std = discriminator.state_dict()

mapping.eval()
synthesis.eval()
mapping_ema.eval()
synthesis_ema.eval()
discriminator.eval()


print('\nCopying...')



'''
在 https://github.com/xiezhy6/PASTA-GAN 的legacy.py的load_network_pkl()方法里插入torch.save()的代码获得权重，如下所示：

def load_network_pkl(f, force_fp16=False):
    data = _LegacyUnpickler(f).load()

    torch.save(data['G_ema'].state_dict(), "G_ema_256.pth")
    torch.save(data['D'].state_dict(), "D_256.pth")

'''
G_ema_ckpt_file = '../G_ema_00.pth'
G_ckpt_file = '../G_00.pth'
D_ckpt_file = '../D_00.pth'
save_name = '../styleganv2ada_32_afhqcat.pdparams'
G_ema_state_dict = torch.load(G_ema_ckpt_file, map_location=torch.device('cpu'))
G_state_dict = torch.load(G_ckpt_file, map_location=torch.device('cpu'))
D_state_dict = torch.load(D_ckpt_file, map_location=torch.device('cpu'))


synthesis_ema_dic = {}
mapping_ema_dic = {}
others = {}
for key, value in G_ema_state_dict.items():
    # if 'tracked' in key:
    #     continue
    if 'synthesis' in key:
        synthesis_ema_dic[key] = value.data.numpy()
    elif 'mapping' in key:
        mapping_ema_dic[key] = value.data.numpy()
    else:
        others[key] = value.data.numpy()

synthesis_dic = {}
mapping_dic = {}
for key, value in G_state_dict.items():
    # if 'tracked' in key:
    #     continue
    if 'synthesis' in key:
        synthesis_dic[key] = value.data.numpy()
    elif 'mapping' in key:
        mapping_dic[key] = value.data.numpy()
    else:
        others[key] = value.data.numpy()

discriminator_dic = {}
for key, value in D_state_dict.items():
    discriminator_dic[key] = value.data.numpy()

print()

for key in mapping_dic.keys():
    name2 = key
    w = mapping_dic[key]
    name2 = name2.replace('mapping.', '')
    if '.linear.weight' in key:
        w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
    print(key)
    copy(name2, w, mapping_std)
mapping.set_state_dict(mapping_std)

for key in mapping_ema_dic.keys():
    name2 = key
    w = mapping_ema_dic[key]
    name2 = name2.replace('mapping.', '')
    if '.linear.weight' in key:
        w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
    print(key)
    copy(name2, w, mapping_ema_std)
mapping_ema.set_state_dict(mapping_ema_std)



map = {}

conv_i = 0
torgb_i = 0
for block_idx, res in enumerate(synthesis.block_resolutions):
    in_channels = synthesis.channels_dict[res // 2] if res > 4 else 0
    is_last = synthesis.is_lasts[block_idx]
    architecture = synthesis.architectures[block_idx]

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


for key in synthesis_dic.keys():
    name2 = None
    for key2 in map.keys():
        if key2 in key:
            name2 = key.replace(key2, map[key2])
            name2 = name2.split('ynthesis.')[1]
            break
    w = synthesis_dic[key]
    if '.linear.weight' in key:
        print('.linear.weight...')
        w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
    if '.noise_strength' in key:
        print('noise_strength...')
        w = np.reshape(w, [1, ])
    # print(key)
    copy(name2, w, synthesis_std)
synthesis.set_state_dict(synthesis_std)

for key in synthesis_ema_dic.keys():
    name2 = None
    for key2 in map.keys():
        if key2 in key:
            name2 = key.replace(key2, map[key2])
            name2 = name2.split('ynthesis.')[1]
            break
    w = synthesis_ema_dic[key]
    if '.linear.weight' in key:
        print('.linear.weight...')
        w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
    if '.noise_strength' in key:
        print('noise_strength...')
        w = np.reshape(w, [1, ])
    # print(key)
    copy(name2, w, synthesis_ema_std)
synthesis_ema.set_state_dict(synthesis_ema_std)

for key in discriminator_dic.keys():
    name2 = key
    w = discriminator_dic[key]
    name2 = name2.replace('discriminator.', '')
    if '.linear.weight' in key:
        w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
    if '.noise_strength' in key:
        print()
        w = np.reshape(w, [1, ])
    print(key)
    copy(name2, w, discriminator_std)
discriminator.set_state_dict(discriminator_std)


class_idx = None
label = paddle.zeros([1, mapping.c_dim])
if mapping.c_dim != 0:
    if class_idx is None:
        print('Must specify class label with --class when using a conditional network')
    label[:, class_idx] = 1
else:
    if class_idx is not None:
        print('warn: --class=lbl ignored when running on an unconditional network')

# noise_mode = ['const', 'random', 'none']
noise_mode = 'const'
truncation_psi = 1.0
seeds = [85, 100, 75, 458, 1500]
# Generate images.
for seed_idx, seed in enumerate(seeds):
    print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
    z = paddle.to_tensor(np.random.RandomState(seed).randn(1, mapping.z_dim))

    # ws = mapping(z, label, truncation_psi=truncation_psi, truncation_cutoff=None)
    # img = synthesis(ws, noise_mode=noise_mode)
    ws = mapping_ema(z, label, truncation_psi=truncation_psi, truncation_cutoff=None)
    img = synthesis_ema(ws, noise_mode=noise_mode)

    img = (paddle.transpose(img, (0, 2, 3, 1)) * 127.5 + 128)
    img = paddle.clip(img, 0, 255)
    img = paddle.cast(img, dtype=paddle.uint8)
    img_rgb = img.numpy()[0]
    img_bgr = img_rgb[:, :, [2, 1, 0]]
    # cv2.imwrite(f'{outdir}/seed{seed:04d}.png', img_bgr)
    cv2.imwrite(f'seed{seed:04d}.png', img_bgr)





class Model(paddle.nn.Layer):
    def __init__(self, synthesis, mapping, synthesis_ema, mapping_ema, discriminator):
        super().__init__()
        self.nets = {}
        self.nets['synthesis'] = synthesis
        self.nets['mapping'] = mapping
        self.nets['synthesis_ema'] = synthesis_ema
        self.nets['mapping_ema'] = mapping_ema
        if discriminator is not None:
            self.nets['discriminator'] = discriminator

    def forward(self, x):
        return x


model = Model(synthesis, mapping, synthesis_ema, mapping_ema, discriminator=discriminator)

state_dicts = {}
for net_name, net in model.nets.items():
    state_dicts[net_name] = net.state_dict()

# paddle.save(state_dicts, save_name)
save(state_dicts, save_name)
print()









