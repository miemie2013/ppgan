import paddle
import cv2

from ppgan.utils.filesystem import save
from ppgan.models.generators.generator_pastagan import ConstEncoderNetwork, StyleEncoderNetwork, MappingNetwork, SynthesisNetwork
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

synthesis_kwargs = dict(
    channel_base=16384,
    channel_max=512,
    num_fp16_res=3,
    conv_clamp=256,
    use_noise=True,
)

z_dim = 0
c_dim = 512
w_dim = 512
num_ws = 14

mapping_kwargs = dict(
    num_layers=1,
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
synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=num_ws, **mapping_kwargs)
const_encoding = ConstEncoderNetwork(input_nc=3 + 3, output_nc=512, ngf=64, n_downsampling=6)
style_encoding = StyleEncoderNetwork(input_nc=30 * 2, output_nc=512, ngf=64, n_downsampling=6)


style_encoding_std = style_encoding.state_dict()
const_encoding_std = const_encoding.state_dict()
mapping_std = mapping.state_dict()
synthesis_std = synthesis.state_dict()

style_encoding.eval()
const_encoding.eval()
mapping.eval()
synthesis.eval()


print('\nCopying...')



'''
在 https://github.com/xiezhy6/PASTA-GAN 的legacy.py的load_network_pkl()方法里插入torch.save()的代码获得权重，如下所示：

def load_network_pkl(f, force_fp16=False):
    data = _LegacyUnpickler(f).load()

    torch.save(data['G_ema'].state_dict(), "G_ema_256.pth")
    torch.save(data['D'].state_dict(), "D_256.pth")

'''
ckpt_file = 'G_ema_256.pth'
save_name = 'G_ema_256.pdparams'
state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))


synthesis_dic = {}
mapping_dic = {}
const_encoding_dic = {}
style_encoding_dic = {}
others = {}
for key, value in state_dict.items():
    # if 'tracked' in key:
    #     continue
    if 'synthesis' in key:
        synthesis_dic[key] = value.data.numpy()
    elif 'mapping' in key:
        mapping_dic[key] = value.data.numpy()
    elif 'const_encoding' in key:
        const_encoding_dic[key] = value.data.numpy()
    elif 'style_encoding' in key:
        style_encoding_dic[key] = value.data.numpy()
    else:
        others[key] = value.data.numpy()

print()

for key in style_encoding_dic.keys():
    name2 = key
    w = style_encoding_dic[key]
    name2 = name2.replace('style_encoding.', '')
    if '.linear.weight' in key:
        w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
    print(key)
    copy(name2, w, style_encoding_std)
style_encoding.set_state_dict(style_encoding_std)


for key in const_encoding_dic.keys():
    name2 = key
    w = const_encoding_dic[key]
    name2 = name2.replace('const_encoding.', '')
    if '.linear.weight' in key:
        w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
    print(key)
    copy(name2, w, const_encoding_std)
const_encoding.set_state_dict(const_encoding_std)

for key in mapping_dic.keys():
    name2 = key
    w = mapping_dic[key]
    name2 = name2.replace('mapping.', '')
    if '.linear.weight' in key:
        w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
    print(key)
    copy(name2, w, mapping_std)
mapping.set_state_dict(mapping_std)


for key in synthesis_dic.keys():
    name2 = key
    w = synthesis_dic[key]
    name2 = name2.replace('synthesis.', '')
    if '.linear.weight' in key:
        w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
    if '.noise_strength' in key:
        print()
        w = np.reshape(w, [1, ])
    print(key)
    copy(name2, w, synthesis_std)
synthesis.set_state_dict(synthesis_std)




dic2 = np.load('data.npz')
norm_img_c_tensor = dic2['norm_img_c_tensor']
retain_tensor = dic2['retain_tensor']
pose_tensor = dic2['pose_tensor']
gen_z = dic2['gen_z']

norm_img_c_tensor = paddle.to_tensor(norm_img_c_tensor)
retain_tensor = paddle.to_tensor(retain_tensor)
pose_tensor = paddle.to_tensor(pose_tensor)
gen_z = paddle.to_tensor(gen_z)

gen_c, cat_feat_list = style_encoding(norm_img_c_tensor, retain_tensor)


gen_c2 = dic2['gen_c']
cat_feats0 = dic2['cat_feats0']
cat_feats1 = dic2['cat_feats1']
cat_feats2 = dic2['cat_feats2']
cat_feats3 = dic2['cat_feats3']

ddd = np.sum((gen_c2 - gen_c.numpy())**2)
print('ddd=%.6f' % ddd)

ddd = np.sum((cat_feats0 - cat_feat_list[0].numpy())**2)
print('ddd=%.6f' % ddd)

ddd = np.sum((cat_feats1 - cat_feat_list[1].numpy())**2)
print('ddd=%.6f' % ddd)

ddd = np.sum((cat_feats2 - cat_feat_list[2].numpy())**2)
print('ddd=%.6f' % ddd)

ddd = np.sum((cat_feats3 - cat_feat_list[3].numpy())**2)
print('ddd=%.6f' % ddd)


pose_feat = const_encoding(pose_tensor)
pose_feat2 = dic2['pose_feat']

ddd = np.sum((pose_feat2 - pose_feat.numpy())**2)
print('ddd=%.6f' % ddd)



ws = mapping(gen_z, gen_c)
ws2 = dic2['ws']

ddd = np.sum((ws2 - ws.numpy())**2)
print('ddd=%.6f' % ddd)

cat_feats = {}
for cat_feat in cat_feat_list:
    h = cat_feat.shape[2]
    cat_feats[str(h)] = cat_feat




denorm_upper_clothes_tensor = dic2['denorm_upper_clothes_tensor']
denorm_upper_clothes_tensor = paddle.to_tensor(denorm_upper_clothes_tensor)
denorm_lower_clothes_tensor = dic2['denorm_lower_clothes_tensor']
denorm_lower_clothes_tensor = paddle.to_tensor(denorm_lower_clothes_tensor)
denorm_upper_mask_tensor = dic2['denorm_upper_mask_tensor']
denorm_upper_mask_tensor = paddle.to_tensor(denorm_upper_mask_tensor)
denorm_lower_mask_tensor = dic2['denorm_lower_mask_tensor']
denorm_lower_mask_tensor = paddle.to_tensor(denorm_lower_mask_tensor)


# 因为有噪声，所以不一定 等于0
gen_coarse_imgs, gen_imgs, _, _ = synthesis(ws, pose_feat, cat_feats, denorm_upper_clothes_tensor,
                                            denorm_lower_clothes_tensor, denorm_upper_mask_tensor,
                                            denorm_lower_mask_tensor)

gen_coarse_imgs2 = dic2['gen_coarse_imgs']
gen_imgs2 = dic2['gen_imgs']

ddd = np.sum((gen_coarse_imgs2 - gen_coarse_imgs.numpy())**2)
print('ddd=%.6f' % ddd)

ddd = np.sum((gen_imgs2 - gen_imgs.numpy())**2)
print('ddd=%.6f' % ddd)



print('\n\nfor 10 times...')
for i in range(10):
    gen_coarse_imgs, gen_imgs, _, _ = synthesis(ws, pose_feat, cat_feats, denorm_upper_clothes_tensor,
                                                denorm_lower_clothes_tensor, denorm_upper_mask_tensor,
                                                denorm_lower_mask_tensor)

    ddd = np.sum((gen_coarse_imgs2 - gen_coarse_imgs.numpy()) ** 2)
    print('ddd=%.6f' % ddd)

    ddd = np.sum((gen_imgs2 - gen_imgs.numpy()) ** 2)
    print('ddd=%.6f' % ddd)
    print()

    gen_imgs = gen_imgs.numpy()
    for ii in range(gen_imgs.shape[0]):
        gen_img = gen_imgs[ii]
        gen_img = (gen_img.transpose(1, 2, 0) + 1.0) * 127.5
        gen_img = gen_img[:, 32:224, [2, 1, 0]]
        gen_img = np.clip(gen_img, 0, 255)
        gen_img = gen_img.astype(np.uint8)

        save_path = 'aaaaa%.2d.jpg'%i
        # print(save_path)
        # print(gen_img.shape)
        cv2.imwrite(save_path, gen_img)



class Model(paddle.nn.Layer):
    def __init__(self, synthesis, mapping, const_encoding, style_encoding, discriminator):
        super().__init__()
        self.nets = {}
        self.nets['synthesis'] = synthesis
        self.nets['mapping'] = mapping
        self.nets['const_encoding'] = const_encoding
        self.nets['style_encoding'] = style_encoding
        if discriminator is not None:
            self.nets['discriminator'] = discriminator

    def forward(self, x):
        return x


model = Model(synthesis, mapping, const_encoding, style_encoding, discriminator=None)

state_dicts = {}
for net_name, net in model.nets.items():
    state_dicts[net_name] = net.state_dict()

# paddle.save(state_dicts, save_name)
save(state_dicts, save_name)
print()









