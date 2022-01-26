import paddle

from ppgan.models.generators.generator_pastagan import ConstEncoderNetwork, StyleEncoderNetwork
# from ppgan.models.pastagan_model import PastaGANModel

import numpy as np
import torch


# class GeneratorV18(torch.nn.Module):

# synthesis = SynthesisNetworkV18(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
# num_ws = synthesis.num_ws
# mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
# const_encoding = ConstEncoderNetwork(input_nc=3 + 3, output_nc=512, ngf=64, n_downsampling=6)
style_encoding = StyleEncoderNetwork(input_nc=30 * 2, output_nc=512, ngf=64, n_downsampling=6)



dic2 = np.load('data.npz')
norm_img_c_tensor = dic2['norm_img_c_tensor']
retain_tensor = dic2['retain_tensor']
norm_img_c_tensor = paddle.to_tensor(norm_img_c_tensor)
retain_tensor = paddle.to_tensor(retain_tensor)


ckpt_file = 'G_ema_256.pth'
state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))


gen_c, cat_feat_list = style_encoding(norm_img_c_tensor, retain_tensor)

gen_c2 = dic2['gen_c']
cat_feats0 = dic2['cat_feats0']
cat_feats1 = dic2['cat_feats1']
cat_feats2 = dic2['cat_feats2']
cat_feats3 = dic2['cat_feats3']

# ddd = np.sum((aaaaaaaaaaa2 - aaaaaaaaaaa.numpy())**2)
# print('ddd=%.6f' % ddd)


# torch.save(ckpt_state, args.output_ckpt)
print()









