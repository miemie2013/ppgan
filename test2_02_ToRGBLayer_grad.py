
import torch
import numpy as np
from training.networks import ToRGBLayer


w_dim = 512
out_channels = 512
img_channels = 3
conv_clamp = 256
channels_last = False
fused_modconv = False

toRGBLayer = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                        conv_clamp=conv_clamp, channels_last=channels_last)
toRGBLayer.train()
torch.save(toRGBLayer.state_dict(), "pytorch_toRGBLayer.pth")

batch_size = 2
ws = torch.randn([batch_size, 512])
ws.requires_grad_(True)
x = torch.randn([batch_size, 512, 16, 16])
x.requires_grad_(True)

dic = {}
pre_name = 'toRGBLayer'

y = toRGBLayer(x, ws, dic, pre_name + '.toRGBLayer', fused_modconv=fused_modconv)
dy_dx = torch.autograd.grad(outputs=[y.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]
dy_dws = torch.autograd.grad(outputs=[y.sum()], inputs=[ws], create_graph=True, only_inputs=True)[0]
dic[pre_name + '.dy_dx'] = dy_dx.cpu().detach().numpy()
dic[pre_name + '.dy_dws'] = dy_dws.cpu().detach().numpy()
dic[pre_name + '.output'] = y.cpu().detach().numpy()
dic[pre_name + '.input0'] = x.cpu().detach().numpy()
dic[pre_name + '.input1'] = ws.cpu().detach().numpy()
np.savez('toRGBLayer_grad', **dic)
print()
