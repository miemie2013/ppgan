
import torch
import numpy as np
from training.networks import SynthesisLayer


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
torch.save(synthesisLayer.state_dict(), "pytorch_synthesisLayer.pth")

batch_size = 2
ws = torch.randn([batch_size, 512])
ws.requires_grad_(True)
x = torch.randn([batch_size, 512, 4, 4])
x.requires_grad_(True)

dic = {}
pre_name = 'synthesisLayer'

y = synthesisLayer(x, ws, dic, pre_name + '.synthesisLayer', fused_modconv=fused_modconv, **layer_kwargs)
dy_dx = torch.autograd.grad(outputs=[y.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]
dy_dws = torch.autograd.grad(outputs=[y.sum()], inputs=[ws], create_graph=True, only_inputs=True)[0]
dic[pre_name + '.dy_dx'] = dy_dx.cpu().detach().numpy()
dic[pre_name + '.dy_dws'] = dy_dws.cpu().detach().numpy()
dic[pre_name + '.output'] = y.cpu().detach().numpy()
dic[pre_name + '.input0'] = x.cpu().detach().numpy()
dic[pre_name + '.input1'] = ws.cpu().detach().numpy()
np.savez('synthesisLayer_grad', **dic)
print()
