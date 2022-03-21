
import torch
import math
import numpy as np

import torch.nn.functional as F




scale = 1.0 / math.sqrt(3)

dic = {}
for batch_idx in range(8):
    x = torch.randn([2, 512, 16, 16])
    styles = torch.randn([2, 512])
    weight = torch.randn([512, 512, 3, 3])


    weight.requires_grad_(True)
    x.requires_grad_(True)
    styles.requires_grad_(True)
    batch_size = 2

    out_C, in_C, kH, kW = weight.shape


    style = styles.reshape((batch_size, 1, in_C, 1, 1))
    weight_pytorch = scale * weight * style
    demod = (weight_pytorch.square().sum(axis=[2, 3, 4]) + 1e-8).rsqrt()
    weight_pytorch = weight_pytorch * demod.reshape((batch_size, out_C, 1, 1, 1))

    out = weight_pytorch
    loss = torch.square(out)

    dloss_dstyles = torch.autograd.grad(outputs=[loss.sum()], inputs=[styles], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dloss_dstyles'%batch_idx] = dloss_dstyles.cpu().detach().numpy()
    dic['batch_%.3d.out'%batch_idx] = out.cpu().detach().numpy()
    dic['batch_%.3d.styles'%batch_idx] = styles.cpu().detach().numpy()
    dic['batch_%.3d.weight'%batch_idx] = weight.cpu().detach().numpy()
    dic['batch_%.3d.x'%batch_idx] = x.cpu().detach().numpy()
np.savez('18_grad', **dic)
print()
