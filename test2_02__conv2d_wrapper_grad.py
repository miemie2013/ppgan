
import torch
import numpy as np

from torch_utils.ops.conv2d_resample import _conv2d_wrapper


dic = {}
for batch_idx in range(20):
    w = torch.randn([3, 512, 1, 1])
    x = torch.randn([2, 512, 16, 16])
    w.requires_grad_(True)
    x.requires_grad_(True)

    py0 = 0
    px0 = 0
    groups = 1
    flip_weight = True
    y = _conv2d_wrapper(x=x, w=w, padding=[py0, px0], groups=groups, flip_weight=flip_weight)

    dy_dx = torch.autograd.grad(outputs=[y.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]
    dy_dw = torch.autograd.grad(outputs=[y.sum()], inputs=[w], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dy_dx'%batch_idx] = dy_dx.cpu().detach().numpy()
    dic['batch_%.3d.dy_dw'%batch_idx] = dy_dw.cpu().detach().numpy()
    dic['batch_%.3d.y'%batch_idx] = y.cpu().detach().numpy()
    dic['batch_%.3d.x'%batch_idx] = x.cpu().detach().numpy()
    dic['batch_%.3d.w'%batch_idx] = w.cpu().detach().numpy()
np.savez('02_grad', **dic)
print()
