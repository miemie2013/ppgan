
import torch
import numpy as np

import torch.nn.functional as F







dic = {}
for batch_idx in range(8):
    # w = torch.randn([11, 12, 1, 1])
    # x = torch.randn([11, 12, 1, 1])

    # w = torch.randn([11, 12, 13, 16])
    # x = torch.randn([11, 12, 1, 1])

    # w = torch.randn([11, 12, 13, 1])
    # x = torch.randn([11, 12, 1, 16])

    # w = torch.randn([11, 12, 13, 1])
    # x = torch.randn([1, 1, 1, 16])

    w = torch.randn([11, 1, 13, 1])
    x = torch.randn([1, 12, 1, 16])


    w.requires_grad_(True)
    x.requires_grad_(True)

    y = (x * w).sum(dim=[2, 3])

    dy_dw = torch.autograd.grad(outputs=[y.sum()], inputs=[w], create_graph=True, only_inputs=True)[0]
    dy_dx = torch.autograd.grad(outputs=[y.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dy_dw'%batch_idx] = dy_dw.cpu().detach().numpy()
    dic['batch_%.3d.dy_dx'%batch_idx] = dy_dx.cpu().detach().numpy()
    dic['batch_%.3d.y'%batch_idx] = y.cpu().detach().numpy()
    dic['batch_%.3d.w'%batch_idx] = w.cpu().detach().numpy()
    dic['batch_%.3d.x'%batch_idx] = x.cpu().detach().numpy()
np.savez('09_grad', **dic)
print()
