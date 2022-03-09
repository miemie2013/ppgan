
import torch
import numpy as np

import torch.nn.functional as F







dic = {}
for batch_idx in range(20):
    w = torch.randn([8, 12, 12, 16, 16])
    w.requires_grad_(True)

    # rsqrt()表示平方根的倒数
    y = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
    # y = (w.sum(dim=[2, 3, 4]) + 10000).rsqrt()
    # y = (w + 100).rsqrt()

    dy_dw = torch.autograd.grad(outputs=[y.sum()], inputs=[w], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dy_dw'%batch_idx] = dy_dw.cpu().detach().numpy()
    dic['batch_%.3d.y'%batch_idx] = y.cpu().detach().numpy()
    dic['batch_%.3d.w'%batch_idx] = w.cpu().detach().numpy()
np.savez('08_grad', **dic)
print()
