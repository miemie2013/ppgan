
import torch
import numpy as np








dic = {}
for batch_idx in range(8):
    x = torch.randn([2, 3, 1024])
    x.requires_grad_(True)
    C = torch.rand([2, 4, 4])

    y = C[:, :3, :3] @ x + C[:, :3, 3:]

    dloss_dx = torch.autograd.grad(outputs=[y.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dloss_dx'%batch_idx] = dloss_dx.cpu().detach().numpy()
    dic['batch_%.3d.out'%batch_idx] = y.cpu().detach().numpy()
    dic['batch_%.3d.x'%batch_idx] = x.cpu().detach().numpy()
    dic['batch_%.3d.C'%batch_idx] = C.cpu().detach().numpy()
np.savez('15_grad', **dic)
print()
