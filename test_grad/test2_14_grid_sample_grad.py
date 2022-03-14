
import torch
import numpy as np

from torch_utils.ops import grid_sample_gradfix







dic = {}
for batch_idx in range(8):
    x = torch.randn([4, 3, 88, 88])
    x.requires_grad_(True)
    grid222 = torch.rand([4, 76, 76, 2])
    grid = grid222 * 1.98 - 0.99

    # x = torch.randn([2, 1, 2, 2])
    # x.requires_grad_(True)
    # grid222 = torch.rand([2, 2, 2, 2])
    # grid = grid222 * 1.98 - 0.99

    y = grid_sample_gradfix.grid_sample(x, grid)
    # y = torch.nn.functional.grid_sample(input=x, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    # y = torch.nn.functional.grid_sample(input=x, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    dloss_dx = torch.autograd.grad(outputs=[y.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dloss_dx'%batch_idx] = dloss_dx.cpu().detach().numpy()
    dic['batch_%.3d.out'%batch_idx] = y.cpu().detach().numpy()
    dic['batch_%.3d.x'%batch_idx] = x.cpu().detach().numpy()
    dic['batch_%.3d.grid'%batch_idx] = grid.cpu().detach().numpy()
np.savez('14_grad', **dic)
print()
