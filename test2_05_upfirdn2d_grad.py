
import torch
import numpy as np

import torch.nn.functional as F
from torch_utils.ops.upfirdn2d import upfirdn2d


dic = {}
batch_size = 2
for batch_idx in range(8):
    x_shape = [1, 512, 9, 9]
    f_shape = [4, 4]
    up = 1
    down = 1
    padding = [1, 1, 1, 1]
    flip_filter = False
    gain = 4

    # x_shape = [1, 3, 4, 4]
    # f_shape = [4, 4]
    # up = 2
    # down = 1
    # padding = [2, 1, 2, 1]
    # flip_filter = False
    # gain = 4

    # x_shape = [1, 512, 17, 17]
    # f_shape = [4, 4]
    # up = 1
    # down = 1
    # padding = [1, 1, 1, 1]
    # flip_filter = False
    # gain = 4

    # x_shape = [1, 3, 8, 8]
    # f_shape = [4, 4]
    # up = 2
    # down = 1
    # padding = [2, 1, 2, 1]
    # flip_filter = False
    # gain = 4

    # x_shape = [1, 512, 33, 33]
    # f_shape = [4, 4]
    # up = 1
    # down = 1
    # padding = [1, 1, 1, 1]
    # flip_filter = False
    # gain = 4

    # x_shape = [1, 3, 16, 16]
    # f_shape = [4, 4]
    # up = 2
    # down = 1
    # padding = [2, 1, 2, 1]
    # flip_filter = False
    # gain = 4

    # x_shape = [1, 512, 65, 65]
    # f_shape = [4, 4]
    # up = 1
    # down = 1
    # padding = [1, 1, 1, 1]
    # flip_filter = False
    # gain = 4

    # x_shape = [1, 3, 32, 32]
    # f_shape = [4, 4]
    # up = 2
    # down = 1
    # padding = [2, 1, 2, 1]
    # flip_filter = False
    # gain = 4

    # x_shape = [1, 256, 129, 129]
    # f_shape = [4, 4]
    # up = 1
    # down = 1
    # padding = [1, 1, 1, 1]
    # flip_filter = False
    # gain = 4

    # x_shape = [1, 3, 64, 64]
    # f_shape = [4, 4]
    # up = 2
    # down = 1
    # padding = [2, 1, 2, 1]
    # flip_filter = False
    # gain = 4

    # x_shape = [1, 128, 257, 257]
    # f_shape = [4, 4]
    # up = 1
    # down = 1
    # padding = [1, 1, 1, 1]
    # flip_filter = False
    # gain = 4

    # x_shape = [1, 3, 128, 128]
    # f_shape = [4, 4]
    # up = 2
    # down = 1
    # padding = [2, 1, 2, 1]
    # flip_filter = False
    # gain = 4

    # x_shape = [1, 64, 513, 513]
    # f_shape = [4, 4]
    # up = 1
    # down = 1
    # padding = [1, 1, 1, 1]
    # flip_filter = False
    # gain = 4

    # x_shape = [1, 3, 256, 256]
    # f_shape = [4, 4]
    # up = 2
    # down = 1
    # padding = [2, 1, 2, 1]
    # flip_filter = False
    # gain = 4



    x_shape[0] = batch_size
    f = torch.randn(f_shape)
    x = torch.randn(x_shape)
    x.requires_grad_(True)

    y = upfirdn2d(x, f, up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain)

    dy_dx = torch.autograd.grad(outputs=[y.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dy_dx'%batch_idx] = dy_dx.cpu().detach().numpy()
    dic['batch_%.3d.y'%batch_idx] = y.cpu().detach().numpy()
    dic['batch_%.3d.x'%batch_idx] = x.cpu().detach().numpy()
    dic['batch_%.3d.f'%batch_idx] = f.cpu().detach().numpy()
np.savez('05_grad', **dic)
print()
