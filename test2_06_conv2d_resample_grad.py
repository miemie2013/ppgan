
import torch
import numpy as np

import torch.nn.functional as F
from torch_utils.ops.conv2d_resample import conv2d_resample


class Model(torch.nn.Module):
    def __init__(self, w_shape):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(w_shape))

    def forward(self, x, f, up, down, padding, groups, flip_weight, flip_filter):
        y = conv2d_resample(x, self.weight, f=f, up=up, down=down, padding=padding, groups=groups, flip_weight=flip_weight, flip_filter=flip_filter)
        return y


lr = 0.0001
dic = {}
batch_size = 2
for batch_idx in range(8):
    x_shape = [1, 512, 4, 4]
    w_shape = [512, 512, 3, 3]
    f_shape = [4, 4]
    up = 1
    down = 1
    padding = 1
    groups = 1
    flip_weight = True
    flip_filter = False

    x_shape = [1, 512, 4, 4]
    w_shape = [3, 512, 1, 1]
    f_shape = None
    up = 1
    down = 1
    padding = 0
    groups = 1
    flip_weight = True
    flip_filter = False

    # x_shape = [1, 512, 4, 4]
    # w_shape = [512, 512, 3, 3]
    # f_shape = [4, 4]
    # up = 2
    # down = 1
    # padding = 1
    # groups = 1
    # flip_weight = False
    # flip_filter = False

    # x_shape = [1, 512, 8, 8]
    # w_shape = [512, 512, 3, 3]
    # f_shape = [4, 4]
    # up = 1
    # down = 1
    # padding = 1
    # groups = 1
    # flip_weight = True
    # flip_filter = False

    # x_shape = [1, 512, 8, 8]
    # w_shape = [3, 512, 1, 1]
    # f_shape = None
    # up = 1
    # down = 1
    # padding = 0
    # groups = 1
    # flip_weight = True
    # flip_filter = False

    # x_shape = [1, 512, 8, 8]
    # w_shape = [512, 512, 3, 3]
    # f_shape = [4, 4]
    # up = 2
    # down = 1
    # padding = 1
    # groups = 1
    # flip_weight = False
    # flip_filter = False

    # x_shape = [1, 512, 16, 16]
    # w_shape = [512, 512, 3, 3]
    # f_shape = [4, 4]
    # up = 1
    # down = 1
    # padding = 1
    # groups = 1
    # flip_weight = True
    # flip_filter = False

    # x_shape = [1, 512, 16, 16]
    # w_shape = [3, 512, 1, 1]
    # f_shape = None
    # up = 1
    # down = 1
    # padding = 0
    # groups = 1
    # flip_weight = True
    # flip_filter = False

    # x_shape = [1, 512, 16, 16]
    # w_shape = [512, 512, 3, 3]
    # f_shape = [4, 4]
    # up = 2
    # down = 1
    # padding = 1
    # groups = 1
    # flip_weight = False
    # flip_filter = False

    # x_shape = [1, 512, 32, 32]
    # w_shape = [512, 512, 3, 3]
    # f_shape = [4, 4]
    # up = 1
    # down = 1
    # padding = 1
    # groups = 1
    # flip_weight = True
    # flip_filter = False

    # x_shape = [1, 512, 32, 32]
    # w_shape = [3, 512, 1, 1]
    # f_shape = None
    # up = 1
    # down = 1
    # padding = 0
    # groups = 1
    # flip_weight = True
    # flip_filter = False

    # x_shape = [1, 512, 32, 32]
    # w_shape = [512, 512, 3, 3]
    # f_shape = [4, 4]
    # up = 2
    # down = 1
    # padding = 1
    # groups = 1
    # flip_weight = False
    # flip_filter = False

    # x_shape = [1, 512, 64, 64]
    # w_shape = [512, 512, 3, 3]
    # f_shape = [4, 4]
    # up = 1
    # down = 1
    # padding = 1
    # groups = 1
    # flip_weight = True
    # flip_filter = False

    # x_shape = [1, 512, 64, 64]
    # w_shape = [3, 512, 1, 1]
    # f_shape = None
    # up = 1
    # down = 1
    # padding = 0
    # groups = 1
    # flip_weight = True
    # flip_filter = False

    # x_shape = [1, 512, 64, 64]
    # w_shape = [256, 512, 3, 3]
    # f_shape = [4, 4]
    # up = 2
    # down = 1
    # padding = 1
    # groups = 1
    # flip_weight = False
    # flip_filter = False

    # x_shape = [1, 256, 128, 128]
    # w_shape = [256, 256, 3, 3]
    # f_shape = [4, 4]
    # up = 1
    # down = 1
    # padding = 1
    # groups = 1
    # flip_weight = True
    # flip_filter = False

    # x_shape = [1, 256, 128, 128]
    # w_shape = [3, 256, 1, 1]
    # f_shape = None
    # up = 1
    # down = 1
    # padding = 0
    # groups = 1
    # flip_weight = True
    # flip_filter = False

    # x_shape = [1, 256, 128, 128]
    # w_shape = [128, 256, 3, 3]
    # f_shape = [4, 4]
    # up = 2
    # down = 1
    # padding = 1
    # groups = 1
    # flip_weight = False
    # flip_filter = False

    x_shape = [1, 128, 256, 256]
    w_shape = [128, 128, 3, 3]
    f_shape = [4, 4]
    up = 1
    down = 1
    padding = 1
    groups = 1
    flip_weight = True
    flip_filter = False

    # x_shape = [1, 128, 256, 256]
    # w_shape = [3, 128, 1, 1]
    # f_shape = None
    # up = 1
    # down = 1
    # padding = 0
    # groups = 1
    # flip_weight = True
    # flip_filter = False

    # x_shape = [1, 128, 256, 256]
    # w_shape = [64, 128, 3, 3]
    # f_shape = [4, 4]
    # up = 2
    # down = 1
    # padding = 1
    # groups = 1
    # flip_weight = False
    # flip_filter = False

    # x_shape = [1, 64, 512, 512]
    # w_shape = [64, 64, 3, 3]
    # f_shape = [4, 4]
    # up = 1
    # down = 1
    # padding = 1
    # groups = 1
    # flip_weight = True
    # flip_filter = False

    # x_shape = [1, 64, 512, 512]
    # w_shape = [3, 64, 1, 1]
    # f_shape = None
    # up = 1
    # down = 1
    # padding = 0
    # groups = 1
    # flip_weight = True
    # flip_filter = False





    # x_shape = [1, 512, 16, 16]
    # w_shape = [3, 512, 1, 1]
    # f_shape = [4, 4]
    # up = 1
    # down = 2
    # padding = 0
    # groups = 1
    # flip_weight = True
    # flip_filter = False

    x_shape[0] = batch_size
    x = torch.randn(x_shape)
    x.requires_grad_(True)
    if f_shape is None:
        f = None
    else:
        f = torch.randn(f_shape)
        # f.requires_grad_(True)

    if batch_idx == 0:
        model = Model(w_shape)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        torch.save(model.state_dict(), "model.pth")


    y = model(x, f=f, up=up, down=down, padding=padding, groups=groups, flip_weight=flip_weight, flip_filter=flip_filter)

    dy_dx = torch.autograd.grad(outputs=[y.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]
    dy_dw = torch.autograd.grad(outputs=[y.sum()], inputs=[model.weight], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dy_dx'%batch_idx] = dy_dx.cpu().detach().numpy()
    dic['batch_%.3d.dy_dw'%batch_idx] = dy_dw.cpu().detach().numpy()
    dic['batch_%.3d.y'%batch_idx] = y.cpu().detach().numpy()
    dic['batch_%.3d.x'%batch_idx] = x.cpu().detach().numpy()
    if f is not None:
        dic['batch_%.3d.f'%batch_idx] = f.cpu().detach().numpy()


    loss = y.sum() + dy_dx.sum() + dy_dw.sum()
    # loss = y.sum() + dy_dx.sum()
    # loss = y.sum() + dy_dw.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
np.savez('06_grad', **dic)
print()
