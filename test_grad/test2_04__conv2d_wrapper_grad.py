
import torch
import numpy as np

import torch.nn.functional as F
from torch_utils.ops.conv2d_resample import _conv2d_wrapper


class Model(torch.nn.Module):
    def __init__(self, w_shape):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(w_shape))

    def forward(self, x, stride, padding, groups, transpose, flip_weight):
        y = _conv2d_wrapper(x=x, w=self.weight, stride=stride, padding=padding, groups=groups, transpose=transpose, flip_weight=flip_weight)
        return y


lr = 0.0001
dic = {}
batch_size = 2
for batch_idx in range(8):
    x_shape = [1, 8, 1, 1]
    w_shape = [4, 8, 1, 1]
    stride = 1
    padding = [0, 0]
    groups = 1
    transpose = False
    flip_weight = False

    x_shape = [1, 8, 1, 1]
    w_shape = [16, 8, 1, 1]
    stride = 1
    padding = [0, 0]
    groups = 1
    transpose = False
    flip_weight = True

    x_shape = [1, 8, 1, 1]
    w_shape = [16, 4, 1, 1]
    stride = 1
    padding = [0, 0]
    groups = 2
    transpose = False
    flip_weight = False

    x_shape = [1, 512, 4, 4]
    w_shape = [512, 512, 3, 3]
    stride = 1
    padding = [1, 1]
    groups = 1
    transpose = False
    flip_weight = True

    x_shape = [1, 512, 4, 4]
    w_shape = [3, 512, 1, 1]
    stride = 1
    padding = [0, 0]
    groups = 1
    transpose = False
    flip_weight = True

    x_shape = [1, 512, 4, 4]
    w_shape = [512, 512, 3, 3]
    stride = 2
    padding = [0, 0]
    groups = 1
    transpose = True
    flip_weight = True

    x_shape = [1, 512, 8, 8]
    w_shape = [512, 512, 3, 3]
    stride = 1
    padding = [1, 1]
    groups = 1
    transpose = False
    flip_weight = True

    # x_shape = [1, 512, 8, 8]
    # w_shape = [3, 512, 1, 1]
    # stride = 1
    # padding = [0, 0]
    # groups = 1
    # transpose = False
    # flip_weight = True

    # x_shape = [1, 512, 8, 8]
    # w_shape = [512, 512, 3, 3]
    # stride = 2
    # padding = [0, 0]
    # groups = 1
    # transpose = True
    # flip_weight = True

    # x_shape = [1, 512, 16, 16]
    # w_shape = [512, 512, 3, 3]
    # stride = 1
    # padding = [1, 1]
    # groups = 1
    # transpose = False
    # flip_weight = True

    # x_shape = [1, 512, 16, 16]
    # w_shape = [3, 512, 1, 1]
    # stride = 1
    # padding = [0, 0]
    # groups = 1
    # transpose = False
    # flip_weight = True

    # x_shape = [1, 512, 16, 16]
    # w_shape = [512, 512, 3, 3]
    # stride = 2
    # padding = [0, 0]
    # groups = 1
    # transpose = True
    # flip_weight = True

    # x_shape = [1, 512, 32, 32]
    # w_shape = [512, 512, 3, 3]
    # stride = 1
    # padding = [1, 1]
    # groups = 1
    # transpose = False
    # flip_weight = True

    # x_shape = [1, 512, 32, 32]
    # w_shape = [3, 512, 1, 1]
    # stride = 1
    # padding = [0, 0]
    # groups = 1
    # transpose = False
    # flip_weight = True

    # x_shape = [1, 512, 32, 32]
    # w_shape = [512, 512, 3, 3]
    # stride = 2
    # padding = [0, 0]
    # groups = 1
    # transpose = True
    # flip_weight = True

    # x_shape = [1, 512, 64, 64]
    # w_shape = [512, 512, 3, 3]
    # stride = 1
    # padding = [1, 1]
    # groups = 1
    # transpose = False
    # flip_weight = True

    # x_shape = [1, 512, 64, 64]
    # w_shape = [3, 512, 1, 1]
    # stride = 1
    # padding = [0, 0]
    # groups = 1
    # transpose = False
    # flip_weight = True

    # x_shape = [1, 512, 64, 64]
    # w_shape = [512, 256, 3, 3]
    # stride = 2
    # padding = [0, 0]
    # groups = 1
    # transpose = True
    # flip_weight = True

    # x_shape = [1, 256, 128, 128]
    # w_shape = [256, 256, 3, 3]
    # stride = 1
    # padding = [1, 1]
    # groups = 1
    # transpose = False
    # flip_weight = True

    # x_shape = [1, 256, 128, 128]
    # w_shape = [3, 256, 1, 1]
    # stride = 1
    # padding = [0, 0]
    # groups = 1
    # transpose = False
    # flip_weight = True

    x_shape = [1, 256, 128, 128]
    w_shape = [256, 128, 3, 3]
    stride = 2
    padding = [0, 0]
    groups = 1
    transpose = True
    flip_weight = True

    x_shape = [1, 128, 256, 256]
    w_shape = [128, 128, 3, 3]
    stride = 1
    padding = [1, 1]
    groups = 1
    transpose = False
    flip_weight = True

    x_shape = [1, 128, 256, 256]
    w_shape = [3, 128, 1, 1]
    stride = 1
    padding = [0, 0]
    groups = 1
    transpose = False
    flip_weight = True

    x_shape = [1, 128, 256, 256]
    w_shape = [128, 64, 3, 3]
    stride = 2
    padding = [0, 0]
    groups = 1
    transpose = True
    flip_weight = True

    x_shape = [1, 64, 512, 512]
    w_shape = [64, 64, 3, 3]
    stride = 1
    padding = [1, 1]
    groups = 1
    transpose = False
    flip_weight = True

    x_shape = [1, 64, 512, 512]
    w_shape = [3, 64, 1, 1]
    stride = 1
    padding = [0, 0]
    groups = 1
    transpose = False
    flip_weight = True

    x_shape = [1, 512, 9, 9]
    w_shape = [512, 512, 3, 3]
    stride = 2
    padding = 0
    groups = 1
    transpose = False
    flip_weight = True



    x_shape[0] = batch_size
    x = torch.randn(x_shape)
    x.requires_grad_(True)

    if batch_idx == 0:
        model = Model(w_shape)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        torch.save(model.state_dict(), "model.pth")

    y = model(x=x, stride=stride, padding=padding, groups=groups, transpose=transpose, flip_weight=flip_weight)

    # loss = torch.sigmoid(y)
    loss = y

    grads = torch.autograd.grad(outputs=[loss.sum()], inputs=[x, model.weight], create_graph=True, only_inputs=True)
    dloss_dx, dloss_dw = grads

    dic['batch_%.3d.dloss_dx'%batch_idx] = dloss_dx.cpu().detach().numpy()
    dic['batch_%.3d.dloss_dw'%batch_idx] = dloss_dw.cpu().detach().numpy()
    dic['batch_%.3d.y'%batch_idx] = y.cpu().detach().numpy()
    dic['batch_%.3d.x'%batch_idx] = x.cpu().detach().numpy()


    # loss = loss.sum() + dloss_dx.sum() + dloss_dw.sum()
    loss = loss.sum() + dloss_dx.sum()
    # loss = loss.sum() + dloss_dw.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
np.savez('04_grad', **dic)
print()
