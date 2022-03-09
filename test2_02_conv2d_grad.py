
import torch
import numpy as np

import torch.nn.functional as F







dic = {}
for batch_idx in range(20):
    # kernel_size = 1
    # stride = 1
    # padding = 0
    # output_padding = 0
    # dilation = 1
    # groups = 1

    # kernel_size = 1
    # stride = 2
    # padding = 0
    # output_padding = 0
    # dilation = 1
    # groups = 2

    # kernel_size = 3
    # stride = 1
    # padding = 0
    # output_padding = 0
    # dilation = 1
    # groups = 1

    # kernel_size = 3
    # stride = 2
    # padding = 0
    # output_padding = 0
    # dilation = 1
    # groups = 2

    # kernel_size = 3
    # stride = 1
    # padding = 1
    # output_padding = 0
    # dilation = 1
    # groups = 2

    # kernel_size = 3
    # stride = 2
    # padding = 1
    # output_padding = 0
    # dilation = 1
    # groups = 1

    kernel_size = 3
    stride = 2
    padding = 1
    output_padding = 0
    dilation = 1
    groups = 2

    w = torch.randn([512, 256 // groups, kernel_size, kernel_size])
    x = torch.randn([2, 256, 16, 16])
    # w = torch.randn([4, 4 // groups, kernel_size, kernel_size])
    # x = torch.randn([1, 4, 3, 3])
    w.requires_grad_(True)
    x.requires_grad_(True)
    bias = None

    y = F.conv2d(input=x, weight=w, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    dy_dx = torch.autograd.grad(outputs=[y.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]
    dy_dw = torch.autograd.grad(outputs=[y.sum()], inputs=[w], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dy_dx'%batch_idx] = dy_dx.cpu().detach().numpy()
    dic['batch_%.3d.dy_dw'%batch_idx] = dy_dw.cpu().detach().numpy()
    dic['batch_%.3d.y'%batch_idx] = y.cpu().detach().numpy()
    dic['batch_%.3d.x'%batch_idx] = x.cpu().detach().numpy()
    dic['batch_%.3d.w'%batch_idx] = w.cpu().detach().numpy()
np.savez('02_grad', **dic)
print()
