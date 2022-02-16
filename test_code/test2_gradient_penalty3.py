

import numpy as np
import torch
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle import ParamAttr
from paddle.nn.initializer import Constant
from paddle.vision.ops import DeformConv2D



'''
测试梯度惩罚

'''



# h = 255
# w = 255

h = 256
w = 256

# h = 257
# w = 257

# h = 256
# w = 257

# x = np.random.normal(size=[2, 3, h, w])
# x = paddle.to_tensor(x)
# x = paddle.cast(x, paddle.float32)

dic3 = np.load('affine_grid_data.npz')
x = dic3['images']
x = paddle.to_tensor(x)
x = paddle.cast(x, paddle.float32)


class MyLayer(nn.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()

    def forward(self, images):
        dic2 = np.load('affine_grid_data.npz')

        G_inv = dic2['G_inv']
        shape2 = dic2['shape']

        shape = []
        for sp in shape2:
            shape.append(sp)

        G_inv = paddle.to_tensor(G_inv, dtype='float32')

        grid = paddle.nn.functional.affine_grid(theta=G_inv[:, :2, :], out_shape=shape, align_corners=False)
        # out1 = paddle.mean(grid, axis=[1, 2, 3])
        # out2 = paddle.mean(images, axis=[1, 2, 3])
        # out = out1 + out2
        # images = paddle.nn.functional.grid_sample(images, grid=grid, mode='bilinear', padding_mode='zeros',
        #                                           align_corners=False)
        N, C, H, W = images.shape

        grid_x = grid[:, :, :, :1]
        grid_y = grid[:, :, :, 1:]
        # grid_x = grid[:, :2, :2, :1]
        # grid_y = grid[:, :2, :2, 1:]

        _xt = (grid_x + 1.0) * float(W) / 2.0
        _yt = (grid_y + 1.0) * float(H) / 2.0

        _yt = paddle.reshape(_yt, (-1, 1))  # [N*out_H*out_W, 1]
        _xt = paddle.reshape(_xt, (-1, 1))  # [N*out_H*out_W, 1]


        print()

        # 为了避免使用for循环遍历每一张图片，还要给y坐标（代表行号）加上图片的偏移来一次性抽取出更兴趣的像素。
        # row_offset = torch.arange(0, N, dtype=torch.float32, device=dcn_weight.device) * pad_x_H  # [N, ]
        # row_offset = row_offset[:, np.newaxis, np.newaxis].repeat((1, out_H * out_W * kH * kW, 1))  # [N, out_H*out_W*kH*kW, 1]
        # row_offset = torch.reshape(row_offset, (N * out_H * out_W * kH * kW, 1))  # [N*out_H*out_W*kH*kW, 1]
        # row_offset.requires_grad = False
        # _yt += row_offset

        _y1 = paddle.floor(_yt)
        _x1 = paddle.floor(_xt)
        _y2 = _y1 + 1.0
        _x2 = _x1 + 1.0
        _y1x1 = paddle.concat([_y1, _x1], -1)
        _y1x2 = paddle.concat([_y1, _x2], -1)
        _y2x1 = paddle.concat([_y2, _x1], -1)
        _y2x2 = paddle.concat([_y2, _x2], -1)

        pad_x = images
        N, in_C, pad_x_H, pad_x_W = pad_x.shape
        pad_x = paddle.transpose(pad_x, [0, 2, 3, 1])  # [N, pad_x_H, pad_x_W, C]
        pad_x = paddle.reshape(pad_x, (N*pad_x_H, pad_x_W, in_C))  # [N*pad_x_H, pad_x_W, C]

        _y1x1_int = paddle.cast(_y1x1, dtype=paddle.int64)   # [N*out_H*out_W*kH*kW, 2]
        v1 = paddle.gather_nd(pad_x, _y1x1_int)   # [N*out_H*out_W*kH*kW, in_C]
        _y1x2_int = paddle.cast(_y1x2, dtype=paddle.int64)   # [N*out_H*out_W*kH*kW, 2]
        v2 = paddle.gather_nd(pad_x, _y1x2_int)   # [N*out_H*out_W*kH*kW, in_C]
        _y2x1_int = paddle.cast(_y2x1, dtype=paddle.int64)   # [N*out_H*out_W*kH*kW, 2]
        v3 = paddle.gather_nd(pad_x, _y2x1_int)   # [N*out_H*out_W*kH*kW, in_C]
        _y2x2_int = paddle.cast(_y2x2, dtype=paddle.int64)   # [N*out_H*out_W*kH*kW, 2]
        v4 = paddle.gather_nd(pad_x, _y2x2_int)   # [N*out_H*out_W*kH*kW, in_C]

        lh = _yt - _y1  # [N*out_H*out_W*kH*kW, 1]
        lw = _xt - _x1
        hh = 1 - lh
        hw = 1 - lw
        w1 = hh * hw
        w2 = hh * lw
        w3 = lh * hw
        w4 = lh * lw
        value = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4   # [N*out_H*out_W*kH*kW, in_C]

        return value

mylayer = MyLayer()
mylayer.train()


x.stop_gradient = False
y = mylayer(x)


dydx = paddle.grad(
    outputs=[y.sum()],
    inputs=[x],
    create_graph=True,  # 最终loss里包含梯度，需要求梯度的梯度，所以肯定需要建立反向图。
    retain_graph=True)[0]

print(dydx)



