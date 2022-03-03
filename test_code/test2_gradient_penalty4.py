

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



# dic3 = np.load('affine_grid_data.npz')
# x = dic3['images']
x = np.random.normal(size=[2, 3, 1024, 64])


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

        align_corners = False

        grid = paddle.nn.functional.affine_grid(theta=G_inv[:, :2, :], out_shape=shape, align_corners=False)
        images2 = paddle.nn.functional.grid_sample(images, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=align_corners)

        return images2


class MyLayer2(nn.Layer):
    def __init__(self):
        super(MyLayer2, self).__init__()

    def forward(self, images):
        dic2 = np.load('affine_grid_data.npz')

        G_inv = dic2['G_inv']
        shape2 = dic2['shape']

        shape = []
        for sp in shape2:
            shape.append(sp)

        G_inv = paddle.to_tensor(G_inv, dtype='float32')

        align_corners = False

        grid = paddle.nn.functional.affine_grid(theta=G_inv[:, :2, :], out_shape=shape, align_corners=False)
        # images2 = paddle.nn.functional.grid_sample(images, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=align_corners)


        grid2 = grid
        grid_x = grid2[:, :, :, :1]
        grid_y = grid2[:, :, :, 1:]

        self.input_shape = images.shape
        N, C, ori_H, ori_W = self.input_shape
        _, out_H, out_W, _ = grid.shape

        '''
        看了Paddle的源代码Paddle/paddle/fluid/operators/grid_sampler_op.cu的
        template <typename T>
        static __forceinline__ __device__ T _unnormalize(T coord, int size,
                                                         bool align_corners) {
          if (align_corners) {
            return ((coord + 1.f) / 2) * (size - 1);
          } else {
            return ((coord + 1.f) * size - 1) / 2;
          }
        }
        其中coord是x或y，是-1到1之间的数值；size是ori_W或者ori_H。
        '''
        if align_corners:
            _xt = (grid_x + 1.0) * (float(ori_W) - 1.0) / 2.0
            _yt = (grid_y + 1.0) * (float(ori_H) - 1.0) / 2.0
        else:
            _xt = ((grid_x + 1.0) * float(ori_W) - 1.0) / 2.0
            _yt = ((grid_y + 1.0) * float(ori_H) - 1.0) / 2.0



        _y1 = paddle.floor(_yt)
        _x1 = paddle.floor(_xt)
        _y2 = _y1 + 1.0
        _x2 = _x1 + 1.0
        self._x1 = _x1.numpy()
        self._x2 = _x2.numpy()
        self._y1 = _y1.numpy()
        self._y2 = _y2.numpy()



        lh = _yt - _y1  # [N*out_H*out_W*kH*kW, 1]
        lw = _xt - _x1
        hh = 1 - lh
        hw = 1 - lw
        w1 = hh * hw
        w2 = hh * lw
        w3 = lh * hw
        w4 = lh * lw
        w1 = paddle.transpose(w1, perm=[0, 3, 1, 2])
        w2 = paddle.transpose(w2, perm=[0, 3, 1, 2])
        w3 = paddle.transpose(w3, perm=[0, 3, 1, 2])
        w4 = paddle.transpose(w4, perm=[0, 3, 1, 2])


        _x1 = _x1 / (ori_W - 1.0) * 2.0 - 1.0
        _y1 = _y1 / (ori_H - 1.0) * 2.0 - 1.0
        _x2 = _x2 / (ori_W - 1.0) * 2.0 - 1.0
        _y2 = _y2 / (ori_H - 1.0) * 2.0 - 1.0
        _x1y1 = paddle.concat([_x1, _y1], -1)
        _x1y2 = paddle.concat([_x1, _y2], -1)
        _x2y1 = paddle.concat([_x2, _y1], -1)
        _x2y2 = paddle.concat([_x2, _y2], -1)


        v1 = paddle.nn.functional.grid_sample(images, grid=_x1y1, mode='nearest', padding_mode='zeros', align_corners=align_corners)
        v2 = paddle.nn.functional.grid_sample(images, grid=_x2y1, mode='nearest', padding_mode='zeros', align_corners=align_corners)
        v3 = paddle.nn.functional.grid_sample(images, grid=_x1y2, mode='nearest', padding_mode='zeros', align_corners=align_corners)
        v4 = paddle.nn.functional.grid_sample(images, grid=_x2y2, mode='nearest', padding_mode='zeros', align_corners=align_corners)


        images3 = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4   # [N*out_H*out_W*kH*kW, in_C]
        self.temp_w1 = w1.numpy()
        self.temp_w2 = w2.numpy()
        self.temp_w3 = w3.numpy()
        self.temp_w4 = w4.numpy()
        # ddd = np.mean((images2.numpy() - images3.numpy()) ** 2)
        # print('ddd=%.6f' % ddd)

        # return images3
        return v1

    def get_grad(self, grad, place=None):

        temp_w1 = self.temp_w1
        temp_w1 = temp_w1.astype(np.float32)
        # pos = paddle.to_tensor(temp_w1, place=place)
        w1 = paddle.to_tensor(temp_w1)
        dy_dv1 = w1

        temp_w2 = self.temp_w2
        temp_w2 = temp_w2.astype(np.float32)
        # pos = paddle.to_tensor(temp_w1, place=place)
        w2 = paddle.to_tensor(temp_w2)
        dy_dv2 = w2

        temp_w3 = self.temp_w3
        temp_w3 = temp_w3.astype(np.float32)
        # pos = paddle.to_tensor(temp_w1, place=place)
        w3 = paddle.to_tensor(temp_w3)
        dy_dv3 = w3

        temp_w4 = self.temp_w4
        temp_w4 = temp_w4.astype(np.float32)
        # pos = paddle.to_tensor(temp_w1, place=place)
        w4 = paddle.to_tensor(temp_w4)
        dy_dv4 = w4

        N, out_C, out_H, out_W = grad.shape
        dy_dv1 = paddle.tile(dy_dv1, [1, out_C, 1, 1])
        dy_dv2 = paddle.tile(dy_dv2, [1, out_C, 1, 1])
        dy_dv3 = paddle.tile(dy_dv3, [1, out_C, 1, 1])
        dy_dv4 = paddle.tile(dy_dv4, [1, out_C, 1, 1])

        N, C, ori_H, ori_W = self.input_shape



        _x1 = paddle.to_tensor(self._x1.astype(np.int64))  # [N, out_H, out_W, 1]
        _y1 = paddle.to_tensor(self._y1.astype(np.int64))  # [N, out_H, out_W, 1]
        _x1 = paddle.reshape(_x1, (N*out_H*out_W, ))  # [N*out_H*out_W, ]
        _y1 = paddle.reshape(_y1, (N*out_H*out_W, ))  # [N*out_H*out_W, ]

        batch_offset = paddle.arange(0, N-1e-3, step=1.0, dtype=paddle.float32) * ori_H * ori_W
        batch_offset = paddle.cast(batch_offset, dtype=paddle.int64)
        batch_offset = paddle.reshape(batch_offset, (N, 1, 1))  # [N, 1, 1]
        batch_offset = paddle.tile(batch_offset, [1, out_H, out_W])  # [N, out_H, out_W]
        batch_offset = paddle.reshape(batch_offset, (N*out_H*out_W, ))  # [N*out_H*out_W, ]
        aaa1 = _x1 + _y1 * out_W + batch_offset

        dloss_dx = paddle.zeros((N*ori_H*ori_W, C), dtype=paddle.float32)  # [N*ori_H*ori_W, C]
        grad_temp = paddle.transpose(grad, perm=[0, 2, 3, 1])          # [N, out_H, out_W, out_C]
        grad_temp = paddle.reshape(grad_temp, (N*out_H*out_W, out_C))  # [N*out_H*out_W, out_C]

        dloss_dx = paddle.scatter(dloss_dx, aaa1, grad_temp, overwrite=False)
        dloss_dx = paddle.reshape(dloss_dx, (N, ori_H, ori_W, C))  # [N, ori_H, ori_W, C]
        dloss_dx = paddle.transpose(dloss_dx, perm=[0, 3, 1, 2])   # [N, C, ori_H, ori_W]


        return dloss_dx




mylayer = MyLayer()
mylayer.train()

mylayer2 = MyLayer2()
mylayer2.train()


x.stop_gradient = False
# y = mylayer(x)
y2 = mylayer2(x)

# ddd = np.mean((y2.numpy() - y.numpy()) ** 2)
# print('ddd=%.6f' % ddd)
#
# dydx = paddle.grad(
#     outputs=[y.sum()],
#     inputs=[x],
#     # create_graph=True,  # 最终loss里包含梯度，需要求梯度的梯度，所以肯定需要建立反向图。
#     create_graph=False,  # 最终
#     retain_graph=True)[0]

dddy2dx = paddle.grad(
    outputs=[y2.sum()],
    inputs=[x],
    create_graph=False,  # 最终
    retain_graph=True)[0]
dloss_dy = paddle.grad(
    outputs=[y2.sum()],
    inputs=[y2],
    create_graph=True,
    retain_graph=True)[0]
dy2dx = mylayer2.get_grad(dloss_dy)


# ddd = np.mean((dy2dx.numpy() - dydx.numpy()) ** 2)
# print('ddd=%.6f' % ddd)

# eee = dy2dx.numpy()
# eee2 = dddy2dx.numpy()

ddd = paddle.sum((dy2dx - dddy2dx) ** 2)

# ddd = np.sum((dy2dx.numpy() - dddy2dx.numpy()) ** 2)
print('ddd=%.6f' % ddd)

# print(dydx)
print()


