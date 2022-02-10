

import numpy as np
import torch
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

'''
测试梯度惩罚

'''



# h = 255
# w = 255

# h = 256
# w = 256

h = 257
w = 257

# h = 256
# w = 257

x = np.random.normal(size=[8, 3, h, w])
x = paddle.to_tensor(x)
x = paddle.cast(x, paddle.float32)


class MyLayer(nn.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()
        self.conv1 = nn.Conv2D(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2D(8, 8, kernel_size=1, padding=0)
        self.skip = nn.Conv2D(3, 8, kernel_size=1, padding=0)

        stride2_kernel = np.zeros((8, 1, 2, 2), dtype=np.float32)
        stride2_kernel[:, :, 0, 0] = 1.0
        # self.stride2_kernel = paddle.to_tensor(stride2_kernel, place=place)
        self.stride2_kernel = paddle.to_tensor(stride2_kernel)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        y = self.skip(input)
        # out = x.add_(y)  # 因为add_()没有实现二阶梯度，所以用其它等价实现。
        downx = 2
        downy = downx
        # 因为:: （paddle.strided_slice()）没有实现二阶梯度，所以用其它等价实现。
        x3 = x[:, :, ::downy, ::downx]  # RuntimeError: (NotFound) The Op strided_slice_grad doesn't have any grad op.
        y3 = y[:, :, ::downy, ::downx]  # RuntimeError: (NotFound) The Op strided_slice_grad doesn't have any grad op.
        N, C, H, W = x.shape
        pad_height_bottom = 0
        pad_width_right = 0
        if H % 2 == 1:
            pad_height_bottom = 1
            pad_width_right = 1
        x2 = F.conv2d(x, self.stride2_kernel, bias=None, stride=2, groups=8,
                      padding=[[0, 0], [0, 0], [0, pad_height_bottom], [0, pad_width_right]])
        y2 = F.conv2d(y, self.stride2_kernel, bias=None, stride=2, groups=8,
                      padding=[[0, 0], [0, 0], [0, pad_height_bottom], [0, pad_width_right]])
        ddd = np.sum((x3.numpy() - x2.numpy()) ** 2)
        print('ddd=%.6f' % ddd)
        ddd = np.sum((y3.numpy() - y2.numpy()) ** 2)
        print('ddd=%.6f' % ddd)


        # x = F.conv2d(x, self.stride2_kernel, bias=None, stride=2, padding=0, groups=8)
        # y = F.conv2d(y, self.stride2_kernel, bias=None, stride=2, padding=0, groups=8)

        out = x + y
        return out

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



