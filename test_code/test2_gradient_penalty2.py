

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

x = np.random.normal(size=[8, 3, h, w])
x = paddle.to_tensor(x)
x = paddle.cast(x, paddle.float32)




class MyDCNv2(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 weight_attr=None,
                 bias_attr=None):
        super(MyDCNv2, self).__init__()
        assert weight_attr is not False, "weight_attr should not be False in Conv."
        self.weight_attr = weight_attr
        self.bias_attr = bias_attr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups.")
        self.groups = groups

        filter_shape = [out_channels, in_channels // groups, kernel_size, kernel_size]

        self.weight = self.create_parameter(
            shape=filter_shape,
            attr=self.weight_attr)
        self.bias = self.create_parameter(
            attr=self.bias_attr, shape=[out_channels, ], is_bias=True)

    def forward(self, x, offset, mask):
        in_C = self.in_channels
        out_C = self.out_channels
        stride = self.stride
        padding = self.padding
        # dilation = self.dilation
        groups = self.groups
        N, _, H, W = x.shape
        _, w_in, kH, kW = self.weight.shape
        out_W = (W + 2 * padding - (kW - 1)) // stride
        out_H = (H + 2 * padding - (kH - 1)) // stride

        # ================== 1.先对图片x填充得到填充后的图片pad_x ==================
        pad_x_H = H + padding * 2 + 1
        pad_x_W = W + padding * 2 + 1
        pad_x = F.pad(x, pad=[0, 0, 0, 0, padding, padding + 1, padding, padding + 1], value=0.0)

        # ================== 2.求所有采样点的坐标 ==================
        # 卷积核中心点在pad_x中的位置
        y_outer, x_outer = paddle.meshgrid([paddle.arange(out_H), paddle.arange(out_W)])
        y_outer = y_outer * stride + padding
        x_outer = x_outer * stride + padding
        start_pos_yx = paddle.stack((y_outer, x_outer), 2).cast(dtype='float32')  # [out_H, out_W, 2]         仅仅是卷积核中心点在pad_x中的位置
        start_pos_yx = paddle.unsqueeze(start_pos_yx, axis=[0, 3])                # [1, out_H, out_W, 1, 2]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_yx = paddle.tile(start_pos_yx, [N, 1, 1, kH * kW, 1])  # [N, out_H, out_W, kH*kW, 2]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_y = start_pos_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_x = start_pos_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_y.stop_gradient = True
        start_pos_x.stop_gradient = True

        # 卷积核内部的偏移
        half_W = (kW - 1) // 2
        half_H = (kH - 1) // 2
        y_inner, x_inner = paddle.meshgrid([paddle.arange(kH), paddle.arange(kW)])
        y_inner -= half_H
        x_inner -= half_W
        filter_inner_offset_yx = paddle.stack((y_inner, x_inner), 2).cast(dtype='float32')     # [kH, kW, 2]       卷积核内部的偏移
        filter_inner_offset_yx = paddle.reshape(filter_inner_offset_yx, (1, 1, 1, kH * kW, 2))  # [1, 1, 1, kH*kW, 2]   卷积核内部的偏移
        filter_inner_offset_yx = paddle.tile(filter_inner_offset_yx, [N, out_H, out_W, 1, 1])  # [N, out_H, out_W, kH*kW, 2]   卷积核内部的偏移
        filter_inner_offset_y = filter_inner_offset_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]   卷积核内部的偏移
        filter_inner_offset_x = filter_inner_offset_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]   卷积核内部的偏移
        filter_inner_offset_y.stop_gradient = True
        filter_inner_offset_x.stop_gradient = True

        # 预测的偏移
        offset = paddle.transpose(offset, [0, 2, 3, 1])  # [N, out_H, out_W, kH*kW*2]
        offset_yx = paddle.reshape(offset, (N, out_H, out_W, kH * kW, 2))  # [N, out_H, out_W, kH*kW, 2]
        offset_y = offset_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]
        offset_x = offset_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]

        # 最终采样位置。
        pos_y = start_pos_y + filter_inner_offset_y + offset_y  # [N, out_H, out_W, kH*kW, 1]
        pos_x = start_pos_x + filter_inner_offset_x + offset_x  # [N, out_H, out_W, kH*kW, 1]
        pos_y = paddle.clip(pos_y, 0.0, H + padding * 2 - 1.0)  # 最终采样位置限制在pad_x内
        pos_x = paddle.clip(pos_x, 0.0, W + padding * 2 - 1.0)  # 最终采样位置限制在pad_x内

        # ================== 3.采样。用F.grid_sample()双线性插值采样。 ==================
        pos_x = pos_x / (pad_x_W - 1) * 2.0 - 1.0
        pos_y = pos_y / (pad_x_H - 1) * 2.0 - 1.0
        xtyt = paddle.concat([pos_x, pos_y], -1)  # [N, out_H, out_W, kH*kW, 2]
        xtyt = paddle.reshape(xtyt, (N, out_H, out_W * kH * kW, 2))  # [N, out_H, out_W*kH*kW, 2]
        value = F.grid_sample(pad_x, xtyt, mode='bilinear', padding_mode='zeros', align_corners=True)  # [N, in_C, out_H, out_W*kH*kW]
        value = paddle.reshape(value, (N, in_C, out_H, out_W, kH * kW))    # [N, in_C, out_H, out_W, kH * kW]
        value = value.transpose((0, 1, 4, 2, 3))                           # [N, in_C, kH * kW, out_H, out_W]

        # ================== 4.乘以重要程度 ==================
        # 乘以重要程度
        mask = paddle.unsqueeze(mask, [1])  # [N,    1, kH * kW, out_H, out_W]
        value = value * mask                # [N, in_C, kH * kW, out_H, out_W]
        new_x = paddle.reshape(value, (N, in_C * kH * kW, out_H, out_W))  # [N, in_C * kH * kW, out_H, out_W]

        # ================== 5.乘以本层的权重，加上偏置 ==================
        # 1x1卷积
        rw = paddle.reshape(self.weight, (out_C, w_in * kH * kW, 1, 1))  # [out_C, w_in, kH, kW] -> [out_C, w_in*kH*kW, 1, 1]  变成1x1卷积核
        out = F.conv2d(new_x, rw, bias=self.bias, stride=1, groups=groups)  # [N, out_C, out_H, out_W]
        return out




class MyLayer(nn.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()
        # self.conv1 = nn.Conv2D(3, 8, kernel_size=3, padding=1)

        filter_size = 3
        input_dim = 3
        filters = 8
        stride = 1
        groups = 1
        lr = 1.0
        conv_battr = False
        self.padding = 1
        self.offset_channel = 2 * filter_size ** 2
        self.mask_channel = filter_size ** 2

        self.conv_offset = nn.Conv2D(
            in_channels=input_dim,
            out_channels=3 * filter_size ** 2,
            kernel_size=filter_size,
            stride=stride,
            padding=self.padding,
            weight_attr=ParamAttr(initializer=Constant(0.)),
            bias_attr=ParamAttr(initializer=Constant(0.)))
        # 官方的DCNv2
        # self.conv = DeformConv2D(
        #     in_channels=input_dim,
        #     out_channels=filters,
        #     kernel_size=filter_size,
        #     stride=stride,
        #     padding=self.padding,
        #     dilation=1,
        #     groups=groups,
        #     weight_attr=ParamAttr(learning_rate=lr),
        #     bias_attr=conv_battr)
        # 自实现的DCNv2
        self.conv = MyDCNv2(
            in_channels=input_dim,
            out_channels=filters,
            kernel_size=filter_size,
            stride=stride,
            padding=self.padding,
            dilation=1,
            groups=groups,
            weight_attr=ParamAttr(learning_rate=lr),
            bias_attr=conv_battr)


        self.conv2 = nn.Conv2D(8, 8, kernel_size=1, padding=0)
        self.skip = nn.Conv2D(3, 8, kernel_size=1, padding=0)

    def forward(self, input):
        # x = self.conv1(input)

        offset_mask = self.conv_offset(input)
        offset, mask = paddle.split(
            offset_mask,
            num_or_sections=[self.offset_channel, self.mask_channel],
            axis=1)
        mask = F.sigmoid(mask)
        x = self.conv(input, offset, mask=mask)





        x = self.conv2(x)
        y = self.skip(input)

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



