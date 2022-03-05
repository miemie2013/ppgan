
import paddle
import numpy as np
import paddle.nn.functional as F

from ppgan.models.generators.generator_styleganv2ada import Conv2D_Grad



def F_conv2d(x, w, b, stride, padding, groups=1):
    # 卷积层推理时的前向传播。
    N, C, H, W = x.shape
    assert (C % groups == 0), "(C % groups == 0)"
    out_C, c, kH, kW = w.shape    # c = C // groups
    assert (out_C % groups == 0), "(out_C % groups == 0)"
    oc = out_C // groups
    out_W = (W+2*padding-(kW-1)) // stride
    out_H = (H+2*padding-(kH-1)) // stride
    out = np.zeros((N, out_C, out_H, out_W), np.float32)

    # 1.先对图片x填充得到填充后的图片pad_x
    pad_x = np.zeros((N, C, H + padding*2, W + padding*2), np.float32)
    pad_x[:, :, padding:padding + H, padding:padding + W] = x

    # 2.卷积核滑动，只会在H和W两个方向上滑动
    for i in range(out_H):   # i是纵坐标
        for j in range(out_W):   # j是横坐标
            ori_x = j*stride   # 卷积核在pad_x中的横坐标，等差数列，公差是stride
            ori_y = i*stride   # 卷积核在pad_x中的纵坐标，等差数列，公差是stride
            part_x = pad_x[:, :, ori_y:ori_y+kH, ori_x:ori_x+kW]   # 截取卷积核所处的位置的像素 [N, C, kH, kW]
            part_x = np.reshape(part_x, (N, groups, 1, c, kH, kW))      # [N, groups, 1, c, kH, kW]。
            exp_w = np.reshape(w, (1, groups, oc, c, kH, kW))      # 卷积核，[1, groups, oc, c, kH, kW]。
            mul = part_x * exp_w   # 卷积核和exp_part_x相乘，[N, groups, oc, c, kH, kW]。
            mul = np.sum(mul, axis=(3, 4, 5))       # 后3维求和，[N, groups, oc]。
            mul = np.reshape(mul, (N, out_C))       # 后3维求和，[N, out_C]。
            if b is not None:
                mul += b    # 加上偏移，[N, out_C]。
            # 将得到的新像素写进out的对应位置
            out[:, :, i, j] = mul
    return out


dic2 = np.load('02_grad.npz')
for batch_idx in range(20):
    print('======================== batch_%.3d ========================'%batch_idx)
    stride = 2
    padding = 1
    dilation = 1
    groups = 256

    dy_dx_pytorch = dic2['batch_%.3d.dy_dx'%batch_idx]
    dy_dw_pytorch = dic2['batch_%.3d.dy_dw'%batch_idx]
    y_pytorch = dic2['batch_%.3d.y'%batch_idx]
    w = dic2['batch_%.3d.w'%batch_idx]
    x = dic2['batch_%.3d.x'%batch_idx]
    # x = paddle.to_tensor(x)
    # x.stop_gradient = False
    # w = paddle.to_tensor(w)
    # w.stop_gradient = False
    bias = None

    y = F_conv2d(x=x, w=w, b=bias, stride=stride, padding=padding, groups=groups)

    aaaaaa = y
    ddd = np.sum((y_pytorch - aaaaaa) ** 2)
    print('ddd=%.6f' % ddd)
print()
