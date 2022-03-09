
import paddle
import numpy as np
import paddle.nn.functional as F

from ppgan.models.generators.generator_styleganv2ada import Conv2D_Grad



grad_layer = Conv2D_Grad()

dic2 = np.load('02_grad.npz')
for batch_idx in range(20):
    print('======================== batch_%.3d ========================'%batch_idx)
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

    dy_dx_pytorch = dic2['batch_%.3d.dy_dx'%batch_idx]
    dy_dw_pytorch = dic2['batch_%.3d.dy_dw'%batch_idx]
    y_pytorch = dic2['batch_%.3d.y'%batch_idx]
    w = dic2['batch_%.3d.w'%batch_idx]
    x = dic2['batch_%.3d.x'%batch_idx]
    x = paddle.to_tensor(x)
    x.stop_gradient = False
    w = paddle.to_tensor(w)
    w.stop_gradient = False
    bias = None

    y = F.conv2d(x=x, weight=w, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    # N, C, H, W = x.shape
    # aaa = F.conv2d(x=paddle.reshape(x, (N*groups, C // groups, H, W)), weight=w, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=1)
    # _, out_C, out_H, out_W = aaa.shape
    # aaa = paddle.reshape(aaa, (N, groups, out_C, out_H, out_W))
    # y = paddle.mean(aaa, axis=1)

    # dy_dx = paddle.grad(outputs=[y.sum()], inputs=[x], create_graph=True)[0]
    # dy_dw = paddle.grad(outputs=[y.sum()], inputs=[w], create_graph=True)[0]
    dysum_dy = paddle.ones(y.shape, dtype=paddle.float32)
    # dy_dx = grad_layer(dysum_dy, y, x=x, weight=w, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    # output_padding = 111
    # if kernel_size == 1 and stride == 1 and padding == 0:
    #     output_padding = 0
    # elif kernel_size == 1 and stride == 2 and padding == 0:
    #     output_padding = 1
    # elif kernel_size == 3 and stride == 1 and padding == 0:
    #     output_padding = 0
    # elif kernel_size == 3 and stride == 2 and padding == 0:
    #     output_padding = 1
    # elif kernel_size == 3 and stride == 1 and padding == 1:
    #     output_padding = 0
    # elif kernel_size == 3 and stride == 2 and padding == 1:
    #     output_padding = 1
    output_padding = stride - 1
    dy_dx = F.conv2d_transpose(x=dysum_dy, weight=w, bias=bias, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups)

    # 求dloss_dW
    N, out_C, out_H, out_W = y.shape
    out_C, c, kH, kW = w.shape
    g = groups
    oc = out_C // g
    pad_x = F.pad(x, [padding, padding, padding, padding])  # [N, in_C, pad_H, pad_W]
    N, in_C, pad_H, pad_W = pad_x.shape
    pad_x = paddle.transpose(pad_x, [2, 3, 0, 1])  # [N, in_C, pad_H, pad_W] -> [pad_H, pad_W, N, in_C]
    pad_x = paddle.reshape(pad_x, (pad_H, pad_W, N, g, c))  # [pad_H, pad_W, N, g, c]
    kerner_center_y, kerner_center_x = paddle.meshgrid([paddle.arange(out_H), paddle.arange(out_W)])
    kerner_center_y = kerner_center_y * stride + (kH - 1) // 2
    kerner_center_x = kerner_center_x * stride + (kW - 1) // 2
    assert kH == kW
    if kH == 3:
        kerner_center_yx_00 = paddle.stack((kerner_center_y - 1, kerner_center_x - 1), 2).cast(dtype='int32')
        kerner_center_yx_01 = paddle.stack((kerner_center_y - 1, kerner_center_x), 2).cast(dtype='int32')
        kerner_center_yx_02 = paddle.stack((kerner_center_y - 1, kerner_center_x + 1), 2).cast(dtype='int32')
        kerner_center_yx_10 = paddle.stack((kerner_center_y, kerner_center_x - 1), 2).cast(dtype='int32')
        kerner_center_yx_11 = paddle.stack((kerner_center_y, kerner_center_x), 2).cast(dtype='int32')
        kerner_center_yx_12 = paddle.stack((kerner_center_y, kerner_center_x + 1), 2).cast(dtype='int32')
        kerner_center_yx_20 = paddle.stack((kerner_center_y + 1, kerner_center_x - 1), 2).cast(dtype='int32')
        kerner_center_yx_21 = paddle.stack((kerner_center_y + 1, kerner_center_x), 2).cast(dtype='int32')
        kerner_center_yx_22 = paddle.stack((kerner_center_y + 1, kerner_center_x + 1), 2).cast(dtype='int32')
        kerner_pos_yx = paddle.stack((kerner_center_yx_00, kerner_center_yx_01, kerner_center_yx_02,
                                      kerner_center_yx_10, kerner_center_yx_11, kerner_center_yx_12,
                                      kerner_center_yx_20, kerner_center_yx_21, kerner_center_yx_22), 0)  # [kH*kW, out_H, out_W, 2]
    elif kH == 1:
        kerner_center_yx_00 = paddle.stack((kerner_center_y, kerner_center_x), 2).cast(dtype='int32')
        kerner_pos_yx = paddle.unsqueeze(kerner_center_yx_00, 0)  # [kH*kW, out_H, out_W, 2]
    else:
        raise NotImplementedError("kH \'{}\' is not implemented.".format(kH))
    kerner_pos_yx = paddle.reshape(kerner_pos_yx, (-1, 2))  # [kH*kW, out_H, out_W, 2] -> [kH*kW*out_H*out_W, 2]
    kerner_pos_yx.stop_gradient = True
    dY_dW = paddle.gather_nd(pad_x, kerner_pos_yx)  # [pad_H, pad_W, N, g, c] -> [kH*kW*out_H*out_W, N, g, c]
    dY_dW = paddle.reshape(dY_dW, (kH, kW, out_H, out_W, N, g, c))  # [kH, kW, out_H, out_W, N, g, c]
    dY_dW = paddle.transpose(dY_dW, [4, 5, 6, 2, 3, 0, 1])             # [N, g, c, out_H, out_W, kH, kW]
    dY_dW = paddle.reshape(dY_dW, (N, g, 1, c, out_H, out_W, kH, kW))     # [N, g, 1, c, out_H, out_W, kH, kW]
    grad = paddle.reshape(dysum_dy, (N, g, oc, 1, out_H, out_W, 1, 1))    # [N, g, oc, 1, out_H, out_W, 1, 1]
    # 旧的方案，用逐元素相乘，显存爆炸
    # dloss_dW = grad * dY_dW                                               # [N, g, oc, c, out_H, out_W, kH, kW]
    # dloss_dW = paddle.sum(dloss_dW, axis=[0, 4, 5])    # [g, oc, c, kH, kW]
    # dloss_dW = paddle.reshape(dloss_dW, (g*oc, c, kH, kW))
    # 新的方案，用1x1卷积等价实现，显存不爆炸。
    dY_dW = paddle.transpose(dY_dW, [3, 1, 2, 0, 4, 5, 6, 7])    # [c, g, 1, N, out_H, out_W, kH, kW]
    grad = paddle.transpose(grad, [3, 1, 2, 0, 4, 5, 6, 7])      # [1, g, oc, N, out_H, out_W, 1, 1]
    dY_dW = paddle.reshape(dY_dW, (c, g*N*out_H*out_W, kH, kW))
    grad = paddle.reshape(grad, (g*oc, N*out_H*out_W, 1, 1))
    dloss_dW = F.conv2d(dY_dW, grad, groups=g)  # [c, g*oc, kH, kW]
    dloss_dW = paddle.transpose(dloss_dW, [1, 0, 2, 3])  # [g*oc, c, kH, kW]

    dy_dw = dloss_dW

    aaaaaa = y.numpy()
    ddd = np.sum((y_pytorch - aaaaaa) ** 2)
    print('ddd=%.6f' % ddd)

    dy_dx_paddle = dy_dx.numpy()
    ddd = np.sum((dy_dx_pytorch - dy_dx_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    dy_dw_paddle = dy_dw.numpy()
    ddd = np.sum((dy_dw_pytorch - dy_dw_paddle) ** 2)
    print('ddd=%.6f' % ddd)
print()
