
import paddle
import numpy as np
import paddle.nn.functional as F

from ppgan.models.generators.generator_styleganv2ada import Conv2D_Grad



grad_layer = Conv2D_Grad()

dic2 = np.load('02_grad.npz')
for batch_idx in range(20):
    print('======================== batch_%.3d ========================'%batch_idx)
    stride = 2
    padding = 1
    dilation = 1
    groups = 1

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
    dy_dx = F.conv2d_transpose(x=dysum_dy, weight=w, bias=bias, stride=stride, padding=padding, output_padding=1, dilation=dilation, groups=groups)

    # 求dloss_dW
    N, out_C, out_H, out_W = y.shape
    out_C, c, kH, kW = w.shape
    pad_x = F.pad(x, [padding, padding, padding, padding])  # [N, in_C, pad_H, pad_W]
    N, in_C, pad_H, pad_W = pad_x.shape
    pad_x = paddle.transpose(pad_x, [2, 3, 0, 1])  # [N, in_C, pad_H, pad_W] -> [pad_H, pad_W, N, in_C]
    kerner_center_y, kerner_center_x = paddle.meshgrid([paddle.arange(y.shape[2]), paddle.arange(y.shape[3])])
    kerner_center_y = kerner_center_y * stride + padding
    kerner_center_x = kerner_center_x * stride + padding
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
    dY_dW = paddle.gather_nd(pad_x, kerner_pos_yx)  # [pad_H, pad_W, N, in_C] -> [kH*kW*out_H*out_W, N, in_C]
    dY_dW = paddle.reshape(dY_dW, (kH, kW, out_H, out_W, N, in_C))  # [kH, kW, out_H, out_W, N, in_C]
    dY_dW = paddle.transpose(dY_dW, [4, 5, 2, 3, 0, 1])             # [N, in_C, out_H, out_W, kH, kW]
    dY_dW = paddle.reshape(dY_dW, (N, 1, in_C, out_H, out_W, kH, kW))     # [N, 1, in_C, out_H, out_W, kH, kW]
    grad = paddle.reshape(dysum_dy, (N, out_C, 1, out_H, out_W, 1, 1))    # [N, out_C, 1, out_H, out_W, 1, 1]
    dloss_dW = grad * dY_dW                                              # [N, out_C, in_C, out_H, out_W, kH, kW]
    dloss_dW = paddle.sum(dloss_dW, axis=[0, 3, 4])    # [out_C, in_C, kH, kW]
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
