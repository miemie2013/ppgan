
import paddle
import numpy as np
import paddle.nn.functional as F




dic2 = np.load('03_grad.npz')
for batch_idx in range(20):
    print('======================== batch_%.3d ========================'%batch_idx)
    kernel_size = 1
    stride = 1
    padding = 0
    output_padding = 0
    dilation = 1
    groups = 1

    kernel_size = 1
    stride = 2
    padding = 0
    output_padding = 0
    dilation = 1
    groups = 1

    kernel_size = 3
    stride = 1
    padding = 0
    output_padding = 0
    dilation = 1
    groups = 1

    kernel_size = 3
    stride = 2
    padding = 0
    output_padding = 0
    dilation = 1
    groups = 1

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

    # kernel_size = 3
    # stride = 2
    # padding = 1
    # output_padding = 0
    # dilation = 1
    # groups = 2

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

    y = F.conv2d_transpose(x=x, weight=w, bias=bias, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups)

    # dy_dx = paddle.grad(outputs=[y.sum()], inputs=[x], create_graph=True)[0]
    # dy_dw = paddle.grad(outputs=[y.sum()], inputs=[w], create_graph=True)[0]
    dysum_dy = paddle.ones(y.shape, dtype=paddle.float32)
    dy_dx = F.conv2d(x=dysum_dy, weight=w, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    # 求dloss_dW
    N, in_C, in_H, in_W = x.shape
    N, out_C, out_H, out_W = y.shape
    in_C, oc, kH, kW = w.shape
    g = groups
    c = in_C // g

    padding_x = kH
    pad_x = F.pad(x, [padding_x, padding_x, padding_x, padding_x])  # [N, in_C, pad_x_H, pad_x_W]
    transpose_x = paddle.transpose(pad_x, [2, 3, 0, 1])  # [N, in_C, pad_x_H, pad_x_W] -> [pad_x_H, pad_x_W, N, in_C]
    pad_y = F.pad(y, [padding, padding, padding, padding])  # [N, in_C, pad_x_H, pad_x_W]
    _, _, pad_y_H, pad_y_W = pad_y.shape
    Y_pos_y, Y_pos_x = paddle.meshgrid([paddle.arange(pad_y_H), paddle.arange(pad_y_W)])
    Y_pos_y = Y_pos_y + padding_x
    Y_pos_x = Y_pos_x + padding_x
    assert kH == kW
    if kH == 3 and stride == 1:
        Y_pos_yx_00 = paddle.stack((Y_pos_y, Y_pos_x), 2).cast(dtype='int32')   # [out_H, out_W, 2]
        Y_pos_yx_01 = paddle.stack((Y_pos_y, Y_pos_x - 1), 2).cast(dtype='int32')   # [out_H, out_W, 2]
        Y_pos_yx_02 = paddle.stack((Y_pos_y, Y_pos_x - 2), 2).cast(dtype='int32')   # [out_H, out_W, 2]
        Y_pos_yx_10 = paddle.stack((Y_pos_y - 1, Y_pos_x), 2).cast(dtype='int32')   # [out_H, out_W, 2]
        Y_pos_yx_11 = paddle.stack((Y_pos_y - 1, Y_pos_x - 1), 2).cast(dtype='int32')   # [out_H, out_W, 2]
        Y_pos_yx_12 = paddle.stack((Y_pos_y - 1, Y_pos_x - 2), 2).cast(dtype='int32')   # [out_H, out_W, 2]
        Y_pos_yx_20 = paddle.stack((Y_pos_y - 2, Y_pos_x), 2).cast(dtype='int32')   # [out_H, out_W, 2]
        Y_pos_yx_21 = paddle.stack((Y_pos_y - 2, Y_pos_x - 1), 2).cast(dtype='int32')   # [out_H, out_W, 2]
        Y_pos_yx_22 = paddle.stack((Y_pos_y - 2, Y_pos_x - 2), 2).cast(dtype='int32')   # [out_H, out_W, 2]
        Y_pos_yx = paddle.stack((Y_pos_yx_00, Y_pos_yx_01, Y_pos_yx_02,
                                 Y_pos_yx_10, Y_pos_yx_11, Y_pos_yx_12,
                                 Y_pos_yx_20, Y_pos_yx_21, Y_pos_yx_22), 0)  # [kH*kW, out_H, out_W, 2]
    elif kH == 3 and stride == 2:
        Y_pos_yx_00 = paddle.stack((Y_pos_y, Y_pos_x), 2).cast(dtype='int32')   # [out_H, out_W, 2]
        Y_pos_yx_01 = paddle.stack((Y_pos_y, Y_pos_x - 1), 2).cast(dtype='int32')   # [out_H, out_W, 2]
        Y_pos_yx_02 = paddle.stack((Y_pos_y, Y_pos_x - 2), 2).cast(dtype='int32')   # [out_H, out_W, 2]
        Y_pos_yx_10 = paddle.stack((Y_pos_y - 1, Y_pos_x), 2).cast(dtype='int32')   # [out_H, out_W, 2]
        Y_pos_yx_11 = paddle.stack((Y_pos_y - 1, Y_pos_x - 1), 2).cast(dtype='int32')   # [out_H, out_W, 2]
        Y_pos_yx_12 = paddle.stack((Y_pos_y - 1, Y_pos_x - 2), 2).cast(dtype='int32')   # [out_H, out_W, 2]
        Y_pos_yx_20 = paddle.stack((Y_pos_y - 2, Y_pos_x), 2).cast(dtype='int32')   # [out_H, out_W, 2]
        Y_pos_yx_21 = paddle.stack((Y_pos_y - 2, Y_pos_x - 1), 2).cast(dtype='int32')   # [out_H, out_W, 2]
        Y_pos_yx_22 = paddle.stack((Y_pos_y - 2, Y_pos_x - 2), 2).cast(dtype='int32')   # [out_H, out_W, 2]
        Y_pos_yx = paddle.stack((Y_pos_yx_00, Y_pos_yx_01, Y_pos_yx_02,
                                 Y_pos_yx_10, Y_pos_yx_11, Y_pos_yx_12,
                                 Y_pos_yx_20, Y_pos_yx_21, Y_pos_yx_22), 0)  # [kH*kW, out_H, out_W, 2]
    elif kH == 1 and stride == 1:
        Y_pos_yx_00 = paddle.stack((Y_pos_y, Y_pos_x), 2).cast(dtype='int32')   # [out_H, out_W, 2]
        Y_pos_yx = paddle.unsqueeze(Y_pos_yx_00, 0)  # [kH*kW, out_H, out_W, 2]
    else:
        raise NotImplementedError("kH \'{}\' is not implemented.".format(kH))
    Y_pos_yx = paddle.reshape(Y_pos_yx, (-1, 2))  # [kH*kW, out_H, out_W, 2] -> [kH*kW*out_H*out_W, 2]
    Y_pos_yx.stop_gradient = True
    dY_dW = paddle.gather_nd(transpose_x, Y_pos_yx)  # [pad_x_H, pad_x_W, N, in_C] -> [kH*kW*out_H*out_W, N, in_C]
    dY_dW = paddle.reshape(dY_dW, (kH, kW, pad_y_H, pad_y_W, N, g, c))  # [kH, kW, out_H, out_W, N, g, c]
    dY_dW = paddle.transpose(dY_dW, [4, 5, 6, 2, 3, 0, 1])                # [N, g, c, out_H, out_W, kH, kW]
    dY_dW = paddle.reshape(dY_dW, (N, g, c, 1, pad_y_H, pad_y_W, kH, kW))     # [N, g, c, 1, out_H, out_W, kH, kW]
    grad = F.pad(dysum_dy, [padding, padding, padding, padding])          # [N, in_C, pad_x_H, pad_x_W]
    grad = paddle.reshape(grad, (N, g, 1, oc, pad_y_H, pad_y_W, 1, 1))    # [N, g, 1, oc, out_H, out_W, 1, 1]
    dloss_dW = grad * dY_dW                                               # [N, g, c, oc, out_H, out_W, kH, kW]
    dloss_dW = paddle.sum(dloss_dW, axis=[0, 4, 5])    # [g, c, oc, kH, kW]
    dloss_dW = paddle.reshape(dloss_dW, (g*c, oc, kH, kW))
    dy_dw = dloss_dW

    aaaaaa = y.numpy()
    ddd = np.sum((y_pytorch - aaaaaa) ** 2)
    print('ddd=%.6f' % ddd)

    dy_dx_paddle = dy_dx.numpy()
    ddd = np.sum((dy_dx_pytorch - dy_dx_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    aaaaaa = dy_dw.numpy()
    ddd = np.sum((dy_dw_pytorch - aaaaaa) ** 2)
    print('ddd=%.6f' % ddd)
print()
