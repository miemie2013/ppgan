
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

    kernel_size = 3
    stride = 1
    padding = 1
    output_padding = 0
    dilation = 1
    groups = 2

    kernel_size = 3
    stride = 2
    padding = 1
    output_padding = 0
    dilation = 1
    groups = 1

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

    pad_dloss_dy = F.pad(dysum_dy, [padding, padding, padding, padding])  # [N, in_C, pad_H, pad_W]
    N, out_C, pad_H, pad_W = pad_dloss_dy.shape
    pad_dloss_dy = paddle.transpose(pad_dloss_dy, [2, 3, 0, 1])  # [N, out_C, pad_H, pad_W] -> [pad_H, pad_W, N, out_C]
    pad_dloss_dy = paddle.reshape(pad_dloss_dy, (pad_H, pad_W, N, g, oc))  # [pad_H, pad_W, N, g, oc]
    kerner_center_y, kerner_center_x = paddle.meshgrid([paddle.arange(in_H), paddle.arange(in_W)])
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
                                      kerner_center_yx_20, kerner_center_yx_21, kerner_center_yx_22), 0)  # [kH*kW, in_H, in_W, 2]
    elif kH == 1:
        kerner_center_yx_00 = paddle.stack((kerner_center_y, kerner_center_x), 2).cast(dtype='int32')
        kerner_pos_yx = paddle.unsqueeze(kerner_center_yx_00, 0)  # [kH*kW, in_H, in_W, 2]
    else:
        raise NotImplementedError("kH \'{}\' is not implemented.".format(kH))
    kerner_pos_yx = paddle.reshape(kerner_pos_yx, (-1, 2))  # [kH*kW, in_H, in_W, 2] -> [kH*kW*in_H*in_W, 2]
    kerner_pos_yx.stop_gradient = True
    dloss_dY = paddle.gather_nd(pad_dloss_dy, kerner_pos_yx)  # [pad_H, pad_W, N, g, oc] -> [kH*kW*in_H*in_W, N, g, oc]
    dloss_dY = paddle.reshape(dloss_dY, (kH, kW, in_H, in_W, N, g, oc))         # [kH, kW, in_H, in_W, N, g, oc]
    dloss_dY = paddle.transpose(dloss_dY, [4, 5, 6, 2, 3, 0, 1])                # [N, g, oc, in_H, in_W, kH, kW]
    dloss_dY = paddle.reshape(dloss_dY, (N, g, 1, oc, in_H, in_W, kH, kW))      # [N, g, 1, oc, in_H, in_W, kH, kW]
    dY_dW = paddle.reshape(x, (N, g, c, 1, in_H, in_W, 1, 1))                   # [N, g, c, 1, in_H, in_W, 1, 1]
    dloss_dW = dloss_dY * dY_dW                                                 # [N, g, c, oc, in_H, in_W, kH, kW]
    dloss_dW = paddle.sum(dloss_dW, axis=[0, 4, 5])    # [g, c, oc, kH, kW]
    dloss_dW = paddle.reshape(dloss_dW, (g*c, oc, kH, kW))
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
