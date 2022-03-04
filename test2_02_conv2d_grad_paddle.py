
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
    groups = 256

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

    dy_dx = paddle.grad(outputs=[y.sum()], inputs=[x], create_graph=True)[0]
    dy_dw = paddle.grad(outputs=[y.sum()], inputs=[w], create_graph=True)[0]
    # dysum_dy = paddle.ones(y.shape, dtype=paddle.float32)
    # dy_dx, dy_dw = grad_layer(dysum_dy, x=x, weight=w, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    # dy_dx = grad_layer(dysum_dy, y, x=x, weight=w, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    aaaaaa = y.numpy()
    ddd = np.sum((y_pytorch - aaaaaa) ** 2)
    print('ddd=%.6f' % ddd)

    aaaaaa = dy_dx.numpy()
    ddd = np.sum((dy_dx_pytorch - aaaaaa) ** 2)
    print('ddd=%.6f' % ddd)

    aaaaaa = dy_dw.numpy()
    ddd = np.sum((dy_dw_pytorch - aaaaaa) ** 2)
    print('ddd=%.6f' % ddd)
print()
