
import paddle
import numpy as np
from ppgan.models.generators.generator_styleganv2ada import _conv2d_wrapper, _conv2d_wrapper_grad

dic2 = np.load('02_grad.npz')
for batch_idx in range(20):
    print('======================== batch_%.3d ========================'%batch_idx)
    dy_dx_pytorch = dic2['batch_%.3d.dy_dx'%batch_idx]
    dy_dw_pytorch = dic2['batch_%.3d.dy_dw'%batch_idx]
    y_pytorch = dic2['batch_%.3d.y'%batch_idx]
    w = dic2['batch_%.3d.w'%batch_idx]
    x = dic2['batch_%.3d.x'%batch_idx]
    x = paddle.to_tensor(x)
    x.stop_gradient = False
    w = paddle.to_tensor(w)
    w.stop_gradient = False

    py0 = 0
    px0 = 0
    groups = 1
    flip_weight = True
    y = _conv2d_wrapper(x=x, w=w, padding=[py0, px0], groups=groups, flip_weight=flip_weight)

    # dy_dx = paddle.grad(outputs=[y.sum()], inputs=[x], create_graph=True)[0]
    # dy_dw = paddle.grad(outputs=[y.sum()], inputs=[w], create_graph=True)[0]
    dysum_dy = paddle.ones(y.shape, dtype=paddle.float32)
    dy_dx, dy_dw = _conv2d_wrapper_grad(dysum_dy, x=x, w=w, padding=[py0, px0], groups=groups, flip_weight=flip_weight)

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
