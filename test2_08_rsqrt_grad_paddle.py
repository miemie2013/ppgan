
import paddle
import numpy as np
import paddle.nn.functional as F

from ppgan.models.generators.generator_styleganv2ada import Conv2D_Grad



grad_layer = Conv2D_Grad()

dic2 = np.load('08_grad.npz')
for batch_idx in range(20):
    print('======================== batch_%.3d ========================'%batch_idx)

    dy_dw_pytorch = dic2['batch_%.3d.dy_dw'%batch_idx]
    y_pytorch = dic2['batch_%.3d.y'%batch_idx]
    w = dic2['batch_%.3d.w'%batch_idx]
    w = paddle.to_tensor(w)
    w.stop_gradient = False

    # rsqrt()表示平方根的倒数
    y = (w.square().sum(axis=[2, 3, 4]) + 1e-8).rsqrt()
    # y = (w.sum(axis=[2, 3, 4]) + 10000).rsqrt()
    # y = (w + 100).rsqrt()
    # dy_dw = paddle.grad(outputs=[y.sum()], inputs=[w], create_graph=True)[0]
    dysum_dy = paddle.ones(y.shape, dtype=paddle.float32)
    dloss_dw_square_sum_add_1e8 = -0.5 * dysum_dy * y * y * y
    dloss_dw_square_sum = dloss_dw_square_sum_add_1e8
    dloss_dw_square = paddle.unsqueeze(dloss_dw_square_sum, axis=[2, 3, 4])
    dloss_dw_square = paddle.tile(dloss_dw_square, [1, 1, w.shape[2], w.shape[3], w.shape[4]])
    dloss_dw = dloss_dw_square * 2 * w

    dy_dw = dloss_dw


    aaaaaa = y.numpy()
    ddd = np.sum((y_pytorch - aaaaaa) ** 2)
    print('ddd=%.6f' % ddd)

    dy_dw_paddle = dy_dw.numpy()
    ddd = np.sum((dy_dw_pytorch - dy_dw_paddle) ** 2)
    print('ddd=%.6f' % ddd)
print()
