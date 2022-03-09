
import paddle
import numpy as np
import paddle.nn.functional as F





dic2 = np.load('09_grad.npz')
for batch_idx in range(8):
    print('======================== batch_%.3d ========================'%batch_idx)

    dy_dw_pytorch = dic2['batch_%.3d.dy_dw'%batch_idx]
    dy_dx_pytorch = dic2['batch_%.3d.dy_dx'%batch_idx]
    y_pytorch = dic2['batch_%.3d.y'%batch_idx]
    w = dic2['batch_%.3d.w'%batch_idx]
    w = paddle.to_tensor(w)
    w.stop_gradient = False
    x = dic2['batch_%.3d.x'%batch_idx]
    x = paddle.to_tensor(x)
    x.stop_gradient = False

    D00, D01, D02, D03 = w.shape
    D10, D11, D12, D13 = x.shape


    y = (x * w).sum(axis=[2, 3])

    # dy_dx = paddle.grad(outputs=[y.sum()], inputs=[x], create_graph=True)[0]
    # dy_dw = paddle.grad(outputs=[y.sum()], inputs=[w], create_graph=True)[0]

    dy_dy = paddle.ones(y.shape, dtype=paddle.float32)
    # sum()的求导有2步，先unsqueeze()再tile()，变回求和之前的形状。
    dy_dsum = paddle.unsqueeze(dy_dy, axis=[2, 3])
    dy_dsum = paddle.tile(dy_dsum, [1, 1, max(D02, D12), max(D03, D13)])
    # dy_dx，还要变成x最初的形状，通过求和。
    dy_dx = dy_dsum * w
    axis = []
    for i, _len in enumerate(x.shape):
        if _len == 1:
            axis.append(i)
    if len(axis) > 0:
        dy_dx = paddle.sum(dy_dx, axis=axis, keepdim=True)
    # dy_dw，还要变成w最初的形状，通过求和。
    dy_dw = dy_dsum * x
    axis = []
    for i, _len in enumerate(w.shape):
        if _len == 1:
            axis.append(i)
    if len(axis) > 0:
        dy_dw = paddle.sum(dy_dw, axis=axis, keepdim=True)

    aaaaaa = y.numpy()
    ddd = np.sum((y_pytorch - aaaaaa) ** 2)
    print('ddd=%.6f' % ddd)

    dy_dw_paddle = dy_dw.numpy()
    ddd = np.sum((dy_dw_pytorch - dy_dw_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    dy_dx_paddle = dy_dx.numpy()
    ddd = np.sum((dy_dx_pytorch - dy_dx_paddle) ** 2)
    print('ddd=%.6f' % ddd)
print()
