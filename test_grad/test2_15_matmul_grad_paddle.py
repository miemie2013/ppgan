import paddle
import numpy as np
import paddle.nn.functional as F




dic2 = np.load('15_grad.npz')
for batch_idx in range(8):
    print('======================== batch_%.3d ========================' % batch_idx)

    dloss_dx_pytorch = dic2['batch_%.3d.dloss_dx' % batch_idx]
    y_pytorch = dic2['batch_%.3d.out' % batch_idx]
    x = dic2['batch_%.3d.x' % batch_idx]
    x = paddle.to_tensor(x)
    x.stop_gradient = False
    C = dic2['batch_%.3d.C' % batch_idx]
    C = paddle.to_tensor(C)

    y = C[:, :3, :3] @ x + C[:, :3, 3:]
    #  [N, A, B] x [N, B, C] = [N, A, C]


    # dloss_dx = paddle.grad(outputs=[y.sum()], inputs=[x], create_graph=False)[0]
    dloss_dout = paddle.ones(y.shape, dtype=paddle.float32)  # [N, A, C]

    dloss_dout = paddle.unsqueeze(dloss_dout, 2)  # [N, A, 1, C]
    dout_dx = C[:, :3, :3]  # [N, A, B]
    dout_dx = paddle.unsqueeze(dout_dx, 3)  # [N, A, B, 1]
    dloss_dx = dloss_dout * dout_dx         # [N, A, B, C]
    dloss_dx = paddle.sum(dloss_dx, axis=1)  # [N, B, C]


    y_paddle = y.numpy()
    ddd = np.mean((y_pytorch - y_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    dloss_dx_paddle = dloss_dx.numpy()
    ddd = np.mean((dloss_dx_pytorch - dloss_dx_paddle) ** 2)
    print('ddd=%.6f' % ddd)
print()
