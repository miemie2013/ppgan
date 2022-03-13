import paddle
import numpy as np
import paddle.nn.functional as F

dic2 = np.load('10_grad.npz')
for batch_idx in range(8):
    print('======================== batch_%.3d ========================' % batch_idx)

    dloss_dx_pytorch = dic2['batch_%.3d.dloss_dx' % batch_idx]
    out_pytorch = dic2['batch_%.3d.out' % batch_idx]
    x = dic2['batch_%.3d.x' % batch_idx]
    x = paddle.to_tensor(x)
    x.stop_gradient = False

    # N, C, H, W = x.shape
    # G = N
    # F = 512
    # c = C // F

    # N, C, H, W = x.shape
    # G = N
    # F = 256
    # c = C // F

    # N, C, H, W = x.shape
    # G = 1
    # F = 512
    # c = C // F

    # N, C, H, W = x.shape
    # G = 1
    # F = 256
    # c = C // F

    N, C, H, W = x.shape
    G = 2
    F = 256
    c = C // F

    y0 = x.reshape((G, -1, F, c, H,
                    W))  # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
    y1 = y0 - y0.mean(0)  # [GnFcHW] Subtract mean over group.
    y2 = y1.square().mean(0)  # [nFcHW]  Calc variance over group.
    y3 = (y2 + 1e-8).sqrt()  # [nFcHW]  Calc stddev over group.
    y4 = y3.mean([2, 3, 4])  # [nF]     Take average over channels and pixels.
    y5 = y4.reshape((-1, F, 1, 1))  # [nF11]   Add missing dimensions.
    y6 = y5.tile([G, 1, H, W])  # [NFHW]   Replicate over group and pixels.
    out = paddle.concat([x, y6], 1)  # [NCHW]   Append to input as new channels.


    # dloss_dx22 = paddle.grad(outputs=[out.sum()], inputs=[x], create_graph=True)[0]
    dloss_dout = paddle.ones(out.shape, dtype=paddle.float32)

    dloss_dx_1 = dloss_dout[:, :C, :, :]
    dloss_dx_2 = dloss_dout[:, C:, :, :]
    dloss_dy6 = dloss_dx_2

    dloss_dy5 = dloss_dy6.reshape((y5.shape[0], -1, F, H, W))
    dloss_dy5 = paddle.sum(dloss_dy5, axis=[1], keepdim=False)
    dloss_dy5 = paddle.sum(dloss_dy5, axis=[2, 3], keepdim=True)

    dloss_dy4 = dloss_dy5.reshape((-1, F))

    dloss_dy3 = dloss_dy4.reshape((-1, F, 1, 1, 1))
    dloss_dy3 = dloss_dy3.tile([1, 1, c, H, W]) / c / H / W

    dloss_dy2 = dloss_dy3 * 0.5 / y3

    dloss_dy1 = dloss_dy2.reshape((1, -1, F, c, H, W))
    dloss_dy1 = dloss_dy1.tile([G, 1, 1, 1, 1, 1]) / G
    dloss_dy1 = dloss_dy1 * 2 * y1

    dloss_dy0 = dloss_dy1 - dloss_dy1.mean(0)

    dloss_dx = dloss_dy0.reshape(x.shape)

    dloss_dx = dloss_dx + dloss_dx_1

    out_paddle = out.numpy()
    ddd = np.mean((out_pytorch - out_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    dloss_dx_paddle = dloss_dx.numpy()
    ddd = np.mean((dloss_dx_pytorch - dloss_dx_paddle) ** 2)
    print('ddd=%.6f' % ddd)
print()
