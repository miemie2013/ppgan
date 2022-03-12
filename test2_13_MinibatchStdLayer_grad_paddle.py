
import paddle
import numpy as np
import paddle.nn.functional as F





dic2 = np.load('10_grad.npz')
for batch_idx in range(8):
    print('======================== batch_%.3d ========================'%batch_idx)

    dloss_dx_pytorch = dic2['batch_%.3d.dloss_dx'%batch_idx]
    out_pytorch = dic2['batch_%.3d.out'%batch_idx]
    x = dic2['batch_%.3d.x'%batch_idx]
    x = paddle.to_tensor(x)
    x.stop_gradient = False



    N, C, H, W = x.shape
    G = N
    F = 512
    c = C // F

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

    y0 = x.reshape((G, -1, F, c, H, W))    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
    y1 = y0 - y0.mean(0)                   # [GnFcHW] Subtract mean over group.
    y2 = y1.square().mean(0)              # [nFcHW]  Calc variance over group.
    y3 = (y2 + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
    y4 = y3.mean([2, 3, 4])               # [nF]     Take average over channels and pixels.
    y5 = y4.reshape((-1, F, 1, 1))          # [nF11]   Add missing dimensions.
    y6 = y5.tile([G, 1, H, W])            # [NFHW]   Replicate over group and pixels.
    out = paddle.concat([x, y6], 1)        # [NCHW]   Append to input as new channels.


    # dloss_dx = paddle.grad(outputs=[out.sum()], inputs=[x], create_graph=True)[0]
    dloss_dout = paddle.ones(out.shape, dtype=paddle.float32)
    dloss_dx_1 = dloss_dout[:, :C, :, :]
    dloss_dx_2 = dloss_dout[:, C:, :, :]
    dloss_dy6 = dloss_dx_2
    dloss_dy5 = dloss_dy6


    out_paddle = out.numpy()
    ddd = np.mean((out_pytorch - out_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    dloss_dx_paddle = dloss_dx.numpy()
    ddd = np.mean((dloss_dx_pytorch - dloss_dx_paddle) ** 2)
    print('ddd=%.6f' % ddd)
print()
