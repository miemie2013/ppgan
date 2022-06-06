
import paddle
import numpy as np
import paddle.nn.functional as F





dic2 = np.load('16_grad.npz')
for batch_idx in range(8):
    print('======================== batch_%.3d ========================'%batch_idx)

    dloss_dstyles_pytorch = dic2['batch_%.3d.dloss_dstyles'%batch_idx]
    out_pytorch = dic2['batch_%.3d.out'%batch_idx]
    styles = dic2['batch_%.3d.styles'%batch_idx]
    styles = paddle.to_tensor(styles)
    styles.stop_gradient = False
    weight = dic2['batch_%.3d.weight'%batch_idx]
    weight = paddle.to_tensor(weight)
    weight.stop_gradient = False
    x = dic2['batch_%.3d.x'%batch_idx]
    x = paddle.to_tensor(x)
    x.stop_gradient = False

    batch_size = 2

    out_C, in_C, kH, kW = weight.shape

    w = weight.unsqueeze(0)           # [1, out_C, in_C, kH, kW]
    styles2 = styles.reshape((batch_size, 1, -1, 1, 1))
    w1 = w * styles2     # [N, out_C, in_C, kH, kW]
    dcoefs = (w1.square().sum(axis=[2, 3, 4]) + 1e-8).rsqrt()        # [N, out_C]
    w2 = w1 * dcoefs.reshape((batch_size, -1, 1, 1, 1))   # [N, out_C, in_C, kH, kW]
    out = w2

    loss = paddle.square(out)
    dloss_dstyles = paddle.grad(outputs=[loss.sum()], inputs=[styles], create_graph=True)[0]


    out_paddle = out.numpy()
    ddd = np.mean((out_pytorch - out_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    dloss_dstyles_paddle = dloss_dstyles.numpy()
    ddd = np.mean((dloss_dstyles_pytorch - dloss_dstyles_paddle) ** 2)
    print('ddd=%.6f' % ddd)
    print()
print()
