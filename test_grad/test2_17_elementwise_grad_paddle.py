
import paddle
import numpy as np
import paddle.nn.functional as F





dic2 = np.load('17_grad.npz')
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

    batch_size = 2

    w = weight.unsqueeze(0) # [1, out_C, in_C]
    w1 = w * styles.reshape((batch_size, 1, -1)) # [N, out_C, in_C]
    dcoefs = w1.sum(axis=[2,])  # [N, out_C]
    out = w1 * dcoefs.reshape((batch_size, -1, 1))  # [N, out_C, in_C]
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
