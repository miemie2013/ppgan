
import paddle
import math
import numpy as np
import paddle.nn.functional as F




scale = 1.0 / math.sqrt(3)

dic2 = np.load('18_grad.npz')
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


    style = styles.reshape((batch_size, 1, in_C, 1, 1))
    weight = scale * weight * style
    demod = paddle.rsqrt((weight * weight).sum([2, 3, 4]) + 1e-8)
    weight = weight * demod.reshape((batch_size, out_C, 1, 1, 1))

    out = weight
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
