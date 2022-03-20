
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

    batch_size = 2

    # styles乘以任意数值就能触发这个bug。什么都不乘的话，不触发这个bug。
    C = 1.0
    w1 = styles * C
    # dcoefs = w1.sum(axis=[2,])  # [N, out_C]
    # out = w1 * dcoefs.reshape((batch_size, -1, 1))  # [N, out_C, in_C]
    out = w1 * paddle.sum(w1, axis=2, keepdim=True)
    loss = paddle.square(out)

    dloss_dstyles = paddle.grad(outputs=[loss.sum()], inputs=[styles], create_graph=True)[0]

    # paddle.grad()，dloss_dw1计算错误，把paddle.sum(w1, axis=2, keepdim=True)看成了常数，
    # 实际上paddle.sum(w1, axis=2, keepdim=True)里也有w1的表达式。
    dloss_dloss = paddle.ones(loss.shape, dtype=paddle.float32)
    dloss_dout = dloss_dloss * 2 * out
    dloss_dw1 = dloss_dout * paddle.sum(w1, axis=2, keepdim=True)
    dloss_dstyles222 = dloss_dw1 * C


    out_paddle = out.numpy()
    ddd = np.mean((out_pytorch - out_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    dloss_dstyles_paddle = dloss_dstyles.numpy()
    ddd = np.mean((dloss_dstyles_pytorch - dloss_dstyles_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    dloss_dstyles222_paddle = dloss_dstyles222.numpy()
    ddd = np.mean((dloss_dstyles222_paddle - dloss_dstyles_paddle) ** 2)
    print('ddd=%.6f' % ddd)
    print()
print()
