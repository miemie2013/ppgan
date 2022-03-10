
import paddle
import numpy as np
import paddle.nn.functional as F





dic2 = np.load('10_grad.npz')
for batch_idx in range(8):
    print('======================== batch_%.3d ========================'%batch_idx)

    dloss_dstyles_pytorch = dic2['batch_%.3d.dloss_dstyles'%batch_idx]
    dloss_dx_pytorch = dic2['batch_%.3d.dloss_dx'%batch_idx]
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

    x_mul_styles = x * paddle.cast(styles, dtype=x.dtype).reshape((batch_size, -1, 1, 1))
    rrr = F.sigmoid(x_mul_styles)
    out = rrr * paddle.cast(styles, dtype=x.dtype).reshape((batch_size, -1, 1, 1))
    loss = paddle.square(out)

    '''
    loss对styles的导数比较复杂，运算关系：
    out = rrr(x * styles) * styles = u * v
    dout_dstyles = u'v * x + uv' = du * v * x + u * dv
    = dconv2d_resample__dx_mul_styles * dx_mul_styles_dstyles * v + u * ddcoefs_dstyles
    = dconv2d_resample__dx_mul_styles * x * v + u * ddcoefs_dstyles

    dloss_dstyles = dloss_dout * dout_dstyles
    = dloss_dout * dconv2d_resample__dx_mul_styles * x * v   +   dloss_dout * u * ddcoefs_dstyles
    = dloss_dstyles_1 + dloss_dstyles_2

    其中u = rrr, v = styles, x = 未被复写的输入x
    du是conv2d_resample的输出x_2对conv2d_resample的输入(x * styles)的导数
    '''

    # dloss_dx = paddle.grad(outputs=[loss.sum()], inputs=[x], create_graph=True)[0]
    # dloss_dstyles = paddle.grad(outputs=[loss.sum()], inputs=[styles], create_graph=True)[0]

    dloss_dloss = paddle.ones(loss.shape, dtype=paddle.float32)
    dloss_dout = dloss_dloss * 2 * out

    # dloss_dstyles = dloss_dout
    u = rrr
    v = paddle.cast(styles, dtype=x.dtype).reshape((batch_size, -1, 1, 1))
    dloss_drrr = dloss_dout * v
    v_dloss_drrr = v * dloss_dout * v
    rrr_dloss_dv = rrr * dloss_dout * rrr
    rrr_dloss_dstyles = paddle.sum(rrr_dloss_dv, axis=[2, 3])
    # rrr_dloss_dstyles = paddle.reshape(rrr_dloss_dstyles, (batch_size, -1))


    dloss_dx_mul_styles = dloss_drrr * rrr * (1.0 - rrr)
    v_dloss_dx_mul_styles = v_dloss_drrr * rrr * (1.0 - rrr)

    dloss_dx = dloss_dx_mul_styles * paddle.cast(styles, dtype=x.dtype).reshape((batch_size, -1, 1, 1))
    v_dloss_dstyles = v_dloss_dx_mul_styles * x
    v_dloss_dstyles = paddle.sum(v_dloss_dstyles, axis=[2, 3])

    dloss_dstyles = rrr_dloss_dstyles + v_dloss_dstyles



    out_paddle = out.numpy()
    ddd = np.sum((out_pytorch - out_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    dloss_dstyles_paddle = dloss_dstyles.numpy()
    ddd = np.sum((dloss_dstyles_pytorch - dloss_dstyles_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    dloss_dx_paddle = dloss_dx.numpy()
    ddd = np.sum((dloss_dx_pytorch - dloss_dx_paddle) ** 2)
    print('ddd=%.6f' % ddd)
print()
