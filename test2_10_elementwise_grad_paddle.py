
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


    w = weight.unsqueeze(0)  # [NOIkk]
    w0 = w
    w = w * styles.reshape((batch_size, 1, -1, 1, 1))  # [NOIkk]
    w1 = w
    _, _, D_w1_2, D_w1_3, D_w1_4 = w1.shape
    dcoefs = w.sum(axis=[2,3,4])  # [NO]

    x_mul_styles = x * paddle.cast(styles, dtype=x.dtype).reshape((batch_size, -1, 1, 1))
    rrr = F.sigmoid(x_mul_styles)
    out = rrr * paddle.cast(dcoefs, dtype=x.dtype).reshape((batch_size, -1, 1, 1))
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

    dloss_dloss = paddle.ones(loss.shape, dtype=paddle.float32)
    dloss_dout = dloss_dloss * 2 * out

    dloss_drrr = dloss_dout * paddle.cast(dcoefs, dtype=x.dtype).reshape((batch_size, -1, 1, 1))

    dloss_dstyles_1 = paddle.cast(dcoefs, dtype=x.dtype).reshape((batch_size, -1, 1, 1)) * dloss_dout   # du * v
    dloss_dstyles_2 = rrr * dloss_dout      # u * dv
    dloss_dstyles_2 = paddle.sum(dloss_dstyles_2, axis=[2, 3])


    dloss_dx_mul_styles = dloss_drrr * rrr * (1.0 - rrr)
    dloss_dstyles_1 = dloss_dstyles_1 * rrr * (1.0 - rrr)

    dloss_dx = dloss_dx_mul_styles * paddle.cast(styles, dtype=x.dtype).reshape((batch_size, -1, 1, 1))
    dloss_dstyles_1 = dloss_dstyles_1 * x
    dloss_dstyles_1 = paddle.sum(dloss_dstyles_1, axis=[2, 3])


    # sum()的求导有2步，先unsqueeze()再tile()，变回求和之前的形状。
    dloss_dstyles_2 = paddle.unsqueeze(dloss_dstyles_2, axis=[2, 3, 4])
    dloss_dstyles_2 = paddle.tile(dloss_dstyles_2, [1, 1, D_w1_2, D_w1_3, D_w1_4])
    dloss_dstyles_2 = dloss_dstyles_2 * w0
    dloss_dstyles_2 = paddle.sum(dloss_dstyles_2, axis=[1, 3, 4])


    dloss_dstyles = dloss_dstyles_1 + dloss_dstyles_2



    out_paddle = out.numpy()
    ddd = np.mean((out_pytorch - out_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    dloss_dstyles_paddle = dloss_dstyles.numpy()
    ddd = np.mean((dloss_dstyles_pytorch - dloss_dstyles_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    dloss_dx_paddle = dloss_dx.numpy()
    ddd = np.mean((dloss_dx_pytorch - dloss_dx_paddle) ** 2)
    print('ddd=%.6f' % ddd)
print()
