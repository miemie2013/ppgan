
import paddle
import numpy as np
import paddle.nn.functional as F



def conv2d_resample(x):
    return x + 100

def fff(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0)  # [NOIkk]
        w = w * styles.reshape((batch_size, 1, -1, 1, 1))  # [NOIkk]
    if demodulate:
        dcoefs = w.sum(axis=[2,3,4])  # [NO]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * paddle.cast(styles, dtype=x.dtype).reshape((batch_size, -1, 1, 1))
        x_mul_styles = x
        x = conv2d_resample(x)
        x_2 = x
        if demodulate and noise is not None:
            x = x * paddle.cast(dcoefs, dtype=x.dtype).reshape((batch_size, -1, 1, 1))
        elif demodulate:
            x = x * paddle.cast(dcoefs, dtype=x.dtype).reshape((batch_size, -1, 1, 1))
        return x, x_2, x_mul_styles




def conv2d_resample_grad(grad):
    return grad

def fff_grad(dloss_dout, x_2, x_mul_styles, 
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0)  # [NOIkk]
        w0 = w
        w = w * styles.reshape((batch_size, 1, -1, 1, 1))  # [NOIkk]
        w1 = w
    if demodulate:
        dcoefs = w.sum(axis=[2,3,4])  # [NO]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        # x = x * paddle.cast(styles, dtype=x.dtype).reshape((batch_size, -1, 1, 1))
        # x = conv2d_resample(x)
        # if demodulate and noise is not None:
        #     x = x * paddle.cast(dcoefs, dtype=x.dtype).reshape((batch_size, -1, 1, 1))
        # elif demodulate:
        #     x = x * paddle.cast(dcoefs, dtype=x.dtype).reshape((batch_size, -1, 1, 1))

        '''
        loss对styles的导数比较复杂，因为本层中styles可能被使用了多次，比如，
        dcoefs的表达式包含有styles，x和styles相乘之后，又和dcoefs相乘。
        为了求出loss对styles的导数，

        简化运算关系：
        out = conv2d_resample(x * styles) * dcoefs(styles) = u * v
        dout_dstyles = u'v * x + uv' = du * v * x + u * dv
        = dout_dx_mul_styles * dx_mul_styles_dstyles * v + u * ddcoefs_dstyles
        = dout_dx_mul_styles * x * v + u * ddcoefs_dstyles

        其中u = x_2, v = dcoefs, x = 未被复写的输入x
        du是out对conv2d_resample的输入(x * styles)的导数
        '''
        dloss_dx = dloss_dout
        if demodulate and noise is not None:
            u = x_2
            v = paddle.cast(dcoefs, dtype=x.dtype).reshape((batch_size, -1, 1, 1))

            dout_dout = paddle.ones(dloss_dout.shape, dtype=paddle.float32)
            dout_dx_mul_styles, _ = conv2d_resample_grad(dout_dout, x_1, x_mul_styles,
                                                         paddle.cast(weight, dtype=x.dtype), filter=resample_filter,
                                                         up=up, down=down, padding=padding, flip_weight=flip_weight)

        elif demodulate:
            # x = x * paddle.cast(dcoefs, dtype=x.dtype).reshape((batch_size, -1, 1, 1))
            raise NotImplementedError("not implemented.")
        elif noise is not None:
            pass

        dloss_dx, dloss_dweight = conv2d_resample_grad(dloss_dx, x_1, x, paddle.cast(weight, dtype=x.dtype), filter=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)

        dloss_dx = dloss_dx * paddle.cast(styles, dtype=x.dtype).reshape((batch_size, -1, 1, 1))

        if demodulate and fused_modconv:
            # 不可能执行这个，因为fused_modconv肯定是False
            pass
        if demodulate:
            # dcoefs = (w.square().sum(axis=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
            ddcoefs_ddcoefs = paddle.ones(dcoefs.shape, dtype=paddle.float32)
            ddcoefs_dw_square_sum_add_1e8 = -0.5 * ddcoefs_ddcoefs * dcoefs * dcoefs * dcoefs
            ddcoefs_dw_square_sum = ddcoefs_dw_square_sum_add_1e8
            ddcoefs_dw_square = paddle.unsqueeze(ddcoefs_dw_square_sum, axis=[2, 3, 4])
            ddcoefs_dw_square = paddle.tile(ddcoefs_dw_square, [1, 1, w.shape[2], w.shape[3], w.shape[4]])
            ddcoefs_dw1 = ddcoefs_dw_square * 2 * w
        if demodulate or fused_modconv:
            # w = weight.unsqueeze(0)  # [NOIkk]
            # w = w * styles.reshape((batch_size, 1, -1, 1, 1))  # [NOIkk]
            ddcoefs_dstyles = ddcoefs_dw1 * w0
            ddcoefs_dstyles = paddle.sum(ddcoefs_dstyles, axis=[1, 3, 4])
            ddcoefs_dstyles = paddle.unsqueeze(ddcoefs_dstyles, axis=[2, 3])
            aaaaaaaaaaa1 = dout_dx_mul_styles
            aaaaaaaaaaa2 = x
            aaaaaaaaaaa3 = v
            aaaaaaaaaaa4 = u
            aaaaaaaaaaa5 = ddcoefs_dstyles
            aaaaaaaaaaa6 = ddcoefs_dw1
            aaaaaaaaaaa7 = w0
            dout_dstyles = dout_dx_mul_styles * x * v + u * ddcoefs_dstyles
            print()
            dloss_dstyles = dloss_dout * dout_dstyles
            dloss_dstyles = paddle.sum(dloss_dstyles, axis=[2, 3])


        print('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
        return dloss_dx, dloss_dstyles



dic2 = np.load('07_grad.npz')
for batch_idx in range(8):
    print('======================== batch_%.3d ========================'%batch_idx)
    x_shape = [2, 512, 4, 4]
    w_shape = [512, 512, 3, 3]
    styles_shape = [2, 512]
    noise_shape = [2, 1, 4, 4]
    up = 1
    down = 1
    padding = 1
    resample_filter_shape = [4, 4]
    demodulate = True
    flip_weight = True
    fused_modconv = False

    x_shape = [2, 512, 4, 4]
    w_shape = [3, 512, 1, 1]
    styles_shape = [2, 512]
    noise_shape = None
    up = 1
    down = 1
    padding = 0
    resample_filter_shape = None
    demodulate = False
    flip_weight = True
    fused_modconv = False

    x_shape = [2, 512, 4, 4]
    w_shape = [512, 512, 3, 3]
    styles_shape = [2, 512]
    noise_shape = [2, 1, 8, 8]
    up = 2
    down = 1
    padding = 1
    resample_filter_shape = [4, 4]
    demodulate = True
    flip_weight = False
    fused_modconv = False

    x_shape = [2, 512, 8, 8]
    w_shape = [512, 512, 3, 3]
    styles_shape = [2, 512]
    noise_shape = [2, 1, 8, 8]
    up = 1
    down = 1
    padding = 1
    resample_filter_shape = [4, 4]
    demodulate = True
    flip_weight = True
    fused_modconv = False

    x_shape = [2, 512, 8, 8]
    w_shape = [3, 512, 1, 1]
    styles_shape = [2, 512]
    noise_shape = None
    up = 1
    down = 1
    padding = 0
    resample_filter_shape = None
    demodulate = False
    flip_weight = True
    fused_modconv = False

    x_shape = [2, 512, 8, 8]
    w_shape = [512, 512, 3, 3]
    styles_shape = [2, 512]
    noise_shape = [2, 1, 16, 16]
    up = 2
    down = 1
    padding = 1
    resample_filter_shape = [4, 4]
    demodulate = True
    flip_weight = False
    fused_modconv = False

    x_shape = [2, 512, 16, 16]
    w_shape = [512, 512, 3, 3]
    styles_shape = [2, 512]
    noise_shape = [2, 1, 16, 16]
    up = 1
    down = 1
    padding = 1
    resample_filter_shape = [4, 4]
    demodulate = True
    flip_weight = True
    fused_modconv = False

    # x_shape = [2, 512, 16, 16]
    # w_shape = [3, 512, 1, 1]
    # styles_shape = [2, 512]
    # noise_shape = None
    # up = 1
    # down = 1
    # padding = 0
    # resample_filter_shape = None
    # demodulate = False
    # flip_weight = True
    # fused_modconv = False

    # x_shape = [2, 512, 16, 16]
    # w_shape = [512, 512, 3, 3]
    # styles_shape = [2, 512]
    # noise_shape = [2, 1, 32, 32]
    # up = 2
    # down = 1
    # padding = 1
    # resample_filter_shape = [4, 4]
    # demodulate = True
    # flip_weight = False
    # fused_modconv = False

    # x_shape = [2, 512, 32, 32]
    # w_shape = [512, 512, 3, 3]
    # styles_shape = [2, 512]
    # noise_shape = [2, 1, 32, 32]
    # up = 1
    # down = 1
    # padding = 1
    # resample_filter_shape = [4, 4]
    # demodulate = True
    # flip_weight = True
    # fused_modconv = False

    # x_shape = [2, 512, 32, 32]
    # w_shape = [3, 512, 1, 1]
    # styles_shape = [2, 512]
    # noise_shape = None
    # up = 1
    # down = 1
    # padding = 0
    # resample_filter_shape = None
    # demodulate = False
    # flip_weight = True
    # fused_modconv = False

    # x_shape = [2, 512, 32, 32]
    # w_shape = [512, 512, 3, 3]
    # styles_shape = [2, 512]
    # noise_shape = [2, 1, 64, 64]
    # up = 2
    # down = 1
    # padding = 1
    # resample_filter_shape = [4, 4]
    # demodulate = True
    # flip_weight = False
    # fused_modconv = False

    # x_shape = [2, 512, 64, 64]
    # w_shape = [512, 512, 3, 3]
    # styles_shape = [2, 512]
    # noise_shape = [2, 1, 64, 64]
    # up = 1
    # down = 1
    # padding = 1
    # resample_filter_shape = [4, 4]
    # demodulate = True
    # flip_weight = True
    # fused_modconv = False

    # x_shape = [2, 512, 64, 64]
    # w_shape = [3, 512, 1, 1]
    # styles_shape = [2, 512]
    # noise_shape = None
    # up = 1
    # down = 1
    # padding = 0
    # resample_filter_shape = None
    # demodulate = False
    # flip_weight = True
    # fused_modconv = False

    # x_shape = [2, 512, 64, 64]
    # w_shape = [256, 512, 3, 3]
    # styles_shape = [2, 512]
    # noise_shape = [2, 1, 128, 128]
    # up = 2
    # down = 1
    # padding = 1
    # resample_filter_shape = [4, 4]
    # demodulate = True
    # flip_weight = False
    # fused_modconv = False

    # x_shape = [2, 256, 128, 128]
    # w_shape = [256, 256, 3, 3]
    # styles_shape = [2, 256]
    # noise_shape = [2, 1, 128, 128]
    # up = 1
    # down = 1
    # padding = 1
    # resample_filter_shape = [4, 4]
    # demodulate = True
    # flip_weight = True
    # fused_modconv = False

    # x_shape = [2, 256, 128, 128]
    # w_shape = [3, 256, 1, 1]
    # styles_shape = [2, 256]
    # noise_shape = None
    # up = 1
    # down = 1
    # padding = 0
    # resample_filter_shape = None
    # demodulate = False
    # flip_weight = True
    # fused_modconv = False

    # x_shape = [2, 256, 128, 128]
    # w_shape = [128, 256, 3, 3]
    # styles_shape = [2, 256]
    # noise_shape = [2, 1, 256, 256]
    # up = 2
    # down = 1
    # padding = 1
    # resample_filter_shape = [4, 4]
    # demodulate = True
    # flip_weight = False
    # fused_modconv = False

    # x_shape = [2, 128, 256, 256]
    # w_shape = [128, 128, 3, 3]
    # styles_shape = [2, 128]
    # noise_shape = [2, 1, 256, 256]
    # up = 1
    # down = 1
    # padding = 1
    # resample_filter_shape = [4, 4]
    # demodulate = True
    # flip_weight = True
    # fused_modconv = False

    # x_shape = [2, 128, 256, 256]
    # w_shape = [3, 128, 1, 1]
    # styles_shape = [2, 128]
    # noise_shape = None
    # up = 1
    # down = 1
    # padding = 0
    # resample_filter_shape = None
    # demodulate = False
    # flip_weight = True
    # fused_modconv = False

    # x_shape = [2, 128, 256, 256]
    # w_shape = [64, 128, 3, 3]
    # styles_shape = [2, 128]
    # noise_shape = [2, 1, 512, 512]
    # up = 2
    # down = 1
    # padding = 1
    # resample_filter_shape = [4, 4]
    # demodulate = True
    # flip_weight = False
    # fused_modconv = False

    # x_shape = [2, 64, 512, 512]
    # w_shape = [64, 64, 3, 3]
    # styles_shape = [2, 64]
    # noise_shape = [2, 1, 512, 512]
    # up = 1
    # down = 1
    # padding = 1
    # resample_filter_shape = [4, 4]
    # demodulate = True
    # flip_weight = True
    # fused_modconv = False

    # x_shape = [2, 64, 512, 512]
    # w_shape = [3, 64, 1, 1]
    # styles_shape = [2, 64]
    # noise_shape = None
    # up = 1
    # down = 1
    # padding = 0
    # resample_filter_shape = None
    # demodulate = False
    # flip_weight = True
    # fused_modconv = False





    dy_dx_pytorch = dic2['batch_%.3d.dy_dx'%batch_idx]
    dy_dstyles_pytorch = dic2['batch_%.3d.dy_dstyles'%batch_idx]
    y_pytorch = dic2['batch_%.3d.y'%batch_idx]
    w = dic2['batch_%.3d.w'%batch_idx]
    x = dic2['batch_%.3d.x'%batch_idx]
    styles = dic2['batch_%.3d.styles'%batch_idx]

    x = paddle.to_tensor(x)
    x.stop_gradient = False
    w = paddle.to_tensor(w)
    w.stop_gradient = False
    styles = paddle.to_tensor(styles)
    styles.stop_gradient = False
    if 'batch_%.3d.noise'%batch_idx in dic2.keys():
        noise = paddle.to_tensor(dic2['batch_%.3d.noise'%batch_idx])
    else:
        noise = None
    if 'batch_%.3d.resample_filter'%batch_idx in dic2.keys():
        resample_filter = paddle.to_tensor(dic2['batch_%.3d.resample_filter'%batch_idx])
    else:
        resample_filter = None

    y, x_2, x_mul_styles = fff(x, w, styles, noise, up=up, down=down, padding=padding, resample_filter=resample_filter,
                         demodulate=demodulate, flip_weight=flip_weight, fused_modconv=fused_modconv)
    # y = fff(x, w, styles, noise, up=up, down=down, padding=padding, resample_filter=resample_filter,
    #                      demodulate=demodulate, flip_weight=flip_weight, fused_modconv=fused_modconv)
    # dy_dx = paddle.grad(outputs=[y.sum()], inputs=[x], create_graph=True)[0]
    # dy_dstyles = paddle.grad(outputs=[y.sum()], inputs=[styles], create_graph=True)[0]
    dysum_dy = paddle.ones(y.shape, dtype=paddle.float32)
    dy_dx, dy_dstyles = fff_grad(dysum_dy, x_2, x_mul_styles, x, w, styles, noise, up=up, down=down, padding=padding, resample_filter=resample_filter,
                                              demodulate=demodulate, flip_weight=flip_weight, fused_modconv=fused_modconv)

    aaaaaa = y.numpy()
    ddd = np.mean((y_pytorch - aaaaaa) ** 2)
    print('ddd=%.6f' % ddd)

    dy_dx_paddle = dy_dx.numpy()
    ddd = np.mean((dy_dx_pytorch - dy_dx_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    dy_dstyles_paddle = dy_dstyles.numpy()
    ddd = np.mean((dy_dstyles_pytorch - dy_dstyles_paddle) ** 2)
    print('ddd=%.6f' % ddd)
print()
