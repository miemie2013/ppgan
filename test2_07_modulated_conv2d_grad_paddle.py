
import paddle
import numpy as np
import paddle.nn.functional as F

from ppgan.models.generators.generator_styleganv2ada import modulated_conv2d, modulated_conv2d_grad


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

    y, x_1, x_2, x_mul_styles = modulated_conv2d(x, w, styles, noise, up=up, down=down, padding=padding, resample_filter=resample_filter,
                         demodulate=demodulate, flip_weight=flip_weight, fused_modconv=fused_modconv)
    # dy_dx = paddle.grad(outputs=[y.sum()], inputs=[x], create_graph=True)[0]
    # dy_dstyles = paddle.grad(outputs=[y.sum()], inputs=[styles], create_graph=True)[0]
    dysum_dy = paddle.ones(y.shape, dtype=paddle.float32)
    dy_dx, dy_dstyles = modulated_conv2d_grad(dysum_dy, x_1, x_2, x_mul_styles, x, w, styles, noise, up=up, down=down, padding=padding, resample_filter=resample_filter,
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
