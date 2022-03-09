
import torch
import numpy as np

import torch.nn.functional as F


def rrr(x):
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
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = w.sum(dim=[2,3,4])  # [NO]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = rrr(x)
        if demodulate and noise is not None:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        return x



dic = {}
batch_size = 2
for batch_idx in range(8):
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




    # x_shape[0] = batch_size
    w = torch.randn(w_shape)
    x = torch.randn(x_shape)
    styles = torch.randn(styles_shape)
    w.requires_grad_(True)
    x.requires_grad_(True)
    styles.requires_grad_(True)
    if noise_shape is None:
        noise = None
    else:
        noise = torch.randn(noise_shape)
        # noise.requires_grad_(True)
    if resample_filter_shape is None:
        resample_filter = None
    else:
        resample_filter = torch.randn(resample_filter_shape)
        # resample_filter.requires_grad_(True)

    y = fff(x, w, styles, noise, up=up, down=down, padding=padding, resample_filter=resample_filter,
                         demodulate=demodulate, flip_weight=flip_weight, fused_modconv=fused_modconv)

    dy_dx = torch.autograd.grad(outputs=[y.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]
    dy_dstyles = torch.autograd.grad(outputs=[y.sum()], inputs=[styles], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dy_dx'%batch_idx] = dy_dx.cpu().detach().numpy()
    dic['batch_%.3d.dy_dstyles'%batch_idx] = dy_dstyles.cpu().detach().numpy()
    dic['batch_%.3d.y'%batch_idx] = y.cpu().detach().numpy()
    dic['batch_%.3d.x'%batch_idx] = x.cpu().detach().numpy()
    dic['batch_%.3d.w'%batch_idx] = w.cpu().detach().numpy()
    dic['batch_%.3d.styles'%batch_idx] = styles.cpu().detach().numpy()
    if noise is not None:
        dic['batch_%.3d.noise'%batch_idx] = noise.cpu().detach().numpy()
    if resample_filter is not None:
        dic['batch_%.3d.resample_filter'%batch_idx] = resample_filter.cpu().detach().numpy()
np.savez('07_grad', **dic)
print()
