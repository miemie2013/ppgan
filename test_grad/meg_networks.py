import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np
import math
import scipy
import scipy.signal
import warnings


#----------------------------------------------------------------------------
def mge_grad(gm2, output, inputs, clear_grad_nets):
    gm2.backward(output)
    dydxs = []
    for input in inputs:
        dydx = input.grad
        dydxs.append(dydx)
        # 清除梯度
        input.grad = None
    for net in clear_grad_nets:
        for m in net.parameters():
            m.grad = None
    return dydxs


#----------------------------------------------------------------------------

def _unbroadcast(x, shape):
    extra_dims = x.ndim - len(shape)
    assert extra_dims >= 0
    dim = [i for i in range(x.ndim) if x.shape[i] > 1 and (i < extra_dims or shape[i - extra_dims] == 1)]
    if len(dim):
        x = x.sum(dim=dim, keepdim=True)
    if extra_dims:
        x = x.reshape(-1, *x.shape[extra_dims+1:])
    assert x.shape == shape
    return x

#----------------------------------------------------------------------------



def _get_filter_size(filter):
    if filter is None:
        return 1, 1
    fw = filter.shape[-1]
    fh = filter.shape[0]
    return fw, fh

def upfirdn2d_setup_filter(shape, normalize=True, flip_filter=False, gain=1, separable=None):
    r"""Convenience function to setup 2D FIR filter for `upfirdn2d()`.

    Args:
        shape:       Torch tensor, numpy array, or python list of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable),
                     `[]` (impulse), or
                     `None` (identity).
        normalize:   Normalize the filter so that it retains the magnitude
                     for constant input signal (DC)? (default: True).
        flip_filter: Flip the filter? (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        separable:   Return a separable filter? (default: select automatically).

    Returns:
        Float32 tensor of the shape
        `[filter_height, filter_width]` (non-separable) or
        `[filter_taps]` (separable).
    """
    # Validate.
    if shape is None:
        shape = 1
    shape = mge.tensor(shape, dtype=np.float32)
    assert shape.ndim in [0, 1, 2]
    assert shape.size > 0
    if shape.ndim == 0:
        shape = shape[np.newaxis]

    # Separable?
    if separable is None:
        separable = (shape.ndim == 1 and shape.size >= 8)
    if shape.ndim == 1 and not separable:
        # ger()相当于向量自乘
        shape = F.expand_dims(shape, 1)  # [n, 1]
        shape = F.matmul(shape, shape.transpose((1, 0)))  # [n, n]
    assert shape.ndim == (1 if separable else 2)

    # Apply normalize, flip, gain, and device.
    if normalize:
        shape /= shape.sum()
    if flip_filter:
        if shape.ndim == 1:
            shape = shape[::-1]
        elif shape.ndim == 2:
            shape = shape[::-1, ::-1]
        elif shape.ndim == 3:
            shape = shape[::-1, ::-1, ::-1]
        elif shape.ndim == 4:
            shape = shape[::-1, ::-1, ::-1, ::-1]
        else:
            raise NotImplementedError("ndim \'{}\' is not implemented.".format(shape.ndim))
    shape = shape * (gain ** (shape.ndim / 2))
    return shape


def bias_act(x, b=None, dim=1, act='linear', alpha=None, gain=None, clamp=None):
    assert clamp is None or clamp >= 0
    def_gain = 1.0
    if act in ['relu', 'lrelu', 'swish']:  # 除了这些激活函数的def_gain = np.sqrt(2)，其余激活函数的def_gain = 1.0
        def_gain = np.sqrt(2)
    def_alpha = 0.0
    if act in ['lrelu']:  # 除了这些激活函数的def_alpha = 0.2，其余激活函数的def_alpha = 0.0
        def_alpha = 0.2

    alpha = float(alpha if alpha is not None else def_alpha)
    gain = float(gain if gain is not None else def_gain)
    clamp = float(clamp if clamp is not None else -1)

    # 加上偏移
    if b is not None:
        new_shape = [-1 if i == dim else 1 for i in range(x.ndim)]
        b_ = F.reshape(b, new_shape)
        x = x + b_

    # 经过激活函数
    alpha = float(alpha)  # 只有leaky_relu需要
    if act == 'linear':
        pass
    elif act == 'relu':
        x = F.relu(x)
    elif act == 'lrelu':
        x = F.leaky_relu(x, alpha)
    elif act == 'tanh':
        x = F.tanh(x)
    elif act == 'sigmoid':
        x = F.sigmoid(x)
    elif act == 'elu':
        x = F.elu(x)
    elif act == 'selu':
        x = F.selu(x)
    elif act == 'softplus':
        x = F.softplus(x)
    elif act == 'swish':
        x = F.sigmoid(x) * x
    else:
        raise NotImplementedError("activation \'{}\' is not implemented.".format(act))

    # 乘以缩放因子
    gain = float(gain)
    if gain != 1:
        x = x * gain

    # 限制范围
    if clamp >= 0:
        x = F.clip(x, -clamp, clamp)
    return x


def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 2:
        padx, pady = padding
        padding = [padx, padx, pady, pady]
    padx0, padx1, pady0, pady1 = padding
    return padx0, padx1, pady0, pady1


def _conv2d_wrapper(x, w, stride=1, padding=0, groups=1, transpose=False, flip_weight=True):
    """Wrapper for the underlying `conv2d()` and `conv_transpose2d()` implementations.
    """
    out_channels, in_channels_per_group, kh, kw = w.shape

    # Flip weight if requested.
    if not flip_weight: # conv2d() actually performs correlation (flip_weight=True) not convolution (flip_weight=False).
        w = w[:, :, ::-1, ::-1]

    # Workaround performance pitfall in cuDNN 8.0.5, triggered when using
    # 1x1 kernel + memory_format=channels_last + less than 64 channels.
    if kw == 1 and kh == 1 and stride == 1 and padding in [0, [0, 0], (0, 0)] and not transpose:
        if x.shape[2] * x.shape[3] == 1 and min(out_channels, in_channels_per_group) < 64:
            if out_channels <= 4 and groups == 1:
                in_shape = x.shape
                x = w.squeeze(3).squeeze(2) @ x.reshape([in_shape[0], in_channels_per_group, -1])
                x = x.reshape([in_shape[0], out_channels, in_shape[2], in_shape[3]])
            else:
                x = x.astype(memory_format=torch.contiguous_format)
                w = w.astype(memory_format=torch.contiguous_format)
                x = F.conv2d(x, w, groups=groups)
            return x.astype(memory_format=torch.channels_last)

    # Otherwise => execute using conv2d_gradfix.
    if transpose:
        if groups != 1:
            out_c, in_c, kH, kW = w.shape
            w = w.reshape((groups, -1, in_c, kH, kW))
        return F.conv_transpose2d(x, weight=w, bias=None, stride=stride, padding=padding, groups=groups, dilation=1)
    else:
        if groups != 1:
            out_c, in_c, kH, kW = w.shape
            w = w.reshape((groups, -1, in_c, kH, kW))
        return F.conv2d(x, weight=w, bias=None, stride=stride, padding=padding, dilation=1, groups=groups)

def _parse_scaling(scaling):
    # scaling 一变二
    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy = scaling
    assert sx >= 1 and sy >= 1
    return sx, sy

def upfirdn2d(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Slow reference implementation of `upfirdn2d()` using standard PyTorch ops.
    """
    # Validate arguments.
    assert isinstance(x, mge.Tensor) and x.ndim == 4
    if f is None:
        f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
    assert isinstance(f, mge.Tensor) and f.ndim in [1, 2]
    # assert f.dtype == np.float32 and not f.requires_grad
    assert f.dtype == np.float32
    batch_size, num_channels, in_height, in_width = x.shape
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)

    # Upsample by inserting zeros.
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    x = F.pad(x, ((0, 0), (0, 0), (0, 0), (0, upx - 1), (0, 0), (0, upy - 1)))
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])

    # Pad or crop.
    x = F.pad(x, ((0, 0), (0, 0), (max(padx0, 0), max(padx1, 0)), (max(pady0, 0), max(pady1, 0))))
    x = x[:, :, max(-pady0, 0) : x.shape[2] - max(-pady1, 0), max(-padx0, 0) : x.shape[3] - max(-padx1, 0)]

    # Setup filter.
    f = f * (gain ** (f.ndim / 2))
    f = f.astype(x.dtype)
    if not flip_filter:
        if f.ndim == 1:
            f = f[::-1]
        elif f.ndim == 2:
            f = f[::-1, ::-1]
        elif f.ndim == 3:
            f = f[::-1, ::-1, ::-1]
        elif f.ndim == 4:
            f = f[::-1, ::-1, ::-1, ::-1]
        else:
            raise NotImplementedError("ndim \'{}\' is not implemented.".format(f.ndim))

    # Convolve with the filter.
    rps = [num_channels, 1] + [1] * f.ndim
    f = F.expand_dims(f, 0)
    f = F.expand_dims(f, 0)
    f = F.tile(f, rps)
    if f.ndim == 4:
        groups = num_channels
        out_c, in_c, kH, kW = f.shape
        f = f.reshape((groups, -1, in_c, kH, kW))
        x = F.conv2d(x, weight=f, groups=groups)
    else:
        groups = num_channels
        f1 = F.expand_dims(f, 2)
        f2 = F.expand_dims(f, 3)
        out_c, in_c, kH, kW = f1.shape
        f1 = f1.reshape((groups, -1, in_c, kH, kW))
        out_c, in_c, kH, kW = f2.shape
        f2 = f2.reshape((groups, -1, in_c, kH, kW))
        x = F.conv2d(x, weight=f1, groups=groups)
        x = F.conv2d(x, weight=f2, groups=groups)

    # Downsample by throwing away pixels.
    x = x[:, :, ::downy, ::downx]
    return x


def downsample2d(x, f, down=2, padding=0, flip_filter=False, gain=1):
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw - downx + 1) // 2,
        padx1 + (fw - downx) // 2,
        pady0 + (fh - downy + 1) // 2,
        pady1 + (fh - downy) // 2,
    ]
    return upfirdn2d(x, f, down=down, padding=p, flip_filter=flip_filter, gain=gain)


def conv2d_resample(x, w, filter=None, up=1, down=1, padding=0, groups=1, flip_weight=True, flip_filter=False):
    r""" 2D卷积（带有上采样或者下采样）
    Padding只在最开始执行一次.

    Args:
        x:              Input tensor of shape
                        `[batch_size, in_channels, in_height, in_width]`.
        w:              Weight tensor of shape
                        `[out_channels, in_channels//groups, kernel_height, kernel_width]`.
        filter:         Low-pass filter for up/downsampling. Must be prepared beforehand by
                        calling upfirdn2d.setup_filter(). None = identity (default).
        up:             Integer upsampling factor (default: 1).
        down:           Integer downsampling factor (default: 1).
        padding:        Padding with respect to the upsampled image. Can be a single number
                        or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                        (default: 0).
        groups:         Split input channels into N groups (default: 1).
        flip_weight:    False = convolution, True = correlation (default: True).
        flip_filter:    False = convolution, True = correlation (default: False).
    """
    assert isinstance(up, int) and (up >= 1)
    assert isinstance(down, int) and (down >= 1)
    assert isinstance(groups, int) and (groups >= 1)
    out_channels, in_channels_per_group, kh, kw = w.shape
    fw, fh = _get_filter_size(filter)
    # 图片4条边上的padding
    px0, px1, py0, py1 = _parse_padding(padding)

    # Adjust padding to account for up/downsampling.
    if up > 1:
        px0 += (fw + up - 1) // 2
        px1 += (fw - up) // 2
        py0 += (fh + up - 1) // 2
        py1 += (fh - up) // 2
    if down > 1:
        px0 += (fw - down + 1) // 2
        px1 += (fw - down) // 2
        py0 += (fh - down + 1) // 2
        py1 += (fh - down) // 2

    # Fast path: 1x1 convolution with downsampling only => downsample first, then convolve.
    if kw == 1 and kh == 1 and (down > 1 and up == 1):
        x = upfirdn2d(x, filter, down=down, padding=[px0, px1, py0, py1], flip_filter=flip_filter)
        x = _conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
        return x

    # Fast path: 1x1 convolution with upsampling only => convolve first, then upsample.
    if kw == 1 and kh == 1 and (up > 1 and down == 1):
        x = _conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
        x = upfirdn2d(x, filter, up=up, padding=[px0, px1, py0, py1], gain=up ** 2, flip_filter=flip_filter)
        return x

    # Fast path: downsampling only => use strided convolution.
    if down > 1 and up == 1:
        x = upfirdn2d(x, filter, padding=[px0, px1, py0, py1], flip_filter=flip_filter)
        x = _conv2d_wrapper(x=x, w=w, stride=down, groups=groups, flip_weight=flip_weight)
        return x

    # Fast path: upsampling with optional downsampling => use transpose strided convolution.
    if up > 1:
        if groups == 1:
            w = F.transpose(w, (1, 0, 2, 3))
        else:
            w = w.reshape((groups, out_channels // groups, in_channels_per_group, kh, kw))
            w = F.transpose(w, (0, 2, 1, 3, 4))
            w = w.reshape((groups * in_channels_per_group, out_channels // groups, kh, kw))
        px0 -= kw - 1
        px1 -= kw - up
        py0 -= kh - 1
        py1 -= kh - up
        pxt = max(min(-px0, -px1), 0)
        pyt = max(min(-py0, -py1), 0)
        x = _conv2d_wrapper(x=x, w=w, stride=up, padding=[pyt,pxt], groups=groups, transpose=True, flip_weight=(not flip_weight))
        x = upfirdn2d(x, filter, padding=[px0 + pxt, px1 + pxt, py0 + pyt, py1 + pyt], gain=up ** 2, flip_filter=flip_filter)
        if down > 1:
            x = upfirdn2d(x, filter, down=down, flip_filter=flip_filter)
        return x

    # Fast path: no up/downsampling, padding supported by the underlying implementation => use plain conv2d.
    if up == 1 and down == 1:
        if px0 == px1 and py0 == py1 and px0 >= 0 and py0 >= 0:
            return _conv2d_wrapper(x=x, w=w, padding=[py0,px0], groups=groups, flip_weight=flip_weight)

    # Fallback: Generic reference implementation.
    x = upfirdn2d(x, (filter if up > 1 else None), up=up, padding=[px0, px1, py0, py1], gain=up ** 2, flip_filter=flip_filter)
    x = _conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
    if down > 1:
        x = upfirdn2d(x, filter, down=down, flip_filter=flip_filter)
    return x


def upsample2d(x, f, up=2, padding=0, flip_filter=False, gain=1):
    r"""Upsample a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape is a multiple of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the output. Can be a single number or a
                     list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    upx, upy = _parse_scaling(up)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw + upx - 1) // 2,
        padx1 + (fw - upx) // 2,
        pady0 + (fh + upy - 1) // 2,
        pady1 + (fh - upy) // 2,
    ]
    return upfirdn2d(x, f, up=up, padding=p, flip_filter=flip_filter, gain=gain*upx*upy)


class Conv2dLayer(M.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.resample_filter = mge.Tensor(upfirdn2d_setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

        # activation_funcs = {
        #     'linear':   dnnlib.EasyDict(func=lambda x, **_:         x,                                          def_alpha=0,    def_gain=1,             cuda_idx=1, ref='',  has_2nd_grad=False),
        #     'relu':     dnnlib.EasyDict(func=lambda x, **_:         torch.nn.functional.relu(x),                def_alpha=0,    def_gain=np.sqrt(2),    cuda_idx=2, ref='y', has_2nd_grad=False),
        #     'lrelu':    dnnlib.EasyDict(func=lambda x, alpha, **_:  torch.nn.functional.leaky_relu(x, alpha),   def_alpha=0.2,  def_gain=np.sqrt(2),    cuda_idx=3, ref='y', has_2nd_grad=False),
        #     'tanh':     dnnlib.EasyDict(func=lambda x, **_:         torch.tanh(x),                              def_alpha=0,    def_gain=1,             cuda_idx=4, ref='y', has_2nd_grad=True),
        #     'sigmoid':  dnnlib.EasyDict(func=lambda x, **_:         torch.sigmoid(x),                           def_alpha=0,    def_gain=1,             cuda_idx=5, ref='y', has_2nd_grad=True),
        #     'elu':      dnnlib.EasyDict(func=lambda x, **_:         torch.nn.functional.elu(x),                 def_alpha=0,    def_gain=1,             cuda_idx=6, ref='y', has_2nd_grad=True),
        #     'selu':     dnnlib.EasyDict(func=lambda x, **_:         torch.nn.functional.selu(x),                def_alpha=0,    def_gain=1,             cuda_idx=7, ref='y', has_2nd_grad=True),
        #     'softplus': dnnlib.EasyDict(func=lambda x, **_:         torch.nn.functional.softplus(x),            def_alpha=0,    def_gain=1,             cuda_idx=8, ref='y', has_2nd_grad=True),
        #     'swish':    dnnlib.EasyDict(func=lambda x, **_:         torch.sigmoid(x) * x,                       def_alpha=0,    def_gain=np.sqrt(2),    cuda_idx=9, ref='x', has_2nd_grad=True),
        # }
        def_gain = 1.0
        if activation in ['relu', 'lrelu', 'swish']:  # 除了这些激活函数的def_gain = np.sqrt(2)，其余激活函数的def_gain = 1.0
            def_gain = np.sqrt(2)
        def_alpha = 0.0
        if activation in ['lrelu']:  # 除了这些激活函数的def_alpha = 0.2，其余激活函数的def_alpha = 0.0
            def_alpha = 0.2


        self.act_gain = def_gain

        weight = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32)
        bias = np.zeros([out_channels], dtype=np.float32) if bias else None
        if trainable:
            self.weight = mge.Parameter(mge.tensor(weight))
            self.bias = mge.Parameter(mge.tensor(bias)) if bias is not None else None
        else:
            self.weight = mge.Tensor(weight)
            if bias is not None:
                self.bias = mge.Tensor(bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.astype(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1) # slightly faster
        x = conv2d_resample(x=x, w=w.astype(x.dtype), filter=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


class FullyConnectedLayer(M.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = mge.Parameter(mge.tensor(np.random.randn(out_features, in_features).astype(np.float32)) / lr_multiplier)
        self.bias = mge.Parameter(mge.tensor(np.ones(out_features, ).astype(np.float32) * bias_init)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.weight_gain = float(self.weight_gain)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.astype(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.astype(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = F.matmul(x, w, transpose_b=True) + F.expand_dims(b, 0)
        else:
            x = F.matmul(x, w, transpose_b=True)
            x = bias_act(x, b, act=self.activation)
        return x


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class StyleGANv2ADA_MappingNetwork(M.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.w_avg = mge.Tensor(np.zeros([w_dim], dtype=np.float32))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        if self.z_dim > 0:
            x = normalize_2nd_moment(z.astype(torch.float32))
        if self.c_dim > 0:
            y = normalize_2nd_moment(self.embed(c.astype(torch.float32)))
            x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))
            # bz = x.shape[0]
            # gpu0_w_avg = x.detach()[:bz//2].mean(dim=0).lerp(self.w_avg, self.w_avg_beta)
            # gpu1_w_avg = x.detach()[bz//2:].mean(dim=0).lerp(self.w_avg, self.w_avg_beta)
            # self.w_avg.copy_(gpu0_w_avg)

        # Broadcast.
        if self.num_ws is not None:
            x = F.expand_dims(x, 1)
            x = F.tile(x, [1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            assert self.w_avg_beta is not None
            if self.num_ws is None or truncation_cutoff is None:
                x = self.w_avg.lerp(x, truncation_psi)
            else:
                x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

def modulated_conv2d(
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

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == np.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = F.expand_dims(weight, 0)  # [NOIkk]
        w = w * styles.reshape((batch_size, 1, -1, 1, 1))  # [NOIkk]
    if demodulate:
        temp = F.square(w)
        temp = temp.sum([2, 3, 4]) + 1e-8
        temp = F.sqrt(temp)  # [NO]
        dcoefs = 1.0 / temp  # [NO]
        # dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape((batch_size, -1, 1, 1, 1))  # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.astype(x.dtype).reshape((batch_size, -1, 1, 1))
        x = conv2d_resample(x=x, w=weight.astype(x.dtype), filter=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = x * dcoefs.astype(x.dtype).reshape((batch_size, -1, 1, 1)) + noise.astype(x.dtype)
        elif demodulate:
            x = x * dcoefs.astype(x.dtype).reshape((batch_size, -1, 1, 1))
        elif noise is not None:
            x = x + noise.astype(x.dtype)
        return x

    # Execute as one fused op using grouped convolution.
    batch_size = int(batch_size)
    x = x.reshape((1, -1, *x.shape[2:]))
    w = w.reshape((-1, in_channels, kh, kw))
    x = conv2d_resample(x=x, w=w.astype(x.dtype), filter=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape((batch_size, -1, *x.shape[2:]))
    if noise is not None:
        # 注意，在最高的N个分辨率中，x是float16，而noise是float32，若写作x = x + noise，则x是float32，这是个错误的写法；
        # 若写作x = x.add_(noise)，则x是float16；
        # 若写作x = x + noise.astype(x.dtype)，则x是float16。
        x = x + noise
    return x


class SynthesisLayer(M.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights?
    ):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.use_noise = False
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.resample_filter = mge.Tensor(upfirdn2d_setup_filter(resample_filter))
        self.padding = kernel_size // 2

        def_gain = 1.0
        if activation in ['relu', 'lrelu', 'swish']:  # 除了这些激活函数的def_gain = np.sqrt(2)，其余激活函数的def_gain = 1.0
            def_gain = np.sqrt(2)
        self.act_gain = def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = mge.Parameter(mge.tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32)))
        if use_noise:
            self.noise_const = mge.Tensor(np.random.randn(resolution, resolution).astype(np.float32))
            self.noise_strength = mge.Parameter(mge.tensor(np.zeros([1, ]).astype(np.float32)))
        self.bias = mge.Parameter(mge.tensor(np.zeros([out_channels, ]).astype(np.float32)))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act(x, self.bias.astype(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


class ToRGBLayer(M.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        # 假设屎山的channels_last都是False
        assert channels_last == False
        self.weight = mge.Parameter(mge.tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32)))
        self.bias = mge.Parameter(mge.tensor(np.zeros([out_channels, ]).astype(np.float32)))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.weight_gain = self.weight_gain.astype(np.float32)


    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act(x, self.bias.astype(x.dtype), clamp=self.conv_clamp)
        return x


class SynthesisBlock(M.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        out_channels,                       # Number of output channels.
        w_dim,                              # Intermediate latent (W) dimensionality.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of output color channels.
        is_last,                            # Is this the last block?
        architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        **layer_kwargs,                     # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.resample_filter = mge.Tensor(upfirdn2d_setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = mge.Parameter(mge.tensor(np.random.randn(out_channels, resolution, resolution).astype(np.float32)))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, **layer_kwargs):
        w_i = 0
        dtype = np.float16 if self.use_fp16 and not force_fp32 else np.float32
        if fused_modconv is None:
            fused_modconv = (not self.training) and (dtype == np.float32 or int(x.shape[0]) == 1)

        # Input.
        if self.in_channels == 0:
            x = self.const.astype(dtype=dtype)
            x = F.expand_dims(x, 0)
            x = F.tile(x, [ws.shape[0], 1, 1, 1])
        else:
            x = x.astype(dtype=dtype)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, ws[:, w_i, :], fused_modconv=fused_modconv, **layer_kwargs); w_i += 1
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, ws[:, w_i, :], fused_modconv=fused_modconv, **layer_kwargs); w_i += 1
            x = self.conv1(x, ws[:, w_i, :], fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs); w_i += 1
            x = y.add_(x)
        else:
            x = self.conv0(x, ws[:, w_i, :], fused_modconv=fused_modconv, **layer_kwargs); w_i += 1
            x = self.conv1(x, ws[:, w_i, :], fused_modconv=fused_modconv, **layer_kwargs); w_i += 1

        # ToRGB.
        if img is not None:
            img = upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, ws[:, w_i, :], fused_modconv=fused_modconv); w_i += 1
            y = y.astype(dtype=np.float32)
            img = img + y if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == np.float32
        return x, img



class StyleGANv2ADA_SynthesisNetwork(M.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0  # 分辨率是2的n次方
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            use_fp16 = False
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, **block_kwargs):
        block_ws = []
        ws = ws.astype(np.float32)
        w_idx = 0
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            # block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
            block_ws.append(ws[:, w_idx:w_idx + block.num_conv + block.num_torgb, :])
            w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img




