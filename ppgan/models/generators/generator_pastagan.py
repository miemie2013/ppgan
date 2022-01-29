# code was heavily based on https://github.com/clovaai/stargan-v2
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/clovaai/stargan-v2#license
import paddle
from paddle import nn
import paddle.nn.functional as F

from .builder import GENERATORS
import numpy as np
import math



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
    shape = paddle.to_tensor(shape, dtype='float32')
    assert shape.ndim in [0, 1, 2]
    assert shape.numel() > 0
    if shape.ndim == 0:
        shape = shape[np.newaxis]

    # Separable?
    if separable is None:
        separable = (shape.ndim == 1 and shape.numel() >= 8)
    if shape.ndim == 1 and not separable:
        # ger()相当于向量自乘
        shape = paddle.unsqueeze(shape, 1)  # [n, 1]
        shape = paddle.matmul(shape, shape.transpose((1, 0)))  # [n, n]
    assert shape.ndim == (1 if separable else 2)

    # Apply normalize, flip, gain, and device.
    if normalize:
        shape /= shape.sum()
    if flip_filter:
        shape = shape.flip(list(range(shape.ndim)))
    shape = shape * (gain ** (shape.ndim / 2))
    return shape


def bias_act(x, b=None, dim=1, act='linear', alpha=None, gain=None, clamp=None):
    assert clamp is None or clamp >= 0
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
        b_ = paddle.reshape(b, new_shape)
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
        x = paddle.tanh(x)
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
        x = paddle.clip(x, -clamp, clamp)
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
        w = w.flip([2, 3])

    # Workaround performance pitfall in cuDNN 8.0.5, triggered when using
    # 1x1 kernel + memory_format=channels_last + less than 64 channels.
    if kw == 1 and kh == 1 and stride == 1 and padding in [0, [0, 0], (0, 0)] and not transpose:
        if x.shape[2] * x.shape[3] == 1 and min(out_channels, in_channels_per_group) < 64:
            if out_channels <= 4 and groups == 1:
                in_shape = x.shape
                x = w.squeeze(3).squeeze(2) @ x.reshape([in_shape[0], in_channels_per_group, -1])
                x = x.reshape([in_shape[0], out_channels, in_shape[2], in_shape[3]])
            else:
                # x = x.to(memory_format=torch.contiguous_format)
                # w = w.to(memory_format=torch.contiguous_format)
                x = F.conv2d(x, w, groups=groups)
            # return x.to(memory_format=torch.channels_last)
            return x

    # Otherwise => execute using conv2d_gradfix.
    if transpose:
        return F.conv2d_transpose(x, weight=w, bias=None, stride=stride, padding=padding, output_padding=0, groups=groups, dilation=1)
    else:
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

def upfirdn2d(x, filter, up=1, down=1, padding=0, flip_filter=False, gain=1):
    if filter is None:
        filter = paddle.ones([1, 1], dtype=paddle.float32)
    batch_size, num_channels, in_height, in_width = x.shape
    upx, upy = _parse_scaling(up)        # scaling 一变二
    downx, downy = _parse_scaling(down)  # scaling 一变二
    padx0, padx1, pady0, pady1 = _parse_padding(padding)

    # Upsample by inserting zeros.
    # paddle最多支持5维张量，所以分开2次pad。
    # 根据data_format指定的意义填充(pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width])
    x = paddle.nn.functional.pad(x, [0, 0, 0, upy - 1, 0, 0], data_format="NCDHW")
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width, 1])
    x = paddle.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, 0], data_format="NCDHW")
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])

    # Pad or crop.
    x = F.pad(x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)])
    x = x[:, :, max(-pady0, 0) : x.shape[2] - max(-pady1, 0), max(-padx0, 0) : x.shape[3] - max(-padx1, 0)]

    # Setup filter.
    filter = filter * (gain ** (filter.ndim / 2))
    filter = paddle.cast(filter, dtype=x.dtype)
    if not flip_filter:
        filter = filter.flip(list(range(filter.ndim)))

    # Convolve with the filter.
    filter = paddle.unsqueeze(filter, [0, 1]).tile([num_channels, 1] + [1] * filter.ndim)
    if filter.ndim == 4:
        x = F.conv2d(x, weight=filter, groups=num_channels)
    else:
        x = F.conv2d(x, weight=filter.unsqueeze(2), groups=num_channels)
        x = F.conv2d(x, weight=filter.unsqueeze(3), groups=num_channels)

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
            w = w.transpose((1, 0, 2, 3))
        else:
            w = w.reshape((groups, out_channels // groups, in_channels_per_group, kh, kw))
            w = w.transpose((0, 2, 1, 3, 4))
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


class Conv2dLayer(nn.Layer):
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
        self.register_buffer('resample_filter', upfirdn2d_setup_filter(resample_filter))
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

        if trainable:
            self.weight = self.create_parameter([out_channels, in_channels, kernel_size, kernel_size],
                                                default_initializer=paddle.nn.initializer.Normal())
            self.bias = self.create_parameter([out_channels], is_bias=True,
                                              default_initializer=paddle.nn.initializer.Constant(0.0)) if bias else None
        else:
            self.weight = self.create_parameter([out_channels, in_channels, kernel_size, kernel_size],
                                                default_initializer=paddle.nn.initializer.Constant(1.0))
            self.weight.stop_gradient = True
            self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = paddle.cast(self.bias, dtype=x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1)  # slightly faster
        x = conv2d_resample(x=x, w=paddle.cast(w, dtype=x.dtype), filter=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


@GENERATORS.register()
class ConstEncoderNetwork(nn.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4):
        super().__init__()

        encoder = []
        encoder += [Conv2dLayer(input_nc, ngf, kernel_size=1)]
        mult_ins = [1, 2, 4, 4, 4, 8]
        mult_outs = [2, 4, 4, 4, 8, 8]
        for i in range(n_downsampling):
            mult_in = mult_ins[i]
            mult_out = mult_outs[i]
            encoder += [Conv2dLayer(ngf * mult_in, ngf * mult_out, kernel_size=3, down=2)]

        self.model = nn.Sequential(*encoder)

    def forward(self, x):
        x = self.model(x)
        return x


class Dense(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # pytorch的nn.InstanceNorm2d()默认是不使用scale和offset的，只做归一化。所以这里设为False。
        self.in_ = nn.InstanceNorm2D(out_channels, weight_attr=False, bias_attr=False)
        self.activation = nn.LeakyReLU()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = x.transpose((0, 2, 3, 1))
        out = self.linear(x)
        out = out.transpose((0, 3, 1, 2))
        out = self.in_(out)
        out = self.activation(out)
        return out


class FullyConnectedLayer(nn.Layer):
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
        self.weight = self.create_parameter([out_features, in_features],
                                            default_initializer=paddle.nn.initializer.Normal(mean=0.0, std=1.0 / lr_multiplier))
        self.bias = self.create_parameter([out_features], is_bias=True,
                                          default_initializer=paddle.nn.initializer.Constant(bias_init)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = paddle.cast(self.weight, dtype=x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = paddle.cast(b, dtype=x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = paddle.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act(x, b, act=self.activation)
        return x





@GENERATORS.register()
class StyleEncoderNetwork(nn.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4):
        super().__init__()
        encoder = []
        encoder += [Conv2dLayer(input_nc, ngf, kernel_size=1)]
        mult_ins = [1, 2, 4]
        mult_outs = [2, 4, 8]
        for i in range(3):
            mult_in = mult_ins[i]
            mult_out = mult_outs[i]
            encoder += [Dense(ngf * mult_in, ngf * mult_in),
                        Conv2dLayer(ngf * mult_in, ngf * mult_out, kernel_size=3, down=2)]
        mult_ins = [8, 8, 8]
        mult_outs = [8, 8, 8]
        for i in range(3):
            mult_in = mult_ins[i]
            mult_out = mult_outs[i]
            encoder += [Dense(ngf * mult_in, ngf * mult_in),
                        Conv2dLayer(ngf * mult_in, ngf * mult_out, kernel_size=3)]

        encoder += [nn.AdaptiveAvgPool2D(1)]
        self.model = nn.Sequential(*encoder)
        self.fc = FullyConnectedLayer(output_nc, output_nc)

        feat_enc = []
        feat_enc += [Conv2dLayer(3, ngf, kernel_size=3)]
        mult_ins = [1, 1, 1]
        mult_outs = [1, 1, 1]
        for i in range(3):
            mult_in = mult_ins[i]
            mult_out = mult_outs[i]
            feat_enc += [Conv2dLayer(ngf * mult_in, ngf * mult_out, kernel_size=3, down=2)]

        self.feat_enc = nn.Sequential(*feat_enc)

    def forward(self, x, const_input):
        const_feats = []
        for _, module in enumerate(self.feat_enc):
            const_input = module(const_input)
            const_feats.append(const_input)

        for _, module in enumerate(self.model):
            x = module(x)
        x = x.reshape((x.shape[0], -1))
        x = self.fc(x)

        return x, const_feats


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(axis=dim, keepdim=True) + eps).rsqrt()


@GENERATORS.register()
class MappingNetwork(nn.Layer):
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
            self.register_buffer('w_avg', paddle.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        if self.z_dim > 0:
            temp1 = paddle.cast(z, dtype='float32')
            x = normalize_2nd_moment(temp1)
        if self.c_dim > 0:
            temp2 = paddle.cast(c, dtype='float32')
            y = normalize_2nd_moment(self.embed(temp2))
            x = paddle.concat([x, y], 1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            temp3 = x.detach().mean(axis=0)
            # temp3 = temp3.lerp(self.w_avg, self.w_avg_beta)
            temp3 = temp3 + self.w_avg_beta * (self.w_avg - temp3)
            # self.w_avg.copy_(temp3)
            self.w_avg = temp3

        # Broadcast.
        if self.num_ws is not None:
            x = x.unsqueeze(1).tile([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            assert self.w_avg_beta is not None
            if self.num_ws is None or truncation_cutoff is None:
                # x = self.w_avg.lerp(x, truncation_psi)
                x = self.w_avg + truncation_psi * (x - self.w_avg)
            else:
                # x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
                x[:, :truncation_cutoff] = self.w_avg + truncation_psi * (x[:, :truncation_cutoff] - self.w_avg)
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
    # misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    # misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    # misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == paddle.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0)  # [NOIkk]
        w = w * styles.reshape((batch_size, 1, -1, 1, 1))  # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(axis=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape((batch_size, -1, 1, 1, 1))  # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * paddle.cast(styles, dtype=x.dtype).reshape((batch_size, -1, 1, 1))
        x = conv2d_resample(x=x, w=paddle.cast(weight, dtype=x.dtype), filter=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = x * paddle.cast(dcoefs, dtype=x.dtype).reshape((batch_size, -1, 1, 1)) + paddle.cast(noise, dtype=x.dtype)
        elif demodulate:
            x = x * paddle.cast(dcoefs, dtype=x.dtype).reshape((batch_size, -1, 1, 1))
        elif noise is not None:
            x = x + paddle.cast(noise, dtype=x.dtype)
        return x

    # Execute as one fused op using grouped convolution.
    x = x.reshape((1, -1, *x.shape[2:]))
    w = w.reshape((-1, in_channels, kh, kw))
    x = conv2d_resample(x=x, w=paddle.cast(w, dtype=x.dtype), filter=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape((batch_size, -1, *x.shape[2:]))
    if noise is not None:
        x = x + noise
    return x


class Spade_Conv2dLayer(nn.Layer):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 out_channels,  # Number of output channels.
                 kernel_size,  # Width and height of the convolution kernel.
                 bias=True,  # Apply additive bias before the activation function?
                 activation='relu',  # Activation function: 'relu', 'lrelu', etc.
                 up=1,  # Integer upsampling factor.
                 down=1,  # Integer downsampling factor.
                 resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
                 conv_clamp=None,  # Clamp the output to +-X, None = disable clamping.
                 channels_last=False,  # Expect the input to have memory_format=channels_last?
                 trainable=True,  # Update the weights of this layer during training?
                 ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d_setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

        def_gain = 1.0
        if activation in ['relu', 'lrelu', 'swish']:  # 除了这些激活函数的def_gain = np.sqrt(2)，其余激活函数的def_gain = 1.0
            def_gain = np.sqrt(2)
        self.act_gain = def_gain

        # 假设屎山的channels_last都是False
        assert channels_last == False
        # memory_format = torch.channels_last if channels_last else torch.contiguous_format
        # weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        self.bias = None
        if trainable:
            self.weight = self.create_parameter([out_channels, in_channels, kernel_size, kernel_size],
                                                default_initializer=paddle.nn.initializer.Normal())
            if bias is not None:
                if bias != False:
                    self.bias = self.create_parameter([out_channels, ],
                                                      default_initializer=paddle.nn.initializer.Constant(0.0))
        else:
            self.weight = self.create_parameter([out_channels, in_channels, kernel_size, kernel_size],
                                                default_initializer=paddle.nn.initializer.Normal())
            self.weight.stop_gradient = True
            if bias is not None:
                if bias != False:
                    self.bias = self.create_parameter([out_channels, ],
                                                      default_initializer=paddle.nn.initializer.Constant(0.0))
                    self.bias.stop_gradient = True

    def forward(self, x, gain=1, no_act=False):
        w = self.weight * self.weight_gain
        b = paddle.cast(self.bias, dtype=x.dtype) if self.bias is not None else None

        if not no_act:
            act_gain = self.act_gain * gain
            act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
            x = bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)

        flip_weight = (self.up == 1)  # slightly faster
        x = conv2d_resample(x=x, w=paddle.cast(w, dtype=x.dtype), filter=self.resample_filter, up=self.up, down=self.down,
                            padding=self.padding, flip_weight=flip_weight)

        return x

class Spade_Norm_Block(nn.Layer):
    def __init__(self,
        in_channels,
        norm_channels,
    ):
        super().__init__()
        self.conv_mlp = Spade_Conv2dLayer(in_channels, norm_channels, kernel_size=3, bias=False)
        self.conv_mlp_act = nn.ReLU()
        self.conv_gamma = Spade_Conv2dLayer(norm_channels, norm_channels, kernel_size=3, bias=False)
        self.conv_beta = Spade_Conv2dLayer(norm_channels, norm_channels, kernel_size=3, bias=False)

        self.param_free_norm = nn.InstanceNorm2D(norm_channels, weight_attr=False, bias_attr=False)

    def forward(self, x, denorm_feats):
        normalized = self.param_free_norm(x)
        actv = self.conv_mlp(denorm_feats, no_act=True)
        actv = self.conv_mlp_act(actv)
        gamma = self.conv_gamma(actv, no_act=True)
        beta = self.conv_beta(actv, no_act=True)

        out = normalized * (1+gamma) + beta
        return out


class Spade_ResBlock(nn.Layer):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 out_channels,  # Number of output channels.
                 kernel_size=3,  # Width and height of the convolution kernel.
                 bias=True,  # Apply additive bias before the activation function?
                 activation='linear',  # Activation function: 'relu', 'lrelu', etc.
                 up=1,  # Integer upsampling factor.
                 down=1,  # Integer downsampling factor.
                 resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
                 conv_clamp=None,  # Clamp the output to +-X, None = disable clamping.
                 channels_last=False,  # Expect the input to have memory_format=channels_last?
                 trainable=True,  # Update the weights of this layer during training?
                 resolution=128,
                 ):
        super().__init__()
        self.register_buffer('resample_filter', upfirdn2d_setup_filter(resample_filter))

        self.conv = Spade_Conv2dLayer(in_channels, in_channels, kernel_size=3, bias=False,
                                      resample_filter=resample_filter, conv_clamp=conv_clamp,
                                      channels_last=channels_last)
        self.conv0 = Spade_Conv2dLayer(in_channels, out_channels, kernel_size=3, bias=False,
                                       resample_filter=resample_filter, conv_clamp=conv_clamp,
                                       channels_last=channels_last)
        self.conv1 = Spade_Conv2dLayer(out_channels, out_channels, kernel_size=3, bias=False,
                                       resample_filter=resample_filter, conv_clamp=conv_clamp,
                                       channels_last=channels_last)
        self.skip = Spade_Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False,
                                      resample_filter=resample_filter, conv_clamp=conv_clamp,
                                      channels_last=channels_last)

        if resolution == 128:
            feat_channels = 128 * 2
        else:
            feat_channels = 64 * 2
        self.spade_skip = Spade_Norm_Block(feat_channels, in_channels)
        self.spade0 = Spade_Norm_Block(feat_channels, in_channels)
        self.spade1 = Spade_Norm_Block(feat_channels, out_channels)

    def forward(self, x, denorm_feat):
        x = self.conv(x, no_act=True)

        y = self.skip(self.spade_skip(x, denorm_feat), gain=np.sqrt(0.5))
        x = self.conv0(self.spade0(x, denorm_feat))
        x = self.conv1(self.spade1(x, denorm_feat), gain=np.sqrt(0.5))

        x = y.add_(x)
        return x

class SynthesisLayer(nn.Layer):
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
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d_setup_filter(resample_filter))
        self.padding = kernel_size // 2

        def_gain = 1.0
        if activation in ['relu', 'lrelu', 'swish']:  # 除了这些激活函数的def_gain = np.sqrt(2)，其余激活函数的def_gain = 1.0
            def_gain = np.sqrt(2)
        self.act_gain = def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        # 假设屎山的channels_last都是False
        assert channels_last == False
        # memory_format = torch.channels_last if channels_last else torch.contiguous_format
        # self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.weight = self.create_parameter([out_channels, in_channels, kernel_size, kernel_size],
                                            default_initializer=paddle.nn.initializer.Normal())

        if use_noise:
            self.register_buffer('noise_const', paddle.randn([resolution, resolution]))
            # self.noise_strength = torch.nn.Parameter(torch.zeros([]))
            # 噪声强度（振幅）
            self.noise_strength = self.create_parameter([1, ],
                                                        default_initializer=paddle.nn.initializer.Constant(0.0))
        # self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.bias = self.create_parameter([out_channels, ],
                                          default_initializer=paddle.nn.initializer.Constant(0.0))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = paddle.randn([x.shape[0], 1, self.resolution, self.resolution]) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act(x, paddle.cast(self.bias, dtype=x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


class ToRGBLayerV18(nn.Layer):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False, is_last=False):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        # 假设屎山的channels_last都是False
        assert channels_last == False
        # memory_format = torch.channels_last if channels_last else torch.contiguous_format
        # self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.weight = self.create_parameter([out_channels, in_channels, kernel_size, kernel_size],
                                            default_initializer=paddle.nn.initializer.Normal())
        self.bias = self.create_parameter([out_channels, ],
                                          default_initializer=paddle.nn.initializer.Constant(0.0))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.is_last = is_last

        if self.is_last:
            # self.m_weight1 = torch.nn.Parameter(torch.randn([1, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
            # self.m_bias1 = torch.nn.Parameter(torch.zeros([1]))
            self.m_weight1 = self.create_parameter([1, in_channels, kernel_size, kernel_size],
                                                   default_initializer=paddle.nn.initializer.Normal())
            self.m_bias1 = self.create_parameter([1, ],
                                                 default_initializer=paddle.nn.initializer.Constant(0.0))

            # self.m_weight2 = torch.nn.Parameter(torch.randn([1, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
            # self.m_bias2 = torch.nn.Parameter(torch.zeros([1]))
            self.m_weight2 = self.create_parameter([1, in_channels, kernel_size, kernel_size],
                                                   default_initializer=paddle.nn.initializer.Normal())
            self.m_bias2 = self.create_parameter([1, ],
                                                 default_initializer=paddle.nn.initializer.Constant(0.0))


    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain

        upper_mask = None
        lower_mask = None
        if self.is_last:
            upper_mask = modulated_conv2d(x=x, weight=self.m_weight1, styles=styles, demodulate=False, fused_modconv=fused_modconv)
            upper_mask = bias_act(upper_mask, paddle.cast(self.m_bias1, dtype=x.dtype), clamp=self.conv_clamp, act='sigmoid')

            lower_mask = modulated_conv2d(x=x, weight=self.m_weight2, styles=styles, demodulate=False, fused_modconv=fused_modconv)
            lower_mask = bias_act(lower_mask, paddle.cast(self.m_bias2, dtype=x.dtype), clamp=self.conv_clamp, act='sigmoid')

        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act(x, paddle.cast(self.bias, dtype=x.dtype), clamp=self.conv_clamp)
        return x, upper_mask, lower_mask



class SynthesisBlock(nn.Layer):
    def __init__(self,
                 in_channels,  # Number of input channels, 0 = first block.
                 out_channels,  # Number of output channels.
                 w_dim,  # Intermediate latent (W) dimensionality.
                 resolution,  # Resolution of this block.
                 img_channels,  # Number of output color channels.
                 is_last,  # Is this the last block?
                 architecture='skip',  # Architecture: 'orig', 'skip', 'resnet'.
                 resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
                 conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 use_fp16=False,  # Use FP16 for this block?
                 fp16_channels_last=False,  # Use channels-last memory format with FP16?
                 **layer_kwargs,  # Arguments for SynthesisLayer.
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
        self.register_buffer('resample_filter', upfirdn2d_setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        # CONST here
        if in_channels == 0:
            self.const = self.create_parameter([out_channels, resolution, resolution],
                                               default_initializer=paddle.nn.initializer.Normal())

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                                        resample_filter=resample_filter, conv_clamp=conv_clamp,
                                        channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
                                    conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayerV18(out_channels, img_channels, w_dim=w_dim,
                                       conv_clamp=conv_clamp, channels_last=self.channels_last, is_last=is_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                                    resample_filter=resample_filter, channels_last=self.channels_last)

        if self.resolution > 16:
            self.merge_conv = Conv2dLayer(out_channels + 64, out_channels, kernel_size=1,
                                          resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, ws, pose_feature, cat_feat, force_fp32=False, fused_modconv=None, **layer_kwargs):
        w_iter = iter(ws.unbind(axis=1))
        dtype = paddle.float16 if self.use_fp16 and not force_fp32 else paddle.float32
        # 假设屎山的channels_last都是False
        assert self.channels_last == False
        # memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            # with misc.suppress_tracer_warnings():  # this value will be treated as a constant
            fused_modconv = (not self.training) and (dtype == paddle.float32 or int(x.shape[0]) == 1)

        # Input.
        if self.in_channels == 0:
            # CONST here
            x = paddle.cast(pose_feature, dtype=dtype)
        else:
            x = paddle.cast(x, dtype=dtype)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

            # add warped feature here
            if x.shape[2] > 16:
                # paddle.cast(styles, dtype=x.dtype)
                # x = paddle.concat([x, cat_feat[str(x.shape[2])].to(dtype=dtype, memory_format=memory_format)], axis=1)
                x = paddle.concat([x, paddle.cast(cat_feat[str(x.shape[2])], dtype=dtype)], axis=1)
                x = self.merge_conv(x)

        # ToRGB.
        if img is not None:
            img = upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y, upper_mask, lower_mask = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            # y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            y = paddle.cast(y, dtype='float32')
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == paddle.float32
        return x, img, upper_mask, lower_mask


class ResBlock(nn.Layer):
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
        self.register_buffer('resample_filter', upfirdn2d_setup_filter(resample_filter))

        self.conv0 = Conv2dLayer(in_channels, out_channels, kernel_size=3, activation=activation, up=up, down=down, bias=bias,
                                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=channels_last)
        self.conv1 = Conv2dLayer(out_channels, out_channels, kernel_size=3, activation=activation, bias=bias, resample_filter=resample_filter,
                                conv_clamp=conv_clamp, channels_last=channels_last)
        self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=up, down=down, resample_filter=resample_filter,
                                conv_clamp=conv_clamp, channels_last=channels_last)

    def forward(self, x):
        y = self.skip(x, gain=np.sqrt(0.5))
        x = self.conv0(x)
        x = self.conv1(x, gain=np.sqrt(0.5))
        x = y + x
        return x



@GENERATORS.register()
class SynthesisNetwork(nn.Layer):
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
            # use_fp16 = (res >= fp16_resolution)
            use_fp16 = False
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

        res = self.block_resolutions[-2]
        in_channels = channels_dict[res]
        out_channels = channels_dict[res]
        self.spade_b128_1 = Spade_ResBlock(in_channels, out_channels)
        self.spade_b128_2 = Spade_ResBlock(in_channels, out_channels)
        self.spade_b128_3 = Spade_ResBlock(in_channels, out_channels)

        res = self.block_resolutions[-1]
        in_channels = channels_dict[res//2]
        out_channels = channels_dict[res]
        self.texture_b256 = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=True, use_fp16=False, **block_kwargs)

        spade_encoder = []
        ngf = 64
        spade_encoder += [Conv2dLayer(3, ngf, kernel_size=7, activation='relu')]
        spade_encoder += [ResBlock(ngf, ngf, kernel_size=4, activation='relu')]                 # 256
        spade_encoder += [ResBlock(ngf, ngf*2, kernel_size=4, activation='relu', down=2)]       # 128
        self.spade_encoder = nn.Sequential(*spade_encoder)

    def get_spade_feat(self, mask_256, denorm_mask, denorm_input):
        mask_256 = paddle.cast(mask_256 > 0.9, dtype=mask_256.dtype)
        mask_128 = F.interpolate(mask_256, scale_factor=0.5)
        denorm_mask_128 = F.interpolate(denorm_mask, scale_factor=0.5)
        mask_128 = paddle.cast(mask_128 > 0.9, dtype=mask_256.dtype)
        denorm_mask_128 = paddle.cast(denorm_mask_128 > 0.9, dtype=mask_256.dtype)

        valid_mask = paddle.cast((mask_128 + denorm_mask_128) == 2.0, dtype=mask_256.dtype)
        res_mask = paddle.cast((mask_128 - valid_mask), dtype=mask_256.dtype)

        denorm_input = denorm_input * mask_256 - (1-mask_256)
        spade_denorm_feat = self.spade_encoder(denorm_input)
        spade_denorm_valid_feat = spade_denorm_feat * valid_mask

        valid_feat_sum = paddle.sum(spade_denorm_valid_feat, axis=[2, 3], keepdim=True)
        valid_mask_sum = paddle.sum(valid_mask, axis=[2, 3], keepdim=True)

        valid_index = paddle.cast(valid_mask_sum > 10, dtype=mask_256.dtype)
        valid_mask_sum = valid_mask_sum * valid_index + (128*128) * (1-valid_index)
        spade_average_feat = valid_feat_sum / valid_mask_sum

        spade_feat = spade_denorm_feat * (1-res_mask) + spade_average_feat * res_mask

        return spade_feat


    def forward(self, ws, pose_feat, cat_feat, denorm_upper_input, denorm_lower_input, denorm_upper_mask, \
                denorm_lower_mask, **block_kwargs):
        block_ws = []
        ws = paddle.cast(ws, dtype='float32')
        w_idx = 0
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            # block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
            block_ws.append(ws[:, w_idx:w_idx + block.num_conv + block.num_torgb, :])
            w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img, upper_mask, lower_mask = block(x, img, cur_ws, pose_feat, cat_feat, force_fp32=True, **block_kwargs)
            if res == 128:
                x_128, img_128 = x.clone(), img.clone()

        spade_upper_feat = self.get_spade_feat(upper_mask.detach(), denorm_upper_mask, denorm_upper_input)
        spade_lower_feat = self.get_spade_feat(lower_mask.detach(), denorm_lower_mask, denorm_lower_input)

        spade_feat = paddle.concat([spade_upper_feat, spade_lower_feat], 1)

        x_spade_128 = self.spade_b128_1(x_128, spade_feat)
        x_spade_128 = self.spade_b128_2(x_spade_128, spade_feat)
        x_spade_128 = self.spade_b128_3(x_spade_128, spade_feat)

        cur_ws = block_ws[-1]
        finetune_x, finetune_img, _, _ = self.texture_b256(x_spade_128, img_128, cur_ws, pose_feat, cat_feat, force_fp32=True, **block_kwargs)

        return img, finetune_img, upper_mask, lower_mask


