# code was heavily based on https://github.com/clovaai/stargan-v2
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/clovaai/stargan-v2#license
import paddle
from paddle import nn
import paddle.nn.functional as F

from .builder import GENERATORS
import numpy as np
import math




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
        b_ = paddle.reshape(b, (1, -1, 1, 1))
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
        x = x.clamp(-clamp, clamp) # pylint: disable=invalid-unary-operand-type
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
    fw, fh = filter.shape
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
            w = w.transpose(0, 1)
        else:
            w = w.reshape(groups, out_channels // groups, in_channels_per_group, kh, kw)
            w = w.transpose(1, 2)
            w = w.reshape(groups * in_channels_per_group, out_channels // groups, kh, kw)
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
        self.bn = nn.InstanceNorm2D(out_channels)
        self.activation = nn.LeakyReLU()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = x.transpose((0, 2, 3, 1))
        out = self.linear(x)
        out = out.transpose((0, 3, 1, 2))
        out = self.bn(out)
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
                                            default_initializer=paddle.nn.initializer.Normal()) / lr_multiplier
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









