import math
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P





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
        b_ = ops.Reshape()(b, new_shape)
        x = x + b_
    x_add_b = x

    # 经过激活函数
    alpha = float(alpha)  # 只有leaky_relu需要
    if act == 'linear':
        pass
    elif act == 'relu':
        x = P.ReLU()(x)
    elif act == 'lrelu':
        x = nn.LeakyReLU(alpha)(x)
    elif act == 'tanh':
        x = P.Tanh()(x)
    elif act == 'sigmoid':
        x = P.Sigmoid()(x)
    # elif act == 'elu':
    #     x = F.elu(x)
    # elif act == 'selu':
    #     x = F.selu(x)
    elif act == 'softplus':
        x = P.Softplus()(x)
    elif act == 'swish':
        temp = P.Sigmoid()(x)
        x = P.Mul()(temp, x)
    else:
        raise NotImplementedError("activation \'{}\' is not implemented.".format(act))
    act_x = x

    # 乘以缩放因子
    gain = float(gain)
    if gain != 1:
        x = x * gain
    gain_x = x

    # 限制范围
    if clamp >= 0:
        x = F.clip(x, -clamp, clamp)
    clamp_x = x
    return clamp_x




class FullyConnectedLayer(nn.Cell):
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
        # weight = ms.Parameter(ms.Tensor([out_features, in_features], ms.float32))
        # self.weight = self.insert_param_to_cell("weight", weight)
        # bias = ms.Parameter(ms.Tensor([out_features, ], ms.float32))
        # self.bias = self.insert_param_to_cell("bias", bias)
        self.weight = Parameter(initializer('normal', [out_features, in_features]), name='weight')
        self.bias = Parameter(initializer('zeros', [out_features]), name='bias')
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.weight_gain = float(self.weight_gain)
        self.bias_gain = lr_multiplier
        # self.layer = nn.Conv2d()

    def construct(self, x):
        # w = self.weight.to(x.dtype) * self.weight_gain
        w = self.weight * self.weight_gain
        b = self.bias
        if b is not None:
            # b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            w_t = P.Transpose()(w, (1, 0))
            out = P.MatMul()(x, w_t)
            _b = P.ExpandDims()(b, 0)
            out = P.Add()(out, _b)
        else:
            w_t = P.Transpose()(w, (1, 0))
            r = P.MatMul()(x, w_t)
            out = bias_act(r, b, act=self.activation)
        return out

