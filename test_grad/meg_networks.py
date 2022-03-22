import math
import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.module as M





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
    x_add_b = x

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
        # w = self.weight.to(x.dtype) * self.weight_gain
        w = self.weight * self.weight_gain
        b = self.bias
        if b is not None:
            # b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            out = F.matmul(x, w, transpose_b=True) + F.expand_dims(b, 0)
        else:
            # r = x.matmul(w.t())
            r = F.matmul(x, w, transpose_b=True)
            out = bias_act(r, b, act=self.activation)
        return out

