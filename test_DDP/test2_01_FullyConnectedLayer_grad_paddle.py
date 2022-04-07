
import paddle
import paddle.nn.functional as F
import numpy as np

class FullyConnectedLayer(paddle.nn.Layer):
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
            # out = paddle.addmm(b.unsqueeze(0), x, w.t())   # 因为paddle.addmm()没有实现二阶梯度，所以用其它等价实现。
            out = paddle.matmul(x, w, transpose_y=True) + b.unsqueeze(0)
        else:
            r = x.matmul(w.t())
            r += b.unsqueeze(0)
            out = F.sigmoid(r)
        return out


batch_size = 16
in_channels = 512
w_dim = 512
lr = 0.1
# activation = 'linear'
# activation = 'lrelu'
# activation = 'relu'
# activation = 'tanh'
activation = 'sigmoid'
# activation = 'elu'
# activation = 'selu'
# activation = 'softplus'
# activation = 'swish'

model = FullyConnectedLayer(w_dim, in_channels, activation=activation, bias_init=1)
model.train()
optimizer = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=lr, momentum=0.9)
model.set_state_dict(paddle.load("01_00.pdparams"))


dic2 = np.load('01.npz')
for batch_idx in range(20):
    print('======================== batch_%.3d ========================'%batch_idx)
    optimizer.clear_gradients()
    ws = dic2['batch_%.3d.input'%batch_idx]
    styles_pytorch = dic2['batch_%.3d.output'%batch_idx]
    dstyles2_dws_pytorch = dic2['batch_%.3d.dstyles2_dws'%batch_idx]
    ws = paddle.to_tensor(ws)
    ws.stop_gradient = False

    styles = model(ws)
    styles2 = F.sigmoid(styles)
    dstyles2_dws = paddle.grad(outputs=[styles2.sum()], inputs=[ws], create_graph=True)[0]


    styles_paddle = styles.numpy()
    ddd = np.sum((styles_pytorch - styles_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    dstyles2_dws_paddle = dstyles2_dws.numpy()
    ddd = np.sum((dstyles2_dws_pytorch - dstyles2_dws_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    loss = dstyles2_dws.sum() + styles2.sum()
    loss.backward()
    optimizer.step()
paddle.save(model.state_dict(), "01_19_DDP.pdparams")
print()
