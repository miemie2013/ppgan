
import paddle
import numpy as np
from ppgan.models.generators.generator_styleganv2ada import Conv2dLayer


in_channels = 256
out_channels = 256
kernel_size = 3
bias = True
activation = 'lrelu'
up = 1
down = 1
resample_filter = [1, 3, 3, 1]
conv_clamp = 256
channels_last = False
trainable = True




x_shape = [2, in_channels, 64, 64]
lr = 0.0001

model = Conv2dLayer(in_channels, out_channels, kernel_size, bias, activation, up, down, resample_filter, conv_clamp, channels_last, trainable)
model.train()
optimizer = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=lr, momentum=0.9)
model.set_state_dict(paddle.load("11.pdparams"))


dic2 = np.load('11.npz')
for batch_idx in range(8):
    print('======================== batch_%.3d ========================'%batch_idx)
    optimizer.clear_gradients()
    y_pytorch = dic2['batch_%.3d.output'%batch_idx]
    dy_dx_pytorch = dic2['batch_%.3d.dy_dx'%batch_idx]
    x = dic2['batch_%.3d.input0'%batch_idx]
    x = paddle.to_tensor(x)
    x.stop_gradient = False

    y = model(x)

    # dy_dx = paddle.grad(outputs=[y.sum()], inputs=[x], create_graph=True)[0]
    dysum_dy = paddle.ones(y.shape, dtype=paddle.float32)
    dy_dx = model.grad_layer(dysum_dy)


    y_paddle = y.numpy()
    ddd = np.sum((y_pytorch - y_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    dy_dx_paddle = dy_dx.numpy()
    ddd = np.sum((dy_dx_pytorch - dy_dx_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    ddd = np.mean((y_pytorch - y_paddle) ** 2)
    print('ddd=%.6f' % ddd)
    ddd = np.mean((dy_dx_pytorch - dy_dx_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    loss = dy_dx.sum() + y.sum()
    # loss = y.sum()
    loss.backward()
    optimizer.step()
print()
