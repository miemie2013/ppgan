
import paddle
import numpy as np
from ppgan.models.generators.generator_styleganv2ada import SynthesisLayer


x_shape = [1, 512, 4, 4]
w_shape = [1, 512]
in_channels = 512
out_channels = 512
w_dim = 512
resolution = 4
kernel_size = 3
up = 1
use_noise = True
activation = 'lrelu'
resample_filter = [1, 3, 3, 1]
conv_clamp = 256
channels_last = False
fused_modconv = True
gain = 1

x_shape = [1, 512, 4, 4]
w_shape = [1, 512]
in_channels = 512
out_channels = 512
w_dim = 512
resolution = 8
kernel_size = 3
up = 2
use_noise = True
activation = 'lrelu'
resample_filter = [1, 3, 3, 1]
conv_clamp = 256
channels_last = False
fused_modconv = True
gain = 1

x_shape = [1, 512, 8, 8]
w_shape = [1, 512]
in_channels = 512
out_channels = 512
w_dim = 512
resolution = 8
kernel_size = 3
up = 1
use_noise = True
activation = 'lrelu'
resample_filter = [1, 3, 3, 1]
conv_clamp = 256
channels_last = False
fused_modconv = True
gain = 1

x_shape = [1, 512, 8, 8]
w_shape = [1, 512]
in_channels = 512
out_channels = 512
w_dim = 512
resolution = 16
kernel_size = 3
up = 2
use_noise = True
activation = 'lrelu'
resample_filter = [1, 3, 3, 1]
conv_clamp = 256
channels_last = False
fused_modconv = True
gain = 1

x_shape = [1, 512, 16, 16]
w_shape = [1, 512]
in_channels = 512
out_channels = 512
w_dim = 512
resolution = 16
kernel_size = 3
up = 1
use_noise = True
activation = 'lrelu'
resample_filter = [1, 3, 3, 1]
conv_clamp = 256
channels_last = False
fused_modconv = True
gain = 1

x_shape = [1, 512, 16, 16]
w_shape = [1, 512]
in_channels = 512
out_channels = 512
w_dim = 512
resolution = 32
kernel_size = 3
up = 2
use_noise = True
activation = 'lrelu'
resample_filter = [1, 3, 3, 1]
conv_clamp = 256
channels_last = False
fused_modconv = True
gain = 1

x_shape = [1, 512, 32, 32]
w_shape = [1, 512]
in_channels = 512
out_channels = 512
w_dim = 512
resolution = 32
kernel_size = 3
up = 1
use_noise = True
activation = 'lrelu'
resample_filter = [1, 3, 3, 1]
conv_clamp = 256
channels_last = False
fused_modconv = True
gain = 1

x_shape = [1, 512, 32, 32]
w_shape = [1, 512]
in_channels = 512
out_channels = 512
w_dim = 512
resolution = 64
kernel_size = 3
up = 2
use_noise = True
activation = 'lrelu'
resample_filter = [1, 3, 3, 1]
conv_clamp = 256
channels_last = False
fused_modconv = True
gain = 1

x_shape = [1, 512, 64, 64]
w_shape = [1, 512]
in_channels = 512
out_channels = 512
w_dim = 512
resolution = 64
kernel_size = 3
up = 1
use_noise = True
activation = 'lrelu'
resample_filter = [1, 3, 3, 1]
conv_clamp = 256
channels_last = False
fused_modconv = True
gain = 1

x_shape = [1, 512, 64, 64]
w_shape = [1, 512]
in_channels = 512
out_channels = 256
w_dim = 512
resolution = 128
kernel_size = 3
up = 2
use_noise = True
activation = 'lrelu'
resample_filter = [1, 3, 3, 1]
conv_clamp = 256
channels_last = False
fused_modconv = True
gain = 1

x_shape = [1, 256, 128, 128]
w_shape = [1, 512]
in_channels = 256
out_channels = 256
w_dim = 512
resolution = 128
kernel_size = 3
up = 1
use_noise = True
activation = 'lrelu'
resample_filter = [1, 3, 3, 1]
conv_clamp = 256
channels_last = False
fused_modconv = True
gain = 1

x_shape = [1, 256, 128, 128]
w_shape = [1, 512]
in_channels = 256
out_channels = 128
w_dim = 512
resolution = 256
kernel_size = 3
up = 2
use_noise = True
activation = 'lrelu'
resample_filter = [1, 3, 3, 1]
conv_clamp = 256
channels_last = False
fused_modconv = True
gain = 1

x_shape = [1, 128, 256, 256]
w_shape = [1, 512]
in_channels = 128
out_channels = 128
w_dim = 512
resolution = 256
kernel_size = 3
up = 1
use_noise = True
activation = 'lrelu'
resample_filter = [1, 3, 3, 1]
conv_clamp = 256
channels_last = False
fused_modconv = True
gain = 1

x_shape = [1, 128, 256, 256]
w_shape = [1, 512]
in_channels = 128
out_channels = 64
w_dim = 512
resolution = 512
kernel_size = 3
up = 2
use_noise = True
activation = 'lrelu'
resample_filter = [1, 3, 3, 1]
conv_clamp = 256
channels_last = False
fused_modconv = True
gain = 1

# x_shape = [1, 64, 512, 512]
# w_shape = [1, 512]
# in_channels = 64
# out_channels = 64
# w_dim = 512
# resolution = 512
# kernel_size = 3
# up = 1
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1



batch_size = 2
lr = 0.0001

# 强制设置为不使用噪声
use_noise = False
model = SynthesisLayer(in_channels, out_channels, w_dim, resolution,
                       kernel_size, up, use_noise, activation, resample_filter, conv_clamp, channels_last)
model.train()
optimizer = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=lr, momentum=0.9)
model.set_state_dict(paddle.load("52.pdparams"))


dic2 = np.load('52.npz')
for batch_idx in range(8):
    print('======================== batch_%.3d ========================'%batch_idx)
    optimizer.clear_gradients()
    x = dic2['batch_%.3d.input0'%batch_idx]
    ws = dic2['batch_%.3d.input1'%batch_idx]
    y_pytorch = dic2['batch_%.3d.output'%batch_idx]
    dy_dws_pytorch = dic2['batch_%.3d.dy_dws'%batch_idx]
    dy_dx_pytorch = dic2['batch_%.3d.dy_dx'%batch_idx]
    ws = paddle.to_tensor(ws)
    ws.stop_gradient = False
    x = paddle.to_tensor(x)
    x.stop_gradient = False
    y = model(x, ws, noise_mode='random', fused_modconv=fused_modconv, gain=gain)

    # dy_dx = paddle.grad(outputs=[y.sum()], inputs=[x], create_graph=True)[0]
    # dy_dws = paddle.grad(outputs=[y.sum()], inputs=[ws], create_graph=True)[0]
    dysum_dy = paddle.ones(y.shape, dtype=paddle.float32)
    dy_dx, dy_dws = model.grad_layer(dysum_dy)

    y_paddle = y.numpy()
    ddd = np.sum((y_pytorch - y_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    dy_dx_paddle = dy_dx.numpy()
    ddd = np.sum((dy_dx_pytorch - dy_dx_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    dy_dws_paddle = dy_dws.numpy()
    ddd = np.sum((dy_dws_pytorch - dy_dws_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    ddd = np.mean((y_pytorch - y_paddle) ** 2)
    print('ddd=%.6f' % ddd)
    ddd = np.mean((dy_dx_pytorch - dy_dx_paddle) ** 2)
    print('ddd=%.6f' % ddd)
    ddd = np.mean((dy_dws_pytorch - dy_dws_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    loss = dy_dx.sum() + dy_dws.sum() + y.sum()   # 分辨率太大（如512）时，和pytorch获得不一样的结果xxx。暂时不修复这个bug
    # loss = dy_dx.sum() + y.sum()
    # loss = dy_dws.sum() + y.sum()    # 和pytorch获得一样的结果。
    # loss = y.sum()
    loss.backward()
    optimizer.step()
print()
