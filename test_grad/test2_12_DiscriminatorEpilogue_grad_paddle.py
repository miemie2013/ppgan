
import paddle
import numpy as np
from ppgan.models.discriminators.discriminator_styleganv2ada import DiscriminatorEpilogue


in_channels = 512
cmap_dim = 0
resolution = 4
img_channels = 3
architecture = 'resnet'
mbstd_group_size = None
mbstd_num_channels = 1




x_shape = [2, in_channels, 4, 4]
lr = 0.0001

model = DiscriminatorEpilogue(in_channels, cmap_dim, resolution, img_channels, architecture, mbstd_group_size, mbstd_num_channels)
model.train()
optimizer = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=lr, momentum=0.9)
model.set_state_dict(paddle.load("12.pdparams"))


dic2 = np.load('12.npz')
for batch_idx in range(8):
    print('======================== batch_%.3d ========================'%batch_idx)
    optimizer.clear_gradients()
    y_pytorch = dic2['batch_%.3d.output'%batch_idx]
    dy_dx_pytorch = dic2['batch_%.3d.dy_dx'%batch_idx]
    x = dic2['batch_%.3d.input0'%batch_idx]
    x = paddle.to_tensor(x)
    x.stop_gradient = False

    y = model(x, None, None)

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
