
import paddle
import numpy as np
from ppgan.models.discriminators.discriminator_styleganv2ada import StyleGANv2ADA_Discriminator


c_dim = 0
w_dim = 512
# img_resolution = 512
# img_resolution = 128
img_resolution = 32
img_channels = 3
channel_base = 32768
channel_max = 512
num_fp16_res = 4
conv_clamp = 256
epilogue_kwargs = dict(
    mbstd_group_size=8,
)


batch_size = 2
x_shape = [batch_size, img_channels, img_resolution, img_resolution]
lr = 0.0001

model = StyleGANv2ADA_Discriminator(c_dim=c_dim,
                                            img_resolution=img_resolution,
                                            img_channels=img_channels,
                                            channel_base=channel_base,
                                            channel_max=channel_max,
                                            num_fp16_res=num_fp16_res,
                                            conv_clamp=conv_clamp,
                                            block_kwargs={},
                                            mapping_kwargs={},
                                            epilogue_kwargs=epilogue_kwargs,
                                            )
model.train()
optimizer = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=lr, momentum=0.9)
model.set_state_dict(paddle.load("54.pdparams"))


dic2 = np.load('54.npz')
for batch_idx in range(8):
    print('======================== batch_%.3d ========================'%batch_idx)
    optimizer.clear_gradients()
    x = dic2['batch_%.3d.input0'%batch_idx]
    y_pytorch = dic2['batch_%.3d.output'%batch_idx]
    dy_dx_pytorch = dic2['batch_%.3d.dy_dx'%batch_idx]

    x = paddle.to_tensor(x)
    x.stop_gradient = False
    y = model(x, None)

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
