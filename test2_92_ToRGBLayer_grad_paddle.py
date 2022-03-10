
import paddle
import numpy as np
from ppgan.models.generators.generator_styleganv2ada import ToRGBLayer

w_dim = 512
out_channels = 512
img_channels = 3
conv_clamp = 256
channels_last = False
fused_modconv = False
batch_size = 2
lr = 0.0001

model = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                   conv_clamp=conv_clamp, channels_last=channels_last)
model.train()
optimizer = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=lr, momentum=0.9)
model.set_state_dict(paddle.load("pytorch_toRGBLayer.pdparams"))


dic2 = np.load('02_toRGBLayer_grad.npz')
for batch_idx in range(20):
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
    y = model(x, ws, fused_modconv=fused_modconv)

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

    # loss = dy_dx.sum() + dy_dws.sum() + y.sum()
    # loss = dy_dx.sum() + y.sum()
    loss = dy_dws.sum() + y.sum()
    # loss = y.sum()
    loss.backward()
    optimizer.step()
print()
