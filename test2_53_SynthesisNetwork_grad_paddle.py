
import paddle
import numpy as np
from ppgan.models.generators.generator_styleganv2ada import StyleGANv2ADA_SynthesisNetwork


w_dim = 512
# img_resolution = 512
# img_resolution = 128
img_resolution = 32
img_channels = 3
channel_base = 32768
channel_max = 512
num_fp16_res = 4
conv_clamp = 256




# 需要强制设置 SynthesisLayer 的self.use_noise = False


synthesis_cfg = dict(
    w_dim=w_dim,
    img_resolution=img_resolution,
    img_channels=img_channels,
    channel_base=channel_base,
    channel_max=channel_max,
    num_fp16_res=num_fp16_res,
    conv_clamp=conv_clamp,
)

batch_size = 2
w_shape = [batch_size, w_dim]
lr = 0.0001

model = StyleGANv2ADA_SynthesisNetwork(**synthesis_cfg)
model.train()
optimizer = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=lr, momentum=0.9)
model.set_state_dict(paddle.load("53.pdparams"))


dic2 = np.load('53.npz')
for batch_idx in range(8):
    print('======================== batch_%.3d ========================'%batch_idx)
    optimizer.clear_gradients()
    ws_ = dic2['batch_%.3d.input1'%batch_idx]
    y_pytorch = dic2['batch_%.3d.output'%batch_idx]
    dy_dws_pytorch = dic2['batch_%.3d.dy_dws'%batch_idx]

    ws = []
    for kkk in range(model.num_ws):
        aaaaaaaaaaa = ws_[:, kkk, :]
        aaaaaaaaaaa = paddle.to_tensor(aaaaaaaaaaa)
        aaaaaaaaaaa.stop_gradient = False
        ws.append(aaaaaaaaaaa)
    y = model(ws)

    # dy_dws_list = paddle.grad(outputs=[y.sum()], inputs=ws, create_graph=True)
    # dy_dws = paddle.stack(dy_dws_list, 1)
    dysum_dy = paddle.ones(y.shape, dtype=paddle.float32)
    dy_dws = model.grad_layer(dysum_dy)
    dy_dws = paddle.stack(dy_dws, 1)


    y_paddle = y.numpy()
    ddd = np.sum((y_pytorch - y_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    dy_dws_paddle = dy_dws.numpy()
    ddd = np.sum((dy_dws_pytorch - dy_dws_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    ddd = np.mean((y_pytorch - y_paddle) ** 2)
    print('ddd=%.6f' % ddd)
    ddd = np.mean((dy_dws_pytorch - dy_dws_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    # 需要强制设置SynthesisLayer的self.use_noise = False，pytorch的也要设置，才会和pytorch输出一样！！！
    loss = dy_dws.sum() + y.sum()
    # loss = y.sum()
    loss.backward()
    optimizer.step()
print()
