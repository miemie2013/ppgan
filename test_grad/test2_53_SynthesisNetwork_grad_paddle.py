
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




# 需要强制设置ppgan/models/generators/generator_styleganv2ada.py里的 SynthesisLayer 的self.use_noise = False


synthesis_cfg = dict(
    w_dim=w_dim,
    img_resolution=img_resolution,
    img_channels=img_channels,
    channel_base=channel_base,
    channel_max=channel_max,
    num_fp16_res=num_fp16_res,
    conv_clamp=conv_clamp,
)

lr = 0.0001

model = StyleGANv2ADA_SynthesisNetwork(**synthesis_cfg)
model.train()
optimizer = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=lr, momentum=0.9)
model.set_state_dict(paddle.load("53.pdparams"))


dic2 = np.load('53.npz')
for batch_idx in range(8):
    print('======================== batch_%.3d ========================'%batch_idx)
    optimizer.clear_gradients()
    ws = dic2['batch_%.3d.input1'%batch_idx]
    y_pytorch = dic2['batch_%.3d.output'%batch_idx]
    dy_dws_pytorch = dic2['batch_%.3d.dy_dws'%batch_idx]
    ws = paddle.to_tensor(ws)
    ws.stop_gradient = False
    y = model(ws)

    dy_dws = paddle.grad(outputs=[y.sum()], inputs=ws, create_graph=True)[0]


    y_paddle = y.numpy()
    ddd = np.sum((y_pytorch - y_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    dy_dws_paddle = dy_dws.numpy()
    ddd = np.sum((dy_dws_pytorch - dy_dws_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    # 需要强制设置ppgan/models/generators/generator_styleganv2ada.py里的 SynthesisLayer 的self.use_noise = False，pytorch的也要设置，才会和pytorch输出一样！！！
    loss = dy_dws.sum() + y.sum()
    # loss = y.sum()
    loss.backward()
    optimizer.step()
print('================= last dy_dws =================')
print('dy_dws_pytorch[:, :2, :2]=\n', dy_dws_pytorch[:, :2, :2])
print()
print('dy_dws_paddle[:, :2, :2]=\n', dy_dws_paddle[:, :2, :2])
print()
