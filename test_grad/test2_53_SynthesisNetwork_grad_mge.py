
import megengine as mge
from megengine.autodiff import GradManager
import numpy as np
from meg_networks import StyleGANv2ADA_SynthesisNetwork, mge_grad


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
optimizer = mge.optimizer.SGD(model.parameters(), lr=lr, momentum=0.9)
model.load_state_dict(mge.load("53.pkl", map_location="cpu"))


dic2 = np.load('53.npz')
for batch_idx in range(8):
    print('======================== batch_%.3d ========================'%batch_idx)
    optimizer.clear_grad()
    ws = dic2['batch_%.3d.input1'%batch_idx]
    y_pytorch = dic2['batch_%.3d.output'%batch_idx]
    dy_dws_pytorch = dic2['batch_%.3d.dy_dws'%batch_idx]


    model_b32_torgb_weight_grad_pytorch = dic2['batch_%.3d.model.b32.torgb.weight.grad'%batch_idx]
    model_b32_conv1_weight_grad_pytorch = dic2['batch_%.3d.model.b32.conv1.weight.grad'%batch_idx]
    model_b32_conv1_bias_grad_pytorch = dic2['batch_%.3d.model.b32.conv1.bias.grad'%batch_idx]


    ws = mge.tensor(ws)
    ws.stop_gradient = False

    gm = GradManager()
    gm2 = GradManager()
    gm.attach([ws])
    gm2.attach([ws])
    gm.attach(model.parameters())
    gm2.attach(model.parameters())


    with gm:
        with gm2:
            y = model(ws)

            dy_dws = mge_grad(gm2, output=y, inputs=[ws], clear_grad_nets=[model])[0]


            y_mge = y.numpy()
            ddd = np.sum((y_pytorch - y_mge) ** 2)
            print('ddd=%.6f' % ddd)

            dy_dws_mge = dy_dws.numpy()
            ddd = np.sum((dy_dws_pytorch - dy_dws_mge) ** 2)
            print('ddd=%.6f' % ddd)

            # 需要强制设置 SynthesisLayer 的self.use_noise = False，pytorch的也要设置，才会和pytorch输出一样！！！
            # loss = dy_dws.sum() + y.sum()
            # loss = y.sum()
            loss = dy_dws.sum()
            gm.backward(loss)

            model_b32_torgb_weight_grad_mge = model.b32.torgb.weight.grad.numpy()
            ddd = np.sum((model_b32_torgb_weight_grad_pytorch - model_b32_torgb_weight_grad_mge) ** 2)
            print('ddd=%.6f' % ddd)

            if model.b32.torgb.bias.grad is not None:
                model_b32_torgb_bias_grad_pytorch = dic2['batch_%.3d.model.b32.torgb.bias.grad' % batch_idx]
                model_b32_torgb_bias_grad_mge = model.b32.torgb.bias.grad.numpy()
                ddd = np.sum((model_b32_torgb_bias_grad_pytorch - model_b32_torgb_bias_grad_mge) ** 2)
                print('ddd=%.6f' % ddd)

            model_b32_conv1_weight_grad_mge = model.b32.conv1.weight.grad.numpy()
            ddd = np.sum((model_b32_conv1_weight_grad_pytorch - model_b32_conv1_weight_grad_mge) ** 2)
            print('ddd=%.6f' % ddd)

            model_b32_conv1_bias_grad_mge = model.b32.conv1.bias.grad.numpy()
            ddd = np.sum((model_b32_conv1_bias_grad_pytorch - model_b32_conv1_bias_grad_mge) ** 2)
            print('ddd=%.6f' % ddd)
    optimizer.step()
print('================= last dy_dws =================')
print('dy_dws_pytorch[:, :2, :2]=\n', dy_dws_pytorch[:, :2, :2])
print()
print('dy_dws_mge[:, :2, :2]=\n', dy_dws_mge[:, :2, :2])
print(mge.__version__)
print()
