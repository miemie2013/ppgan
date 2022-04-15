import os
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)
import paddle
import numpy as np
from ppgan.models.discriminators.discriminator_styleganv2ada import StyleGANv2ADA_Discriminator
from ppgan.models.generators.generator_styleganv2ada import StyleGANv2ADA_AugmentPipe


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


# 所有，除了noise = 0
xflip = 1
rotate90 = 1
xint = 1
xint_max = 0.125
scale = 1
rotate = 1
aniso = 1
xfrac = 1
scale_std = 0.2
rotate_max = 1
aniso_std = 0.2
xfrac_std = 0.125
brightness = 1
contrast = 1
lumaflip = 1
hue = 1
saturation = 1
brightness_std = 0.2
contrast_std = 0.5
hue_max = 1
saturation_std = 1
imgfilter = 1
imgfilter_bands = [1, 1, 1, 1]
imgfilter_std = 1
noise = 0
cutout = 1
noise_std = 0.1
cutout_size = 0.5


augment_pipe = StyleGANv2ADA_AugmentPipe(xflip, rotate90, xint, xint_max,
        scale, rotate, aniso, xfrac, scale_std, rotate_max, aniso_std, xfrac_std,
        brightness, contrast, lumaflip, hue, saturation, brightness_std, contrast_std, hue_max, saturation_std,
        imgfilter, imgfilter_bands, imgfilter_std,
        noise, cutout, noise_std, cutout_size)


debug_percentile = 0.7
dic2 = np.load('54.npz')
for batch_idx in range(8):
    print('======================== batch_%.3d ========================'%batch_idx)
    optimizer.clear_gradients()
    x = dic2['batch_%.3d.input0'%batch_idx]
    y_pytorch = dic2['batch_%.3d.output'%batch_idx]
    dy_dx_pytorch = dic2['batch_%.3d.dy_dx'%batch_idx]

    x = paddle.to_tensor(x)
    x.stop_gradient = False
    aug_x = augment_pipe(x, debug_percentile)
    y = model(aug_x, None)

    # dy_dx = paddle.grad(outputs=[y.sum()], inputs=[x], create_graph=True)[0]
    dy_daug_x = paddle.grad(outputs=[y.sum()], inputs=[aug_x], create_graph=True)[0]
    dy_dx = augment_pipe.grad_layer(dy_daug_x)


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

    r1_penalty = dy_dx.square().sum([1, 2, 3])
    loss_Dr1 = r1_penalty * (10 / 2)
    loss_Dr1 = loss_Dr1.mean()
    loss_Gmain = paddle.nn.functional.softplus(-y)
    loss_Gmain = loss_Gmain.mean()

    loss = loss_Dr1 + loss_Gmain
    # loss = dy_dx.sum() + y.sum()
    # loss = y.sum()
    loss.backward()
    optimizer.step()
print('================= last dy_dx =================')
print('dy_dx_pytorch[:, :2, :2, :2]=\n', dy_dx_pytorch[:, :2, :2, :2])
print()
print('dy_dx_paddle[:, :2, :2, :2]=\n', dy_dx_paddle[:, :2, :2, :2])
print()
