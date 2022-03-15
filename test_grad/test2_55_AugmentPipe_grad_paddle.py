
import paddle
import numpy as np
from ppgan.models.generators.generator_styleganv2ada import StyleGANv2ADA_AugmentPipe


# 默认配置
xflip = 0
rotate90 = 0
xint = 0
xint_max = 0.125
scale = 0
rotate = 0
aniso = 0
xfrac = 0
scale_std = 0.2
rotate_max = 1
aniso_std = 0.2
xfrac_std = 0.125
brightness = 0
contrast = 0
lumaflip = 0
hue = 0
saturation = 0
brightness_std = 0.2
contrast_std = 0.5
hue_max = 1
saturation_std = 1
imgfilter = 0
imgfilter_bands = [1, 1, 1, 1]
imgfilter_std = 1
noise = 0
cutout = 0
noise_std = 0.1
cutout_size = 0.5

# afhqcat配置
# xflip = 1
# rotate90 = 1
# xint = 1
# xint_max = 0.125
# scale = 1
# rotate = 1
# aniso = 1
# xfrac = 1
# scale_std = 0.2
# rotate_max = 1
# aniso_std = 0.2
# xfrac_std = 0.125
# brightness = 1
# contrast = 1
# lumaflip = 1
# hue = 1
# saturation = 1
# brightness_std = 0.2
# contrast_std = 0.5
# hue_max = 1
# saturation_std = 1
# imgfilter = 0
# imgfilter_bands = [1, 1, 1, 1]
# imgfilter_std = 1
# noise = 0
# cutout = 0
# noise_std = 0.1
# cutout_size = 0.5

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


lr = 0.0001

model = StyleGANv2ADA_AugmentPipe(xflip, rotate90, xint, xint_max,
        scale, rotate, aniso, xfrac, scale_std, rotate_max, aniso_std, xfrac_std,
        brightness, contrast, lumaflip, hue, saturation, brightness_std, contrast_std, hue_max, saturation_std,
        imgfilter, imgfilter_bands, imgfilter_std,
        noise, cutout, noise_std, cutout_size)

model.train()
# optimizer = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=lr, momentum=0.9)
# model.set_state_dict(paddle.load("55.pdparams"))

debug_percentile = 0.7
dic2 = np.load('55.npz')
for batch_idx in range(8):
    print('======================== batch_%.3d ========================'%batch_idx)
    # optimizer.clear_gradients()
    x = dic2['batch_%.3d.input0'%batch_idx]
    y_pytorch = dic2['batch_%.3d.output'%batch_idx]
    dy_dx_pytorch = dic2['batch_%.3d.dy_dx'%batch_idx]

    x = paddle.to_tensor(x)
    x.stop_gradient = False
    y = model(x, debug_percentile)

    # dy_dx = paddle.grad(outputs=[y.sum()], inputs=[x], create_graph=True)[0]
    # dy_dx = paddle.grad(outputs=[y.sum()], inputs=[x], create_graph=False)[0]
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

    # loss = dy_dx.sum() + y.sum()
    # loss = y.sum()
    # loss.backward()
    # optimizer.step()
print('================= last dy_dx =================')
print('dy_dx_pytorch[:, :2, :2, :2]=\n', dy_dx_pytorch[:, :2, :2, :2])
print()
print('dy_dx_paddle[:, :2, :2, :2]=\n', dy_dx_paddle[:, :2, :2, :2])
print()
