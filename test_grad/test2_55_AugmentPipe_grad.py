
import torch
import numpy as np
from training.augment import AugmentPipe


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



batch_size = 2
img_channels = 3
# img_channels = 1
img_resolution = 32
x_shape = [batch_size, img_channels, img_resolution, img_resolution]
lr = 0.0001

model = AugmentPipe(xflip, rotate90, xint, xint_max,
        scale, rotate, aniso, xfrac, scale_std, rotate_max, aniso_std, xfrac_std,
        brightness, contrast, lumaflip, hue, saturation, brightness_std, contrast_std, hue_max, saturation_std,
        imgfilter, imgfilter_bands, imgfilter_std,
        noise, cutout, noise_std, cutout_size)


model.train()
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# torch.save(model.state_dict(), "55.pth")

debug_percentile = 0.7
dic = {}
for batch_idx in range(8):
    # optimizer.zero_grad(set_to_none=True)

    x = torch.randn(x_shape)
    x.requires_grad_(True)

    y = model(x, debug_percentile)
    dy_dx = torch.autograd.grad(outputs=[y.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dy_dx'%batch_idx] = dy_dx.cpu().detach().numpy()
    dic['batch_%.3d.output'%batch_idx] = y.cpu().detach().numpy()
    dic['batch_%.3d.input0'%batch_idx] = x.cpu().detach().numpy()

    # loss = dy_dx.sum() + y.sum()
    # loss = y.sum()
    # loss.backward()
    # optimizer.step()
np.savez('55', **dic)
print()
