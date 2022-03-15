import torch
import numpy as np
from training.networks import Discriminator
from training.augment import AugmentPipe


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

model = Discriminator(c_dim=c_dim,
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
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
torch.save(model.state_dict(), "54.pth")


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

augment_pipe = AugmentPipe(xflip, rotate90, xint, xint_max,
        scale, rotate, aniso, xfrac, scale_std, rotate_max, aniso_std, xfrac_std,
        brightness, contrast, lumaflip, hue, saturation, brightness_std, contrast_std, hue_max, saturation_std,
        imgfilter, imgfilter_bands, imgfilter_std,
        noise, cutout, noise_std, cutout_size)


debug_percentile = 0.7
dic = {}
for batch_idx in range(8):
    optimizer.zero_grad(set_to_none=True)

    x = torch.randn(x_shape)
    x.requires_grad_(True)

    aug_x = augment_pipe(x, debug_percentile)
    y = model(aug_x, None)
    dy_dx = torch.autograd.grad(outputs=[y.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dy_dx'%batch_idx] = dy_dx.cpu().detach().numpy()
    dic['batch_%.3d.output'%batch_idx] = y.cpu().detach().numpy()
    dic['batch_%.3d.input0'%batch_idx] = x.cpu().detach().numpy()

    loss = dy_dx.sum() + y.sum()
    # loss = y.sum()
    loss.backward()
    optimizer.step()
np.savez('54', **dic)
print()
