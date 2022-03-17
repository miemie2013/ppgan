import paddle.nn as nn
import paddle

from .builder import DISCRIMINATORS
from ..generators.generator_styleganv2ada import StyleGANv2ADA_MappingNetwork, upfirdn2d_setup_filter, Conv2dLayer, FullyConnectedLayer, downsample2d

import numpy as np


class DiscriminatorBlock(nn.Layer):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.grad_layer = DiscriminatorBlock_Grad(
            in_channels,
            tmp_channels,
            out_channels,
            resolution,
            img_channels,
            first_layer_idx,
            architecture,
            activation,
            resample_filter,
            conv_clamp,
            use_fp16,
            fp16_channels_last,
            freeze_layers,
        )
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d_setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.grad_layer.fromrgb = self.fromrgb

        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
            trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)
        self.grad_layer.conv0 = self.conv0
        self.grad_layer.conv1 = self.conv1

        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)
            self.grad_layer.skip = self.skip

    def forward(self, x, img, force_fp32=False):
        dtype = paddle.float16 if self.use_fp16 and not force_fp32 else paddle.float32
        # 假设屎山的channels_last都是False
        assert self.channels_last == False
        # memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            # x = x.to(dtype=dtype, memory_format=memory_format)
            x = paddle.cast(x, dtype=dtype)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            # img = img.to(dtype=dtype, memory_format=memory_format)
            img = paddle.cast(img, dtype=dtype)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y + x
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img

class DiscriminatorBlock_Grad(object):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.num_layers = 0

    def __call__(self, dloss_dout, dloss_dimg):
        # Main layers.
        if self.architecture == 'resnet':
            dloss_dx_1 = dloss_dout
            dloss_dx_2 = dloss_dout

            dloss_dx_1 = self.conv1.grad_layer(dloss_dx_1)
            dloss_dx_1 = self.conv0.grad_layer(dloss_dx_1)
            dloss_dx_2 = self.skip.grad_layer(dloss_dx_2)

            dloss_dx = dloss_dx_1 + dloss_dx_2
        else:
            raise NotImplementedError("not implemented.")
        if self.in_channels != 0:
            dloss_dimg = None

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            if self.architecture == 'skip':
                raise NotImplementedError("not implemented.")
            dloss_dimg = self.fromrgb.grad_layer(dloss_dx)

        # Input.

        return dloss_dx, dloss_dimg

class MinibatchStdLayer(nn.Layer):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels
        self.grad_layer = MinibatchStdLayer_Grad(
            group_size,
            num_channels,
        )

    def forward(self, x):
        self.grad_layer.x = x
        N, C, H, W = x.shape
        # G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        G = min(self.group_size, N) if self.group_size is not None else N
        F = self.num_channels
        c = C // F
        self.grad_layer.G = G
        self.grad_layer.c = c

        y0 = x.reshape((G, -1, F, c, H, W))    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y1 = y0 - y0.mean(0)                   # [GnFcHW] Subtract mean over group.
        y2 = y1.square().mean(0)              # [nFcHW]  Calc variance over group.
        y3 = (y2 + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y4 = y3.mean([2, 3, 4])               # [nF]     Take average over channels and pixels.
        y5 = y4.reshape((-1, F, 1, 1))          # [nF11]   Add missing dimensions.
        y6 = y5.tile([G, 1, H, W])            # [NFHW]   Replicate over group and pixels.
        out = paddle.concat([x, y6], 1)        # [NCHW]   Append to input as new channels.
        self.grad_layer.y0 = y0
        self.grad_layer.y1 = y1
        self.grad_layer.y2 = y2
        self.grad_layer.y3 = y3
        self.grad_layer.y4 = y4
        self.grad_layer.y5 = y5
        self.grad_layer.y6 = y6
        self.grad_layer.out = out
        return out

class MinibatchStdLayer_Grad(object):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def __call__(self, dloss_dout):
        x = self.x
        N, C, H, W = x.shape
        G = self.G
        c = self.c
        F = self.num_channels
        y0 = self.y0
        y1 = self.y1
        y2 = self.y2
        y3 = self.y3
        y4 = self.y4
        y5 = self.y5
        y6 = self.y6
        out = self.out

        dloss_dx_1 = dloss_dout[:, :C, :, :]
        dloss_dx_2 = dloss_dout[:, C:, :, :]
        dloss_dy6 = dloss_dx_2

        dloss_dy5 = dloss_dy6.reshape((y5.shape[0], -1, F, H, W))
        dloss_dy5 = paddle.sum(dloss_dy5, axis=[1], keepdim=False)
        dloss_dy5 = paddle.sum(dloss_dy5, axis=[2, 3], keepdim=True)

        dloss_dy4 = dloss_dy5.reshape((-1, F))

        dloss_dy3 = dloss_dy4.reshape((-1, F, 1, 1, 1))
        dloss_dy3 = dloss_dy3.tile([1, 1, c, H, W]) / c / H / W

        dloss_dy2 = dloss_dy3 * 0.5 / y3

        dloss_dy1 = dloss_dy2.reshape((1, -1, F, c, H, W))
        dloss_dy1 = dloss_dy1.tile([G, 1, 1, 1, 1, 1]) / G
        dloss_dy1 = dloss_dy1 * 2 * y1

        dloss_dy0 = dloss_dy1 - dloss_dy1.mean(0)

        dloss_dx = dloss_dy0.reshape(x.shape)

        dloss_dx = dloss_dx + dloss_dx_1
        return dloss_dx

class DiscriminatorEpilogue(nn.Layer):
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.grad_layer = DiscriminatorEpilogue_Grad(
            in_channels,
            cmap_dim,
            resolution,
            img_channels,
            architecture,
            mbstd_group_size,
            mbstd_num_channels,
            activation,
            conv_clamp,
        )
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
            self.grad_layer.fromrgb = self.fromrgb
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)
        self.grad_layer.mbstd = self.mbstd
        self.grad_layer.conv = self.conv
        self.grad_layer.fc = self.fc
        self.grad_layer.out = self.out

    def forward(self, x, img, cmap, force_fp32=False):
        _ = force_fp32 # unused
        dtype = paddle.float32

        # FromRGB.
        # x = x.to(dtype=dtype, memory_format=memory_format)
        x = paddle.cast(x, dtype=dtype)
        if self.architecture == 'skip':
            # img = img.to(dtype=dtype, memory_format=memory_format)
            img = paddle.cast(img, dtype=dtype)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        # flatten_x = x.flatten(1)   # 因为flatten()没有实现二阶梯度，所以用其它等价实现。
        self.grad_layer.x_shape = x.shape
        batch_size = x.shape[0]
        flatten_x = x.reshape((batch_size, -1))
        x = self.fc(flatten_x)
        x = self.out(x)

        # Conditioning.
        if self.cmap_dim > 0:
            x = (x * cmap).sum(axis=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
        return x


class DiscriminatorEpilogue_Grad(object):
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

    def __call__(self, dloss_dout):
        # dtype = paddle.float32

        # Conditioning.
        if self.cmap_dim > 0:
            raise NotImplementedError("not implemented.")

        # Main layers.
        dloss_dx = self.out.grad_layer(dloss_dout)
        dloss_dflatten_x = self.fc.grad_layer(dloss_dx)
        dloss_dx = dloss_dflatten_x.reshape(self.x_shape)
        dloss_dx = self.conv.grad_layer(dloss_dx)
        if self.mbstd is not None:
            dloss_dx = self.mbstd.grad_layer(dloss_dx)

        # FromRGB.
        if self.architecture == 'skip':
            raise NotImplementedError("not implemented.")
        # dloss_dx = paddle.cast(dloss_dx, dtype=dtype)
        return dloss_dx


@DISCRIMINATORS.register()
class StyleGANv2ADA_Discriminator(nn.Layer):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        self.grad_layer = StyleGANv2ADA_Discriminator_Grad(
            c_dim,
            img_resolution,
            img_channels,
            architecture,
            channel_base,
            channel_max,
            num_fp16_res,
            conv_clamp,
            cmap_dim,
            block_kwargs,
            mapping_kwargs,
            epilogue_kwargs,
        )
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            # use_fp16 = (res >= fp16_resolution)
            use_fp16 = False
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            setattr(self.grad_layer, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = StyleGANv2ADA_MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
            self.grad_layer.mapping = self.mapping
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)
        self.grad_layer.b4 = self.b4
        self.grad_layer.block_resolutions = self.block_resolutions

    def forward(self, img, c, **block_kwargs):
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x


class StyleGANv2ADA_Discriminator_Grad(object):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        self.c_dim = c_dim

    def __call__(self, dloss_dout):
        dloss_dx = self.b4.grad_layer(dloss_dout)

        if self.c_dim > 0:
            raise NotImplementedError("not implemented.")

        dloss_dimg = None
        for block_idx in range(len(self.block_resolutions) - 1, -1, -1):
            res = self.block_resolutions[block_idx]
            block = getattr(self, f'b{res}')
            dloss_dx, dloss_dimg = block.grad_layer(dloss_dx, dloss_dimg)
        return dloss_dimg


