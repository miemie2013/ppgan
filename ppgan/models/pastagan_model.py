# code was heavily based on https://github.com/clovaai/stargan-v2
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/clovaai/stargan-v2#license
import cv2
from paddle.fluid.layers.nn import soft_relu
from .base_model import BaseModel

from paddle import nn
import paddle
import paddle.nn.functional as F
from .builder import MODELS
from .generators.builder import build_generator
from .discriminators.builder import build_discriminator
from ..modules.init import kaiming_normal_, constant_
from ppgan.utils.visual import make_grid, tensor2img

import numpy as np



def compute_d_loss(nets,
                   lambda_reg,
                   x_real,
                   y_org,
                   y_trg,
                   z_trg=None,
                   x_ref=None,
                   masks=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.stop_gradient = False   # x_real [N, 3, 256, 256]
    out = nets['discriminator'](x_real, y_org)  # [N, ]  输出只取真实类别处的输出。
    loss_real = adv_loss(out, 1)  # [N, ]  交叉熵损失。这是真的图像
    loss_reg = r1_reg(out, x_real)  # [N, ]  梯度惩罚损失

    # with fake images
    with paddle.no_grad():  # 训练判别器时，生成器前向传播应停止梯度。
        if z_trg is not None:
            s_trg = nets['mapping_network'](z_trg, y_trg)   # (N, style_dim)  随机噪声z_trg生成风格编码s_trg，只取目标domain的输出
        else:  # x_ref is not None
            s_trg = nets['style_encoder'](x_ref, y_trg)   # (N, style_dim)  目标domain真实图像x_ref生成风格编码s_trg

        x_fake = nets['generator'](x_real, s_trg, masks=masks)  # 风格编码s_trg和真实图像生成目标domain的图像x_fake
    out = nets['discriminator'](x_fake, y_trg)  # x_fake [N, 3, 256, 256]  注意，x_fake已经停止梯度。   out [N, ]  输出只取真实(目标domain)类别处的输出。
    loss_fake = adv_loss(out, 0)  # [N, ]  交叉熵损失。这是假的图像

    loss = loss_real + loss_fake + lambda_reg * loss_reg   # lambda_reg是梯度惩罚损失的权重
    return loss, {
        'real': loss_real.numpy(),
        'fake': loss_fake.numpy(),
        'reg': loss_reg.numpy()
    }


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = paddle.full_like(logits, fill_value=target)  # [N, ]  标记
    loss = F.binary_cross_entropy_with_logits(logits, targets)  # [N, ]  交叉熵损失
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.shape[0]
    grad_dout = paddle.grad(outputs=d_out.sum(),
                            inputs=x_in,
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.shape == x_in.shape)
    reg = 0.5 * paddle.reshape(grad_dout2, (batch_size, -1)).sum(1).mean(0)
    return reg


def soft_update(source, target, beta=1.0):
    '''
    ema:
    target = beta * source + (1. - beta) * target

    '''
    assert 0.0 <= beta <= 1.0

    if isinstance(source, paddle.DataParallel):
        source = source._layers

    target_model_map = dict(target.named_parameters())
    for param_name, source_param in source.named_parameters():
        target_param = target_model_map[param_name]
        target_param.set_value(beta * source_param +
                               (1.0 - beta) * target_param)


def dump_model(model):
    params = {}
    for k in model.state_dict().keys():
        if k.endswith('.scale'):
            params[k] = model.state_dict()[k].shape
    return params


def compute_g_loss(nets,
                   w_hpf,
                   lambda_sty,
                   lambda_ds,
                   lambda_cyc,
                   x_real,
                   y_org,
                   y_trg,
                   z_trgs=None,
                   x_refs=None,
                   masks=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss。对抗损失
    if z_trgs is not None:
        s_trg = nets['mapping_network'](z_trg, y_trg)   # 第1个随机噪声z_trg 生成风格编码s_trg
    else:
        s_trg = nets['style_encoder'](x_ref, y_trg)   # 目标domain的第1张真实图像x_ref 生成风格编码s_trg

    x_fake = nets['generator'](x_real, s_trg, masks=masks)   # 真实图像和第1个风格编码s_trg生成第1张假图像
    # 这里不先把discriminator冻结起来吗？懂了，后面没有optimizers['discriminator'].step()这句代码，所以discriminator的参数不会更新的。
    out = nets['discriminator'](x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss。风格重构损失
    s_pred = nets['style_encoder'](x_fake, y_trg)   # (N, style_dim)  假图像生成对应domain的风格编码s_pred
    loss_sty = paddle.mean(paddle.abs(s_pred - s_trg))   # 假图像生成对应domain的风格编码s_pred 和 s_trg 取绝对值损失。

    # diversity sensitive loss。差异敏感损失
    if z_trgs is not None:
        s_trg2 = nets['mapping_network'](z_trg2, y_trg)   # 第2个随机噪声z_trg2 生成风格编码s_trg2
    else:
        s_trg2 = nets['style_encoder'](x_ref2, y_trg)   # 目标domain的第2张真实图像x_ref2 生成风格编码s_trg2
    x_fake2 = nets['generator'](x_real, s_trg2, masks=masks)   # 真实图像和第2个风格编码s_trg2生成第2张假图像
    loss_ds = paddle.mean(paddle.abs(x_fake - x_fake2))   # 第1张假图像 和 第2张假图像 取绝对值损失。

    # cycle-consistency loss。循环一致性损失
    if w_hpf > 0:
        if isinstance(nets['fan'], paddle.DataParallel):
            masks = nets['fan']._layers.get_heatmap(x_fake)
        else:
            masks = nets['fan'].get_heatmap(x_fake)
    else:
        masks = None

    s_org = nets['style_encoder'](x_real, y_org)   # x_real 生成风格编码s_org
    x_rec = nets['generator'](x_fake, s_org, masks=masks)   # x_fake“变回”x_real(x_rec)
    loss_cyc = paddle.mean(paddle.abs(x_rec - x_real))   # x_real 和 x_rec 取绝对值损失。

    loss = loss_adv + lambda_sty * loss_sty \
        - lambda_ds * loss_ds + lambda_cyc * loss_cyc
    return loss, {
        'adv': loss_adv.numpy(),
        'sty': loss_sty.numpy(),
        'ds:': loss_ds.numpy(),
        'cyc': loss_cyc.numpy()
    }


def he_init(module):
    if isinstance(module, nn.Conv2D):
        kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            constant_(module.bias, 0)


@MODELS.register()
class PastaGANModel(BaseModel):
    def __init__(
        self,
        synthesis,
        mapping=None,
        const_encoding=None,
        style_encoding=None,
        discriminator=None,
        latent_dim=16,
        lambda_reg=1,
        lambda_sty=1,
        lambda_ds=1,
        lambda_cyc=1,
    ):
        super(PastaGANModel, self).__init__()
        self.nets_ema = {}
        self.nets['synthesis'] = build_generator(synthesis)
        self.nets_ema['synthesis'] = build_generator(synthesis)
        self.nets['mapping'] = build_generator(mapping)
        self.nets_ema['mapping'] = build_generator(mapping)
        self.nets['const_encoding'] = build_generator(const_encoding)
        self.nets_ema['const_encoding'] = build_generator(const_encoding)
        self.nets['style_encoding'] = build_generator(style_encoding)
        self.nets_ema['style_encoding'] = build_generator(style_encoding)
        if discriminator:
            self.nets['discriminator'] = build_discriminator(discriminator)
        self.latent_dim = latent_dim
        self.lambda_reg = lambda_reg
        self.lambda_sty = lambda_sty
        self.lambda_ds = lambda_ds
        self.lambda_cyc = lambda_cyc

        # self.nets['generator'].apply(he_init)
        # self.nets['style_encoder'].apply(he_init)
        # self.nets['mapping_network'].apply(he_init)
        # self.nets['discriminator'].apply(he_init)

        # remember the initial value of ds weight
        # self.initial_lambda_ds = self.lambda_ds

    def setup_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Args:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.input = input

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    def _reset_grad(self, optims):
        for optim in optims.values():
            optim.clear_gradients()

    def train_iter(self, optimizers=None):
        #TODO

        # x_real [N, 3, 256, 256]
        # y_org  [N, ]  x_real的类别id
        x_real, y_org = self.input['src'], self.input['src_cls']
        # x_ref  [N, 3, 256, 256]
        # x_ref2 [N, 3, 256, 256]  x_real的类别id
        # y_trg  [N, ]  x_ref和x_ref2的类别id
        x_ref, x_ref2, y_trg = self.input['ref'], self.input['ref2'], self.input['ref_cls']
        # z_trg  [N, 16]  随机噪声z
        # z_trg2 [N, 16]  随机噪声z2
        z_trg, z_trg2 = self.input['z_trg'], self.input['z_trg2']

        if self.w_hpf > 0:
            if isinstance(self.nets['fan'], paddle.DataParallel):
                masks = self.nets['fan']._layers.get_heatmap(x_real)
            else:
                masks = self.nets['fan'].get_heatmap(x_real)
        else:
            masks = None

        # 查看masks
        # m0, m1 = masks
        # aaa = x_real.numpy()[0]
        # aaa = aaa.transpose(1, 2, 0)
        # aaa = (aaa + 1.0) * 127.5
        # aaa = cv2.cvtColor(aaa, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('aaa1.png', aaa)
        # m0 = m0.numpy()[0][0]
        # m1 = m1.numpy()[0][0]
        # cv2.imwrite('aaa2.png', m0 * 255.0)
        # cv2.imwrite('aaa3.png', m1 * 255.0)

        # ================ train the discriminator ================
        # 训练了2次判别器。第1次和第2次的区别是如何生成假图像：
        # 第1次用      随机噪声z_trg     经过mapping_network生成 风格编码s_trg，再用s_trg和真实图像x_real生成假图像；
        # 第2次用 目标domain真实图像x_ref  经过style_encoder生成  风格编码s_trg，再用s_trg和真实图像x_real生成假图像；

        # lambda_reg是梯度惩罚损失的权重。包括计算
        # (1)真实图像的判断损失（交叉熵）；(2)真实图像的梯度惩罚损失；
        # (3)随机噪声z_trg和真实图像x_real生成的假图像的判断损失（交叉熵）；
        d_loss, d_losses_latent = compute_d_loss(self.nets,
                                                 self.lambda_reg,
                                                 x_real,
                                                 y_org,
                                                 y_trg,
                                                 z_trg=z_trg,
                                                 masks=masks)
        self._reset_grad(optimizers)  # 梯度清0
        d_loss.backward()  # 反向传播
        optimizers['discriminator'].minimize(d_loss)  # 更新参数

        # lambda_reg是梯度惩罚损失的权重。包括计算
        # (1)真实图像的判断损失（交叉熵）；(2)真实图像的梯度惩罚损失；
        # (3)目标domain真实图像x_ref和真实图像x_real生成的假图像的判断损失（交叉熵）；
        d_loss, d_losses_ref = compute_d_loss(self.nets,
                                              self.lambda_reg,
                                              x_real,
                                              y_org,
                                              y_trg,
                                              x_ref=x_ref,
                                              masks=masks)
        self._reset_grad(optimizers)  # 梯度清0
        d_loss.backward()  # 反向传播
        optimizers['discriminator'].step()  # 更新参数

        # ================ train the generator ================
        # 训练了2次生成器。第1次和第2次的区别是如何生成假图像：
        # 第1次用      随机噪声z_trg     经过mapping_network生成 风格编码s_trg，再用s_trg和真实图像x_real生成假图像；
        # 第2次用 目标domain真实图像x_ref  经过style_encoder生成  风格编码s_trg，再用s_trg和真实图像x_real生成假图像；

        g_loss, g_losses_latent = compute_g_loss(self.nets,
                                                 self.w_hpf,
                                                 self.lambda_sty,
                                                 self.lambda_ds,
                                                 self.lambda_cyc,
                                                 x_real,
                                                 y_org,
                                                 y_trg,
                                                 z_trgs=[z_trg, z_trg2],
                                                 masks=masks)
        self._reset_grad(optimizers)
        g_loss.backward()
        optimizers['generator'].step()
        optimizers['mapping_network'].step()
        optimizers['style_encoder'].step()

        g_loss, g_losses_ref = compute_g_loss(self.nets,
                                              self.w_hpf,
                                              self.lambda_sty,
                                              self.lambda_ds,
                                              self.lambda_cyc,
                                              x_real,
                                              y_org,
                                              y_trg,
                                              x_refs=[x_ref, x_ref2],
                                              masks=masks)
        self._reset_grad(optimizers)
        g_loss.backward()
        optimizers['generator'].step()

        # compute moving average of network parameters。指数滑动平均
        soft_update(self.nets['generator'],
                    self.nets_ema['generator'],
                    beta=0.999)
        soft_update(self.nets['mapping_network'],
                    self.nets_ema['mapping_network'],
                    beta=0.999)
        soft_update(self.nets['style_encoder'],
                    self.nets_ema['style_encoder'],
                    beta=0.999)

        # decay weight for diversity sensitive loss
        if self.lambda_ds > 0:
            self.lambda_ds -= (self.initial_lambda_ds / self.total_iter)

        for loss, prefix in zip(
            [d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
            ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
            for key, value in loss.items():
                self.losses[prefix + key] = value
        self.losses['G/lambda_ds'] = self.lambda_ds
        self.losses['Total iter'] = int(self.total_iter)

    def test_iter(self, metrics=None):
        #TODO
        self.nets_ema['synthesis'].eval()
        self.nets_ema['mapping'].eval()
        self.nets_ema['const_encoding'].eval()
        self.nets_ema['style_encoding'].eval()
        # soft_update(self.nets['synthesis'],
        #             self.nets_ema['synthesis'],
        #             beta=0.999)
        # soft_update(self.nets['mapping'],
        #             self.nets_ema['mapping'],
        #             beta=0.999)
        # soft_update(self.nets['const_encoding'],
        #             self.nets_ema['const_encoding'],
        #             beta=0.999)
        # soft_update(self.nets['style_encoding'],
        #             self.nets_ema['style_encoding'],
        #             beta=0.999)

        image = self.input['image']
        pose = self.input['pose']
        norm_img = self.input['norm_img']
        denorm_upper_clothes = self.input['denorm_upper_img']
        denorm_lower_clothes = self.input['denorm_lower_img']
        denorm_upper_mask = self.input['denorm_upper_mask']
        denorm_lower_mask = self.input['denorm_lower_mask']
        batchsize = image.shape[0]
        # batchsize > 1时，还要实现一下两两之间的换装。
        assert batchsize == 1
        person_name = self.input['person_name'][0]
        clothes_name = self.input['clothes_name'][0]
        src_img = self.input['src_img']
        trg_img = self.input['trg_img']
        src_img = src_img.numpy()[0]
        trg_img = trg_img.numpy()[0]

        im_shape = src_img.shape
        h, w = im_shape[0], im_shape[1]
        left_padding = (h - w) // 2
        right_padding = h - w - left_padding
        gray_img = np.ones((h, w, 3), np.uint8) * 127


        image = paddle.cast(image, dtype=paddle.float32)
        pose = paddle.cast(pose, dtype=image.dtype)
        norm_img = paddle.cast(norm_img, dtype=image.dtype)
        denorm_upper_clothes = paddle.cast(denorm_upper_clothes, dtype=image.dtype)
        denorm_lower_clothes = paddle.cast(denorm_lower_clothes, dtype=image.dtype)
        denorm_upper_mask = paddle.cast(denorm_upper_mask, dtype=image.dtype)
        denorm_lower_mask = paddle.cast(denorm_lower_mask, dtype=image.dtype)

        image = image / 127.5 - 1
        pose = pose / 127.5 - 1
        norm_img_c = norm_img / 127.5 - 1
        denorm_upper_clothes = denorm_upper_clothes / 127.5 - 1
        denorm_lower_clothes = denorm_lower_clothes / 127.5 - 1


        retain_tensor = image
        pose_tensor = paddle.concat([pose, retain_tensor], 1)
        # gen_z = paddle.randn([batchsize, 0])
        gen_z = None

        with paddle.no_grad():
            gen_c, cat_feat_list = self.nets['style_encoding'](norm_img_c, retain_tensor)
            pose_feat = self.nets['const_encoding'](pose_tensor)
            ws = self.nets['mapping'](gen_z, gen_c)
            cat_feats = {}
            for cat_feat in cat_feat_list:
                h = cat_feat.shape[2]
                cat_feats[str(h)] = cat_feat
            # 因为有噪声，所以每一次的结果有点小差别
            gen_coarse_imgs, gen_imgs, _, _ = self.nets['synthesis'](ws,
                                                                         pose_feat, cat_feats, denorm_upper_clothes,
                                                                         denorm_lower_clothes, denorm_upper_mask,
                                                                         denorm_lower_mask)
            gen_imgs = gen_imgs.numpy()
            for ii in range(batchsize):
                gen_img = gen_imgs[ii]
                gen_img = (gen_img.transpose(1, 2, 0) + 1.0) * 127.5
                gen_img = gen_img[:, left_padding:left_padding+w, [2, 1, 0]]
                gen_img = np.clip(gen_img, 0, 255)
                gen_img = gen_img.astype(np.uint8)

                row0 = np.concatenate([gray_img, trg_img], 1)
                row1 = np.concatenate([src_img, gen_img], 1)
                result_img = np.concatenate([row0, row1], 0)
                result_img = result_img[:, :, [2, 1, 0]]  # BGR->RGB  ppgan是将RGB格式的图片进行保存的。

        self.visual_items['reference'] = result_img
        # self.nets_ema['generator'].train()
        # self.nets_ema['style_encoder'].train()
