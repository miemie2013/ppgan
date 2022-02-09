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
import sys







class VGGLoss(nn.Layer):
    def __init__(self, ckpt_path, requires_grad=False, weights=[1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19_Feature(ckpt_path=ckpt_path, requires_grad=requires_grad)
        self.criterion = nn.L1Loss()
        self.weights = weights

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class VGG19_Feature(nn.Layer):
    def __init__(self, ckpt_path="", pretrained=True, requires_grad=False):
        super(VGG19_Feature, self).__init__()
        vgg_pretrained_features = vgg19(pretrained=pretrained, progress=False, ckpt_path=ckpt_path).features  # load from local file
        vgg_pretrained_features.eval()
        if not requires_grad:
            for param in vgg_pretrained_features.parameters():
                param.stop_gradient = True
        print('load vgg19 success!')

        self.slice1 = paddle.nn.Sequential()
        self.slice2 = paddle.nn.Sequential()
        self.slice3 = paddle.nn.Sequential()
        self.slice4 = paddle.nn.Sequential()
        self.slice5 = paddle.nn.Sequential()
        for x in range(2):
            self.slice1.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_sublayer(str(x), vgg_pretrained_features[x])

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGG(nn.Layer):
    def __init__(
        self,
        features: nn.Layer,
        num_classes: int = 1000,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2D((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        # if init_weights:
        #     self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2D):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2D):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm: bool = False) -> nn.Sequential:
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2D(kernel_size=2, stride=2)]
        else:
            # v = cast(int, v)
            conv2d = nn.Conv2D(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2D(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)

def vgg19(pretrained: bool = False, progress: bool = True, ckpt_path: str = "", **kwargs) -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, ckpt_path, **kwargs)

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, ckpt_path: str = "", **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        print("vgg19 ckpt_path from local: ", ckpt_path)
        state_dict = paddle.load(ckpt_path)
        model.set_state_dict(state_dict)
    return model



def vgg_preprocess(tensor, vgg_normal_correct=False):
    if vgg_normal_correct:
        tensor = (tensor + 1) / 2
    # input is RGB tensor which ranges in [0,1]
    # output is BGR tensor which ranges in [0,255]
    tensor_bgr = paddle.concat((tensor[:, 2:3, :, :], tensor[:, 1:2, :, :], tensor[:, 0:1, :, :]), axis=1)
    # tensor_bgr = tensor[:, [2, 1, 0], ...]
    tensor_bgr_ml = tensor_bgr - paddle.to_tensor([0.40760392, 0.45795686, 0.48501961]).type_as(tensor_bgr).reshape((1, 3, 1, 1))
    tensor_rst = tensor_bgr_ml * 255
    return tensor_rst

class VGG19_feature_color_torchversion(nn.Layer):
    '''
    NOTE: there is no need to pre-process the input
    input tensor should range in [0,1]
    '''

    def __init__(self, pool='max', vgg_normal_correct=True, ic=3):
        super(VGG19_feature_color_torchversion, self).__init__()
        self.vgg_normal_correct = vgg_normal_correct

        self.conv1_1 = nn.Conv2D(ic, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2D(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2D(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2D(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2D(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2D(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2D(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2D(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2D(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2D(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2D(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2D(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2D(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2D(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2D(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2D(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2D(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2D(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2D(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2D(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2D(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2D(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2D(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2D(kernel_size=2, stride=2)

    def forward(self, x, out_keys, preprocess=True):
        '''
        NOTE: input tensor should range in [0,1]
        '''
        out = {}
        if preprocess:
            x = vgg_preprocess(x, vgg_normal_correct=self.vgg_normal_correct)
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]


def feature_normalize(feature_in):
    feature_in_norm = paddle.norm(feature_in, 2, 1, keepdim=True) + sys.float_info.epsilon
    feature_in_norm = feature_in / feature_in_norm
    return feature_in_norm


class ContextualLoss_forward(nn.Layer):
    '''
        input is Al, Bl, channel = 1, range ~ [0, 255]
    '''

    def __init__(self, PONO=True):
        super(ContextualLoss_forward, self).__init__()
        self.PONO = PONO
        return None

    def forward(self, X_features, Y_features, h=0.1, feature_centering=True):
        '''
        X_features&Y_features are are feature vectors or feature 2d array
        h: bandwidth
        return the per-sample loss
        '''
        batch_size = X_features.shape[0]
        feature_depth = X_features.shape[1]
        feature_size = X_features.shape[2]

        # to normalized feature vectors
        if feature_centering:
            if self.PONO:
                X_features = X_features - Y_features.mean(dim=1).unsqueeze(dim=1)
                Y_features = Y_features - Y_features.mean(dim=1).unsqueeze(dim=1)
            else:
                X_features = X_features - Y_features.reshape((batch_size, feature_depth, -1)).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
                Y_features = Y_features - Y_features.reshape((batch_size, feature_depth, -1)).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        X_features = feature_normalize(X_features).reshape((batch_size, feature_depth, -1))  # batch_size * feature_depth * feature_size * feature_size
        Y_features = feature_normalize(Y_features).reshape((batch_size, feature_depth, -1))  # batch_size * feature_depth * feature_size * feature_size

        # X_features = F.unfold(
        #     X_features, kernel_size=self.opt.match_kernel, stride=1, padding=int(self.opt.match_kernel // 2))  # batch_size * feature_depth_new * feature_size^2
        # Y_features = F.unfold(
        #     Y_features, kernel_size=self.opt.match_kernel, stride=1, padding=int(self.opt.match_kernel // 2))  # batch_size * feature_depth_new * feature_size^2

        # conine distance = 1 - similarity
        X_features_permute = X_features.permute(0, 2, 1)  # batch_size * feature_size^2 * feature_depth
        d = 1 - paddle.matmul(X_features_permute, Y_features)  # batch_size * feature_size^2 * feature_size^2

        # normalized distance: dij_bar
        # d_norm = d
        d_norm = d / (torch.min(d, dim=-1, keepdim=True)[0] + 1e-3)  # batch_size * feature_size^2 * feature_size^2

        # pairwise affinity
        w = torch.exp((1 - d_norm) / h)
        A_ij = w / torch.sum(w, dim=-1, keepdim=True)

        # contextual loss per sample
        CX = torch.mean(torch.max(A_ij, dim=-1)[0], dim=1)
        loss = -paddle.log(CX)

        # contextual loss per batch
        # loss = torch.mean(loss)
        return loss










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
        G_reg_interval=4,
        D_reg_interval=16,
        latent_dim=16,
        lambda_reg=1,
        lambda_sty=1,
        lambda_ds=1,
        lambda_cyc=1,
        r1_gamma=10,
        l1_weight=50.0,
        vgg_weight=50.0,
        pl_weight=0.0,
        contextual_weight=1.0,
        mask_weight=1.0,
        style_mixing_prob=0.9,
        vgg19_ckpt1=None,
        vgg19_ckpt2=None,
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

        self.phases = []
        for name, reg_interval in [('G', G_reg_interval), ('D', D_reg_interval)]:
            if reg_interval is None:
                # opt = dnnlib.util.construct_class_by_name(params=module.parameters(),
                #                                           **opt_kwargs)  # subclass of torch.optim.Optimizer
                # phases += [dnnlib.EasyDict(name=name + 'both', module=module, opt=opt, interval=1)]
                pass
            else:  # Lazy regularization.
                self.phases += [dict(name=name + 'main', interval=1)]
                self.phases += [dict(name=name + 'reg', interval=reg_interval)]

        self.z_dim = self.nets['mapping'].z_dim
        self.batch_idx = 0

        # loss config.
        self.r1_gamma = r1_gamma
        self.l1_weight = l1_weight
        self.vgg_weight = vgg_weight
        self.pl_weight = pl_weight
        self.contextual_weight = contextual_weight
        self.mask_weight = mask_weight
        self.style_mixing_prob = style_mixing_prob
        self.vgg19_ckpt1 = vgg19_ckpt1
        self.vgg19_ckpt2 = vgg19_ckpt2

        # 每个类别的权重（6个类别）
        class_weight = paddle.to_tensor([1., 2., 2., 3., 3., 3.])
        self.ce_parsing = paddle.nn.CrossEntropyLoss(ignore_index=255, weight=class_weight)

        if self.vgg_weight > 0:
            self.criterionVGG = VGGLoss(ckpt_path=self.vgg19_ckpt1, requires_grad=False)

        if self.contextual_weight > 0:
            contextual_vgg_path = self.vgg19_ckpt2
            self.contextual_vgg = VGG19_feature_color_torchversion()
            self.contextual_vgg.set_state_dict(paddle.load(contextual_vgg_path))
            self.contextual_vgg.eval()
            for param in self.contextual_vgg.parameters():
                param.stop_gradient = True
            self.contextual_layers = ['r12','r22','r32','r42','r52']
            self.contextual_forward_loss = ContextualLoss_forward()

        self.augment_pipe = None


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

    def run_G(self, z, c, pose, const_feats, denorm_upper_mask, denorm_lower_mask, denorm_upper_input, denorm_lower_input, sync):
        cat_feats = {}
        for _, cat_feat in enumerate(const_feats):
            h, _ = cat_feat.shape[2], cat_feat.shape[3]
            cat_feats[str(h)] = cat_feat

        pose_feat = self.nets['const_encoding'](pose)

        ws = self.nets['mapping'](z, c)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]

        img, finetune_img, pred_parsing = self.nets['synthesis'](ws, pose_feat, cat_feats, denorm_upper_input, denorm_lower_input, denorm_upper_mask, denorm_lower_mask)

        return img, finetune_img, pred_parsing, ws

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)

        logits = self.nets['discriminator'](img, c)
        return logits

    # 梯度累加（变相增大批大小）。
    def accumulate_gradients(self, phase, real_img, gen_z, style_input, retain, pose, denorm_upper_input,
                             denorm_lower_input, denorm_upper_mask, denorm_lower_mask, gt_parsing, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1 = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        real_c, cat_feats = self.nets['style_encoding'](style_input, retain)
        gen_c = real_c  # 把 real_c 也当做 gen_c作为CGAN的C

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            gen_img, gen_finetune_img, pred_parsing, _gen_ws = self.run_G(gen_z, gen_c, pose, cat_feats,
                                                                          denorm_upper_mask, denorm_lower_mask, \
                                                                          denorm_upper_input, denorm_lower_input,
                                                                          sync=(sync and not do_Gpl))  # May get synced by Gpl.

            # 这里的conditioned GAN的 (gen_img, gen_c) 和 (real_img, real_c) 不是严格对应的。
            # 如果加入pose conditioned, 那么应该 gen_img和real_img严格对应，然后 只用一个real pose, 也就是(gen_img, real_pose) 和 (real_img, real_pose)
            # 视情况, 看是否需要加入L1 和 vgg loss

            gen_logits = self.run_D(gen_img, gen_c, sync=False)
            gen_finetune_logits = self.run_D(gen_finetune_img, gen_c, sync=False)

            training_stats.report('Loss/scores/fake', gen_logits)
            training_stats.report('Loss/signs/fake', gen_logits.sign())
            loss_Gmain = torch.nn.functional.softplus(-gen_logits)  # -log(sigmoid(gen_logits))
            loss_Gmain = loss_Gmain.mean()
            training_stats.report('Loss/G/loss', loss_Gmain)

            training_stats.report('Loss/scores/fake_finetune', gen_finetune_logits)
            training_stats.report('Loss/signs/fake_finetune', gen_finetune_logits.sign())
            loss_Gmain_finetune = torch.nn.functional.softplus(-gen_finetune_logits)  # -log(sigmoid(gen_logits))
            loss_Gmain_finetune = loss_Gmain_finetune.mean()
            training_stats.report('Loss/G/loss_finetune', loss_Gmain_finetune)

            # l1 loss
            loss_G_L1 = 0
            loss_G_finetune_L1 = 0
            if self.l1_weight > 0:
                loss_G_L1 = torch.nn.L1Loss()(gen_img, real_img) * self.l1_weight
                # loss_G_L1 = loss_G_L1.mean()

                loss_G_finetune_L1 = torch.nn.L1Loss()(gen_finetune_img, real_img) * self.l1_weight
                # loss_G_finetune_L1 = loss_G_finetune_L1.mean()
            training_stats.report('Loss/G/L1', loss_G_L1)
            training_stats.report('Loss/G/L1_finetune', loss_G_finetune_L1)

            loss_mask = 0
            if self.mask_weight > 0:
                aaaaaaaaaaaaa = paddle.cast(gt_parsing, dtype=paddle.int64)[:, 0, :, :]
                loss_mask = self.ce_parsing(pred_parsing.transpose((0, 2, 3, 1)), aaaaaaaaaaaaa)
                loss_mask = paddle.mean(loss_mask) * self.mask_weight

            training_stats.report('Loss/G/mask_loss', loss_mask)

            # vgg loss
            loss_G_VGG = 0
            loss_G_finetune_VGG = 0
            if self.vgg_weight > 0:
                loss_G_VGG = self.criterionVGG(gen_img, real_img) * self.vgg_weight
                loss_G_VGG = loss_G_VGG.mean()

                loss_G_finetune_VGG = self.criterionVGG(gen_finetune_img, real_img) * self.vgg_weight
                loss_G_finetune_VGG = loss_G_finetune_VGG.mean()
            training_stats.report('Loss/G/vgg', loss_G_VGG)
            training_stats.report('Loss/G/vgg_finetune', loss_G_finetune_VGG)

            loss_G = (loss_Gmain + loss_Gmain_finetune) / 2 + \
                     (loss_G_L1 + loss_G_finetune_L1) / 2 + \
                     (loss_G_VGG + loss_G_finetune_VGG) / 2 + loss_mask

            loss_G.mul(gain).backward()  # 咩酱：gain即上文提到的这个阶段的训练间隔。

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                # with misc.ddp_sync(self.G_flownet, sync):
                #     flow = self.G_flownet(torch.cat((cloth[:batch_size], aff_pose[:batch_size]), dim=1))
                # warp_cloth = F.grid_sample(cloth[:batch_size, :3, :, :], flow)

                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], pose[:batch_size],
                                             [cat_feat[:batch_size] for cat_feat in cat_feats], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()  # 咩酱：gain即上文提到的这个阶段的训练间隔。

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, gen_finetune_img, _, _gen_ws = self.run_G(gen_z, gen_c, pose, cat_feats, denorm_upper_mask,
                                                                   denorm_lower_mask, \
                                                                   denorm_upper_input, denorm_lower_input, sync=False)
                gen_logits = self.run_D(gen_img, gen_c, sync=False)  # Gets synced by loss_Dreal.
                gen_finetune_logits = self.run_D(gen_finetune_img, gen_c, sync=False)

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))

                training_stats.report('Loss/scores/fake_finetune', gen_finetune_logits)
                training_stats.report('Loss/signs/fake_finetune', gen_finetune_logits.sign())
                loss_Dgen_finetune = torch.nn.functional.softplus(gen_finetune_logits)  # -log(1 - sigmoid(gen_logits))

            with torch.autograd.profiler.record_function('Dgen_backward'):
                ((loss_Dgen.mean() + loss_Dgen_finetune.mean()) / 2).mul(gain).backward()  # 咩酱：gain即上文提到的这个阶段的训练间隔。
                # loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)  # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = \
                        torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True,
                                            only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()  # 咩酱：gain即上文提到的这个阶段的训练间隔。

    def train_iter(self, optimizers=None):
        phase_real = self.input[0]
        phase_pose = self.input[1]
        phase_norm_img = self.input[2]
        phase_norm_img_lower = self.input[3]
        phase_denorm_upper_img = self.input[4]
        phase_denorm_lower_img = self.input[5]
        phase_gt_parsing = self.input[7]
        phase_denorm_upper_mask = self.input[8]
        phase_denorm_lower_mask = self.input[9]
        phase_retain_mask = self.input[12]

        phase_real_tensor = paddle.cast(phase_real, dtype=paddle.float32) / 127.5 - 1
        phase_parts_tensor = paddle.cast(phase_norm_img, dtype=paddle.float32) / 127.5 - 1
        phase_parts_lower_tensor = paddle.cast(phase_norm_img_lower, dtype=paddle.float32) / 127.5 - 1
        phase_parts_tensor = paddle.concat([phase_parts_tensor, phase_parts_lower_tensor], 1)

        phase_denorm_upper_img_tensor = paddle.cast(phase_denorm_upper_img, dtype=paddle.float32) / 127.5 - 1
        phase_denorm_lower_img_tensor = paddle.cast(phase_denorm_lower_img, dtype=paddle.float32) / 127.5 - 1
        phase_denorm_upper_mask_tensor = paddle.cast(phase_denorm_upper_mask, dtype=paddle.float32)
        phase_denorm_lower_mask_tensor = paddle.cast(phase_denorm_lower_mask, dtype=paddle.float32)

        phase_pose_tensor = paddle.cast(phase_pose, dtype=paddle.float32) / 127.5 - 1
        phase_retain_mask = paddle.cast(phase_retain_mask, dtype=paddle.float32)
        phase_head_mask = phase_retain_mask
        phase_head_tensor = phase_head_mask * phase_real_tensor - (1 - phase_head_mask)
        phase_pose_tensor = paddle.concat([phase_pose_tensor, phase_head_tensor], 1)

        phase_gt_parsing_tensor = paddle.cast(phase_gt_parsing, dtype=paddle.float32)

        # process head
        phase_retain_tensor = phase_head_tensor

        phases = self.phases
        batch_size = phase_real_tensor.shape[0]

        all_gen_z = None
        num_gpus = 1  # 显卡数量
        batch_gpu = batch_size // num_gpus  # 一张显卡上的批大小
        if self.z_dim > 0:
            all_gen_z = paddle.randn([len(phases) * batch_size, self.z_dim])  # 咩酱：训练的4个阶段每个gpu的噪声
        else:
            all_gen_z = paddle.randn([len(phases) * batch_size, 1])  # 咩酱：训练的4个阶段每个gpu的噪声
        phases_all_gen_z = paddle.split(all_gen_z, num_or_sections=len(phases))  # 咩酱：训练的4个阶段的噪声
        all_gen_z = [paddle.split(phase_gen_z, num_or_sections=num_gpus) for phase_gen_z in phases_all_gen_z]  # 咩酱：训练的4个阶段每个gpu的噪声

        phase_real_tensor = paddle.split(phase_real_tensor, num_or_sections=num_gpus)
        phase_parts_tensor = paddle.split(phase_parts_tensor, num_or_sections=num_gpus)
        phase_pose_tensor = paddle.split(phase_pose_tensor, num_or_sections=num_gpus)
        phase_retain_tensor = paddle.split(phase_retain_tensor, num_or_sections=num_gpus)
        phase_denorm_upper_img_tensor = paddle.split(phase_denorm_upper_img_tensor, num_or_sections=num_gpus)
        phase_denorm_lower_img_tensor = paddle.split(phase_denorm_lower_img_tensor, num_or_sections=num_gpus)
        phase_gt_parsing_tensor = paddle.split(phase_gt_parsing_tensor, num_or_sections=num_gpus)
        phase_denorm_upper_mask_tensor = paddle.split(phase_denorm_upper_mask_tensor, num_or_sections=num_gpus)
        phase_denorm_lower_mask_tensor = paddle.split(phase_denorm_lower_mask_tensor, num_or_sections=num_gpus)

        del phase_real      # conserve memory
        del phase_pose       # conserve memory
        del phase_head_mask   # conserve memory
        del phase_gt_parsing  # conserve memory

        # Execute training phases.  咩酱：训练的4个阶段。一个批次的图片训练4个阶段。
        for phase, phase_gen_z in zip(phases, all_gen_z):  # 咩酱：phase_gen_z是这个阶段每个gpu的噪声，是一个元组，元组长度等于gpu数量。
            if self.batch_idx % phase['interval'] != 0:  # 咩酱：每一个阶段phase有一个属性interval，即训练间隔，每隔几个批次图片才会执行1次这个阶段！
                continue

            # Initialize gradient accumulation.  咩酱：初始化梯度累加（变相增大批大小）。
            self._reset_grad(optimizers)  # 梯度清0
            # phase.module.requires_grad_(True)      # 网络参数需要梯度

            # 梯度累加。一个总的批次的图片分开{显卡数量}次遍历。
            # Accumulate gradients over multiple rounds.  咩酱：遍历每一个gpu上的批次图片。这样写好奇葩啊！round_idx是gpu_id
            for round_idx, (real_img, gen_z, style_input, retain, pose, denorm_upper_input, denorm_lower_input, \
                            denorm_upper_mask, denorm_lower_mask, gt_parsing) \
                    in enumerate(zip(phase_real_tensor, phase_gen_z, phase_parts_tensor, \
                                     phase_retain_tensor, phase_pose_tensor, phase_denorm_upper_img_tensor,\
                                     phase_denorm_lower_img_tensor, phase_denorm_upper_mask_tensor, \
                                     phase_denorm_lower_mask_tensor, phase_gt_parsing_tensor)):
                sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)   # 咩酱：右边的式子结果一定是0。即只有0号gpu做同步。这是梯度累加的固定写法。
                gain = phase['interval']     # 咩酱：即上文提到的训练间隔。
                # 把style_input当做 real_c 和 gen_c。为了增加可变性, gen_z还是保留

                # 梯度累加（变相增大批大小）。
                self.accumulate_gradients(phase=phase['name'], real_img=real_img, gen_z=gen_z, style_input=style_input,
                                          retain=retain, pose=pose, denorm_upper_input=denorm_upper_input,
                                          denorm_lower_input=denorm_lower_input, denorm_upper_mask=denorm_upper_mask,
                                          denorm_lower_mask=denorm_lower_mask, gt_parsing=gt_parsing, sync=sync, gain=gain)

            # Update weights.
            # phase.module.requires_grad_(False)
            # 梯度裁剪
            # for param in phase.module.parameters():
            #     if param.grad is not None:
            #         misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            phase.opt.step()

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
        self.batch_idx += 1

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
