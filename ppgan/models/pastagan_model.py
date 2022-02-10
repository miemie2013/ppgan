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
        pl_batch_shrink=2,
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
        self.lambda_cyc = lambda_cyc

        # self.nets['generator'].apply(he_init)
        # self.nets['style_encoder'].apply(he_init)
        # self.nets['mapping_network'].apply(he_init)
        # self.nets['discriminator'].apply(he_init)

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
        self.pl_batch_shrink = pl_batch_shrink

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

        # dic2 = np.load('../data77.npz')


        real_c, cat_feats = self.nets['style_encoding'](style_input, retain)
        gen_c = real_c  # 把 real_c 也当做 gen_c作为CGAN的C

        loss_numpy = {}

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

            loss_Gmain = paddle.nn.functional.softplus(-gen_logits)  # -log(sigmoid(gen_logits))
            loss_Gmain = loss_Gmain.mean()

            loss_Gmain_finetune = paddle.nn.functional.softplus(-gen_finetune_logits)  # -log(sigmoid(gen_logits))
            loss_Gmain_finetune = loss_Gmain_finetune.mean()

            loss_numpy['loss_Gmain'] = loss_Gmain.numpy()
            loss_numpy['loss_Gmain_finetune'] = loss_Gmain_finetune.numpy()

            # l1 loss
            loss_G_L1 = 0
            loss_G_finetune_L1 = 0
            if self.l1_weight > 0:
                loss_G_L1 = paddle.nn.L1Loss()(gen_img, real_img) * self.l1_weight
                loss_G_finetune_L1 = paddle.nn.L1Loss()(gen_finetune_img, real_img) * self.l1_weight
                loss_numpy['loss_G_L1'] = loss_G_L1.numpy()
                loss_numpy['loss_G_finetune_L1'] = loss_G_finetune_L1.numpy()

            loss_mask = 0
            if self.mask_weight > 0:
                aaaaaaaaaaaaa = paddle.cast(gt_parsing, dtype=paddle.int64)[:, 0, :, :]
                loss_mask = self.ce_parsing(pred_parsing.transpose((0, 2, 3, 1)), aaaaaaaaaaaaa)
                loss_mask = paddle.mean(loss_mask) * self.mask_weight
                loss_numpy['loss_mask'] = loss_mask.numpy()


            # vgg loss
            loss_G_VGG = 0
            loss_G_finetune_VGG = 0
            if self.vgg_weight > 0:
                loss_G_VGG = self.criterionVGG(gen_img, real_img) * self.vgg_weight
                loss_G_VGG = loss_G_VGG.mean()

                loss_G_finetune_VGG = self.criterionVGG(gen_finetune_img, real_img) * self.vgg_weight
                loss_G_finetune_VGG = loss_G_finetune_VGG.mean()

                loss_numpy['loss_G_VGG'] = loss_G_VGG.numpy()
                loss_numpy['loss_G_finetune_VGG'] = loss_G_finetune_VGG.numpy()

            loss_G = (loss_Gmain + loss_Gmain_finetune) / 2 + \
                     (loss_G_L1 + loss_G_finetune_L1) / 2 + \
                     (loss_G_VGG + loss_G_finetune_VGG) / 2 + loss_mask


            loss_G = loss_G * float(gain)
            loss_G.backward()  # 咩酱：gain即上文提到的这个阶段的训练间隔。

        # Gpl: Apply path length regularization.
        if do_Gpl:
            batch_size = gen_z.shape[0] // self.pl_batch_shrink
            # with misc.ddp_sync(self.G_flownet, sync):
            #     flow = self.G_flownet(torch.cat((cloth[:batch_size], aff_pose[:batch_size]), dim=1))
            # warp_cloth = F.grid_sample(cloth[:batch_size, :3, :, :], flow)

            gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], pose[:batch_size],
                                         [cat_feat[:batch_size] for cat_feat in cat_feats], sync=sync)
            pl_noise = paddle.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
            pl_grads = paddle.grad(
                outputs=[(gen_img * pl_noise).sum()],
                inputs=[gen_ws],
                create_graph=True,  # 最终loss里包含梯度，需要求梯度的梯度，所以肯定需要建立反向图。
                retain_graph=True)[0]
            pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
            pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
            self.pl_mean.copy_(pl_mean.detach())
            pl_penalty = (pl_lengths - pl_mean).square()
            loss_Gpl = pl_penalty * self.pl_weight
            # loss_numpy['loss_Gpl'] = loss_Gpl.numpy()

            loss_Gpl = (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean() * float(gain)
            loss_Gpl.backward()  # 咩酱：gain即上文提到的这个阶段的训练间隔。

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        loss3 = 0.0
        if do_Dmain:
            gen_img, gen_finetune_img, _, _gen_ws = self.run_G(gen_z, gen_c, pose, cat_feats, denorm_upper_mask,
                                                               denorm_lower_mask, \
                                                               denorm_upper_input, denorm_lower_input, sync=False)
            gen_logits = self.run_D(gen_img, gen_c, sync=False)  # Gets synced by loss_Dreal.
            gen_finetune_logits = self.run_D(gen_finetune_img, gen_c, sync=False)

            loss_Dgen = paddle.nn.functional.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))

            loss_Dgen_finetune = paddle.nn.functional.softplus(gen_finetune_logits)  # -log(1 - sigmoid(gen_logits))

            loss_Dgen = loss_Dgen.mean()
            loss_Dgen_finetune = loss_Dgen_finetune.mean()

            loss_numpy['loss_Dgen'] = loss_Dgen.numpy()
            loss_numpy['loss_Dgen_finetune'] = loss_Dgen_finetune.numpy()

            loss3 = ((loss_Dgen + loss_Dgen_finetune) / 2) * float(gain)

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'

            real_img_tmp = real_img.detach()
            real_img_tmp.stop_gradient = not do_Dr1
            real_logits = self.run_D(real_img_tmp, real_c, sync=sync)

            loss_Dreal = 0
            if do_Dmain:
                loss_Dreal = paddle.nn.functional.softplus(-real_logits)  # -log(sigmoid(real_logits))
                loss_numpy['loss_Dreal'] = loss_Dreal.numpy().mean()

            loss_Dr1 = 0
            if do_Dr1:
                r1_grads = paddle.grad(
                    outputs=[real_logits.sum()],
                    inputs=[real_img_tmp],
                    create_graph=True,  # 最终loss里包含梯度，需要求梯度的梯度，所以肯定需要建立反向图。
                    retain_graph=True)[0]

                # r1_grads = paddle.grad(outputs=real_logits.sum(),
                #                        inputs=real_img_tmp,
                #                        create_graph=True)[0]  # 最终loss里包含梯度，需要求梯度的梯度，所以肯定需要建立反向图。

                r1_penalty = r1_grads.square().sum([1, 2, 3])
                loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                loss_numpy['loss_Dr1'] = loss_Dr1.numpy().mean()

            loss4 = (loss_Dreal + loss_Dr1).mean() * float(gain)
            if do_Dmain:
                loss4 += loss3
            loss4.backward()  # 咩酱：gain即上文提到的这个阶段的训练间隔。
        return loss_numpy

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

        # miemie2013: 调试的代码
        # dic2 = np.load('../train_data.npz')
        # phase_real = paddle.to_tensor(dic2['phase_real'], dtype=phase_real.dtype)
        # phase_pose = paddle.to_tensor(dic2['phase_pose'], dtype=phase_pose.dtype)
        # phase_norm_img = paddle.to_tensor(dic2['phase_norm_img'], dtype=phase_norm_img.dtype)
        # phase_norm_img_lower = paddle.to_tensor(dic2['phase_norm_img_lower'], dtype=phase_norm_img_lower.dtype)
        # phase_denorm_upper_img = paddle.to_tensor(dic2['phase_denorm_upper_img'], dtype=phase_denorm_upper_img.dtype)
        # phase_denorm_lower_img = paddle.to_tensor(dic2['phase_denorm_lower_img'], dtype=phase_denorm_lower_img.dtype)
        # phase_gt_parsing = paddle.to_tensor(dic2['phase_gt_parsing'], dtype=phase_gt_parsing.dtype)
        # phase_denorm_upper_mask = paddle.to_tensor(dic2['phase_denorm_upper_mask'], dtype=phase_denorm_upper_mask.dtype)
        # phase_denorm_lower_mask = paddle.to_tensor(dic2['phase_denorm_lower_mask'], dtype=phase_denorm_lower_mask.dtype)
        # phase_retain_mask = paddle.to_tensor(dic2['phase_retain_mask'], dtype=phase_retain_mask.dtype)

        # phase_real2 = paddle.to_tensor(dic2['phase_real'], dtype=phase_real.dtype)
        # phase_pose2 = paddle.to_tensor(dic2['phase_pose'], dtype=phase_pose.dtype)
        # phase_norm_img2 = paddle.to_tensor(dic2['phase_norm_img'], dtype=phase_norm_img.dtype)
        # phase_norm_img_lower2 = paddle.to_tensor(dic2['phase_norm_img_lower'], dtype=phase_norm_img_lower.dtype)
        # phase_denorm_upper_img2 = paddle.to_tensor(dic2['phase_denorm_upper_img'], dtype=phase_denorm_upper_img.dtype)
        # phase_denorm_lower_img2 = paddle.to_tensor(dic2['phase_denorm_lower_img'], dtype=phase_denorm_lower_img.dtype)
        # phase_gt_parsing2 = paddle.to_tensor(dic2['phase_gt_parsing'], dtype=phase_gt_parsing.dtype)
        # phase_denorm_upper_mask2 = paddle.to_tensor(dic2['phase_denorm_upper_mask'], dtype=phase_denorm_upper_mask.dtype)
        # phase_denorm_lower_mask2 = paddle.to_tensor(dic2['phase_denorm_lower_mask'], dtype=phase_denorm_lower_mask.dtype)
        # phase_retain_mask2 = paddle.to_tensor(dic2['phase_retain_mask'], dtype=phase_retain_mask.dtype)


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
        loss_numpys = []
        loss_phase_name = []
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
                loss_numpy = self.accumulate_gradients(phase=phase['name'], real_img=real_img, gen_z=gen_z, style_input=style_input,
                                          retain=retain, pose=pose, denorm_upper_input=denorm_upper_input,
                                          denorm_lower_input=denorm_lower_input, denorm_upper_mask=denorm_upper_mask,
                                          denorm_lower_mask=denorm_lower_mask, gt_parsing=gt_parsing, sync=sync, gain=gain)
                loss_numpys.append(loss_numpy)
                loss_phase_name.append(phase['name'])

            # Update weights.
            # phase.module.requires_grad_(False)
            # 梯度裁剪
            # for param in phase.module.parameters():
            #     if param.grad is not None:
            #         misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            if 'G' in phase['name']:
                optimizers['generator'].step()  # 更新参数
            elif 'D' in phase['name']:
                optimizers['discriminator'].step()  # 更新参数

        # compute moving average of network parameters。指数滑动平均
        soft_update(self.nets['synthesis'],
                    self.nets_ema['synthesis'],
                    beta=0.999)
        soft_update(self.nets['mapping'],
                    self.nets_ema['mapping'],
                    beta=0.999)
        soft_update(self.nets['const_encoding'],
                    self.nets_ema['const_encoding'],
                    beta=0.999)
        soft_update(self.nets['style_encoding'],
                    self.nets_ema['style_encoding'],
                    beta=0.999)

        for loss, prefix in zip(
            loss_numpys,
            loss_phase_name):
            for key, value in loss.items():
                self.losses[prefix + '_' + key] = value
        self.batch_idx += 1

    def test_iter(self, metrics=None):
        self.nets_ema['synthesis'].eval()
        self.nets_ema['mapping'].eval()
        self.nets_ema['const_encoding'].eval()
        self.nets_ema['style_encoding'].eval()

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
            version = self.nets['synthesis'].version
            if version == 'Full':
                gen_coarse_imgs, gen_imgs, _ = self.nets['synthesis'](ws,
                                                                         pose_feat, cat_feats, denorm_upper_clothes,
                                                                         denorm_lower_clothes, denorm_upper_mask,
                                                                         denorm_lower_mask)
            elif version == 'V18':
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
        self.nets_ema['synthesis'].train()
        self.nets_ema['mapping'].train()
        self.nets_ema['const_encoding'].train()
        self.nets_ema['style_encoding'].train()
