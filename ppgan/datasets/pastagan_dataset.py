# code was heavily based on https://github.com/clovaai/stargan-v2
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/clovaai/stargan-v2#license
import PIL
import cv2
import json
import math
from skimage.draw import ellipse
import pycocotools.mask as maskUtils
import paddle
from .base_dataset import BaseDataset
from .builder import DATASETS
import os
from itertools import chain
from pathlib import Path
import traceback
import random
import numpy as np
from PIL import Image

from paddle.io import Dataset, WeightedRandomSampler

kptcolors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85],[255, 0, 0]]

limbseq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
           [1, 16], [16, 18], [3, 17], [6, 18]]


@DATASETS.register()
class PastaGANDataset(BaseDataset):
    """
    """
    def __init__(self, dataroot, txt_name, is_train, preprocess, test_count=0, resolution=None,
                 max_size=None, use_labels=False, xflip=False, random_seed=0):
        """Initialize single dataset class.

        Args:
            dataroot (str): Directory of dataset.
            preprocess (list[dict]): A sequence of data preprocess config.
        """
        super(PastaGANDataset, self).__init__(preprocess)
        self.test_count = test_count

        self.dataroot = dataroot
        self.txt_name = txt_name
        self.is_train = is_train

        self.image_fnames = []     # 人物图片的路径（txt注解文件第0列）
        self.kpt_fnames = []       # 人物关键点文件的路径（txt注解文件第0列）
        self.parsing_fnames = []   # 人物语义分割图的路径（txt注解文件第0列）

        if not self.is_train:  # 测试集才有。
            self.clothes_image_fnames = []     # 衣服图片的路径（txt注解文件第1列）
            self.clothes_kpt_fnames = []       # 衣服关键点文件的路径（txt注解文件第1列）
            self.clothes_parsing_fnames = []   # 衣服语义分割图的路径（txt注解文件第1列）

        txt_path = os.path.join(self.dataroot, txt_name)
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                person, clothes = line.strip().split()

                # 人物（txt注解文件第0列）
                self.image_fnames.append(os.path.join(self.dataroot, 'image', person))
                self.kpt_fnames.append(
                    os.path.join(self.dataroot, 'keypoints', person.replace('.jpg', '_keypoints.json')))
                self.parsing_fnames.append(
                    os.path.join(self.dataroot, 'parsing', person.replace('.jpg', '_label.png')))

                if not self.is_train:
                    # 衣服（txt注解文件第1列）
                    self.clothes_image_fnames.append(os.path.join(self.dataroot, 'image', clothes))
                    self.clothes_kpt_fnames.append(
                        os.path.join(self.dataroot, 'keypoints', clothes.replace('.jpg', '_keypoints.json')))
                    self.clothes_parsing_fnames.append(
                        os.path.join(self.dataroot, 'parsing', clothes.replace('.jpg', '_label.png')))

        if self.is_train:
            # 假设屎山的数据集文件夹路径有'/Zalando'
            assert '/Zalando' in self.dataroot
            dataroot2 = self.dataroot.split('/Zalando')[0]
            vis_dir = os.path.join(dataroot2, 'train_img_vis')  # train_img_vis文件夹里的图片来自训练集image文件夹里的部分图片。
            image_list = sorted(os.listdir(vis_dir))
            vis_index = []
            for image_name in image_list:
                real_path = os.path.join(self.dataroot, 'image', image_name)
                if os.path.exists(real_path):
                    if real_path in self.image_fnames:  # 确保是txt注解文件的第0列的图片。
                        vis_index.append(self.image_fnames.index(real_path))

            self.vis_index = vis_index

            random_mask_acgpn_dir = os.path.join(dataroot2, 'train_random_mask_acgpn')
            # 所有随机掩码图片的路径。
            self._random_mask_acgpn_fnames = [os.path.join(random_mask_acgpn_dir, mask_name) for mask_name in os.listdir(random_mask_acgpn_dir)]
            # 随机掩码图片的数量。12000
            self._mask_acgpn_numbers = len(self._random_mask_acgpn_fnames)
        elif not self.is_train:
            self.vis_index = list(range(64))  # vis_index

        PIL.Image.init()
        if len(self.image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self.dataroot))[0]
        im_shape = list((self.load_image(0))[0].shape)
        raw_shape = [len(self.image_fnames)] + [im_shape[2], im_shape[0], im_shape[1]]
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')

        # 父类
        # super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)
        self.name = name
        self.raw_shape = list(raw_shape)
        self.use_labels = use_labels
        self.raw_labels = None
        self.label_shape = None

        # Apply max_size.
        self.raw_idx = np.arange(self.raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self.raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self.raw_idx)
            self.raw_idx = np.sort(self.raw_idx[:max_size])

        # Apply xflip.
        self.xflip = np.zeros(self.raw_idx.size, dtype=np.uint8)
        if xflip:
            self.raw_idx = np.tile(self.raw_idx, 2)
            self.xflip = np.concatenate([self.xflip, np.ones_like(self.xflip)])

    def valid_joints(self, joint):
        return (joint >= 0.1).all()

    def get_crop(self, keypoints, bpart, order, wh, o_w, o_h, ar=1.0):
        '''
        bpart是 遍历10次，每次要用到的关键点名称
        order是 18个关键点，各个关键点指示的位置的名称
        '''
        joints = keypoints
        bpart_indices = [order.index(b) for b in bpart]  # 根据关键点名称获得关键点的下标。所以要求图片人物一定要有18个关键点。
        part_src = np.float32(joints[bpart_indices][:, :2])  # bpart指示的关键点的xy坐标
        # 有的关键点的置信度低于0.1时，执行if下的代码块。都大于等于0.1时不用执行。
        if not self.valid_joints(joints[bpart_indices][:, 2]):
            if self.is_train:
                if bpart[0] == "lhip" and bpart[1] == "lknee":
                    bpart = ["lhip"]
                    bpart_indices = [order.index(b) for b in bpart]
                    part_src = np.float32(joints[bpart_indices][:, :2])
                elif bpart[0] == "rhip" and bpart[1] == "rknee":
                    bpart = ["rhip"]
                    bpart_indices = [order.index(b) for b in bpart]
                    part_src = np.float32(joints[bpart_indices][:, :2])
                elif bpart[0] == "lshoulder" and bpart[1] == "rshoulder" and bpart[2] == "cnose":
                    bpart = ["lshoulder", "rshoulder", "rshoulder"]
                    bpart_indices = [order.index(b) for b in bpart]
                    part_src = np.float32(joints[bpart_indices][:, :2])
            else:
                if bpart[0] == "lhip" and bpart[1] == "lknee":
                    bpart = ["lhip"]
                    bpart_indices = [order.index(b) for b in bpart]
                    part_src = np.float32(joints[bpart_indices][:, :2])
                elif bpart[0] == "rhip" and bpart[1] == "rknee":
                    bpart = ["rhip"]
                    bpart_indices = [order.index(b) for b in bpart]
                    part_src = np.float32(joints[bpart_indices][:, :2])
                elif bpart[0] == "lknee" and bpart[1] == "lankle":
                    bpart = ["lknee"]
                    bpart_indices = [order.index(b) for b in bpart]
                    part_src = np.float32(joints[bpart_indices][:, :2])
                elif bpart[0] == "rknee" and bpart[1] == "rankle":
                    bpart = ["rknee"]
                    bpart_indices = [order.index(b) for b in bpart]
                    part_src = np.float32(joints[bpart_indices][:, :2])
                elif bpart[0] == "lshoulder" and bpart[1] == "rshoulder" and bpart[2] == "cnose":
                    bpart = ["lshoulder", "rshoulder", "rshoulder"]
                    bpart_indices = [order.index(b) for b in bpart]
                    part_src = np.float32(joints[bpart_indices][:, :2])

        # 有的关键点的置信度低于0.1时，执行if下的代码块。都大于等于0.1时不用执行。
        if not self.valid_joints(joints[bpart_indices][:, 2]):
            return None, None

        # 当图片人物有18个关键点时，肯定不会执行if下的代码块。跳过不理。
        if part_src.shape[0] == 1:
            # leg fallback
            a = part_src[0]
            b = np.float32([a[0], o_h - 1])
            part_src = np.float32([a, b])

        if part_src.shape[0] == 4:
            pass
        elif part_src.shape[0] == 3:
            # lshoulder, rshoulder, cnose
            if bpart == ["lshoulder", "rshoulder", "rshoulder"]:
                segment = part_src[1] - part_src[0]
                normal = np.array([-segment[1], segment[0]])
                if normal[1] > 0.0:
                    normal = -normal

                a = part_src[0] + normal
                b = part_src[0]
                c = part_src[1]
                d = part_src[1] + normal
                part_src = np.float32([a, b, c, d])
            else:
                assert bpart == ["lshoulder", "rshoulder", "cnose"]
                neck = 0.5*(part_src[0] + part_src[1])
                neck_to_nose = part_src[2] - neck
                part_src = np.float32([neck + 2*neck_to_nose, neck])

                # segment box
                segment = part_src[1] - part_src[0]
                normal = np.array([-segment[1], segment[0]])
                alpha = 1.0 / 2.0
                a = part_src[0] + alpha*normal
                b = part_src[0] - alpha*normal
                c = part_src[1] - alpha*normal
                d = part_src[1] + alpha*normal
                part_src = np.float32([b, c, d, a])
        else:
            assert part_src.shape[0] == 2

            segment = part_src[1] - part_src[0]
            normal = np.array([-segment[1], segment[0]])
            alpha = ar / 2.0
            a = part_src[0] + alpha*normal
            b = part_src[0] - alpha*normal
            c = part_src[1] - alpha*normal
            d = part_src[1] + alpha*normal
            part_src = np.float32([a, b, c, d])

        dst = np.float32([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])  # shape=[4, 2]
        part_dst = np.float32(wh * dst)                             # shape=[4, 2]

        # 透视变换?
        M = cv2.getPerspectiveTransform(part_src, part_dst)
        M_inv = cv2.getPerspectiveTransform(part_dst, part_src)
        return M, M_inv

    def normalize(self, upper_img, lower_img, upper_clothes_mask, lower_clothes_mask,
                  upper_pose=None, lower_pose=None, upper_keypoints=None, lower_keypoints=None, box_factor=2):

        '''
        训练时：
        upper_img            [256, 256, 3]    试穿者上衣的图片
        lower_img            [256, 256, 3]    试穿者下装（裤子、裙子等）的图片
        upper_clothes_mask   [256, 256, 3]    试穿者上衣的掩码*255
        lower_clothes_mask   [256, 256, 3]    试穿者下装（裤子、裙子等）的掩码*255
        lower_keypoints      [18, 3]    试穿者关键点
        Return:
        norm_img             xxxxxxxxxxx
        norm_pose            xxxxxxxxxxx

        预测时：
        upper_img            [256, 256, 3]    被扒者上衣的图片
        lower_img            [256, 256, 3]    试穿者下装（裤子、裙子等）的图片
        upper_clothes_mask   [256, 256, 3]    被扒者上衣的掩码
        lower_clothes_mask   [256, 256, 3]    试穿者下装（裤子、裙子等）的掩码
        upper_pose           [256, 256, 3]    被扒者骨骼图
        lower_pose           [256, 256, 3]    试穿者骨骼图
        upper_keypoints      [18, 3]    被扒者关键点
        lower_keypoints      [18, 3]    试穿者关键点
        box_factor           int        图片宽高缩小为原来的1/(2^box_factor)
        Return:
        norm_img             xxxxxxxxxxx
        norm_pose            xxxxxxxxxxx
        denorm_upper_img     xxxxxxxxxxx
        denorm_lower_img     xxxxxxxxxxx
        '''

        h, w = upper_img.shape[:2]
        o_h, o_w = h, w
        h = h // 2 ** box_factor
        w = w // 2 ** box_factor
        wh = np.array([w, h])
        wh = np.expand_dims(wh, 0)

        # 遍历10次，每次要用到的关键点名称
        bparts = [
            ["lshoulder", "lhip", "rhip", "rshoulder"],
            ["lshoulder", "rshoulder", "cnose"],
            ["lshoulder", "lelbow"],
            ["lelbow", "lwrist"],
            ["rshoulder", "relbow"],
            ["relbow", "rwrist"],
            ["lhip", "lknee"],
            ["lknee", "lankle"],
            ["rhip", "rknee"],
            ["rknee", "rankle"]]

        # 18个关键点，各个关键点指示的位置的名称
        order = ['cnose', 'cneck', 'rshoulder', 'relbow', 'rwrist', 'lshoulder',
                 'lelbow', 'lwrist', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee',
                 'lankle', 'reye', 'leye', 'rear', 'lear']
        ar = 0.5

        part_imgs = list()
        if self.is_train:
            part_imgs_lower = list()
            part_clothes_masks = list()
            part_clothes_masks_lower = list()
            M_invs = list()
            denorm_hand_masks = list()

            denorm_upper_img = np.zeros_like(upper_img)
            denorm_lower_img = np.zeros_like(upper_img)

            for ii, bpart in enumerate(bparts):
                part_img = np.zeros((h, w, 3)).astype(np.uint8)
                part_img_lower = np.zeros((h, w, 3)).astype(np.uint8)
                part_clothes_mask = np.zeros((h, w, 3)).astype(np.uint8)
                part_clothes_mask_lower = np.zeros((h, w, 3)).astype(np.uint8)
                M, M_inv = self.get_crop(lower_keypoints, bpart, order, wh, o_w, o_h, ar)  # 根据 试穿者关键点 [18, 3], aaa

                if M is not None:
                    part_img = cv2.warpPerspective(upper_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                    part_clothes_mask = cv2.warpPerspective(upper_clothes_mask, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

                    denorm_patch = cv2.warpPerspective(part_img, M_inv, (o_w, o_h), borderMode=cv2.BORDER_CONSTANT)
                    denorm_clothes_mask_patch = cv2.warpPerspective(part_clothes_mask, M_inv, (o_w, o_h), borderMode=cv2.BORDER_CONSTANT)[..., 0:1]
                    denorm_clothes_mask_patch = (denorm_clothes_mask_patch == 255).astype(np.uint8)

                    denorm_upper_img = denorm_patch * denorm_clothes_mask_patch + denorm_upper_img * (1-denorm_clothes_mask_patch)

                    if ii >= 6:
                        part_img_lower = cv2.warpPerspective(lower_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                        part_clothes_mask_lower = cv2.warpPerspective(lower_clothes_mask, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

                        denorm_patch_lower = cv2.warpPerspective(part_img_lower, M_inv, (o_w, o_h), borderMode=cv2.BORDER_CONSTANT)
                        denorm_clothes_mask_patch_lower = cv2.warpPerspective(part_clothes_mask_lower, M_inv, (o_w, o_h), borderMode=cv2.BORDER_CONSTANT)[..., 0:1]
                        denorm_clothes_mask_patch_lower = (denorm_clothes_mask_patch_lower == 255).astype(np.uint8)

                        denorm_lower_img = denorm_patch_lower * denorm_clothes_mask_patch_lower + denorm_lower_img * (1 - denorm_clothes_mask_patch_lower)

                    M_invs.append(M_inv[np.newaxis, ...])
                else:
                    M_invs.append(np.zeros((1, 3, 3), dtype=np.float32))

                if ii >= 2 and ii <= 5:
                    if M is not None:
                        denorm_hand_masks.append(denorm_clothes_mask_patch)
                    else:
                        denorm_hand_masks.append(np.zeros_like(upper_img)[..., 0:1])
                # if ii == 7 or ii ==9:
                #     if M is not None:
                #         denorm_leg_masks.append(denorm_clothes_mask_patch)
                #     else:
                #         denorm_leg_masks.append(np.zeros_like(lower_img)[..., 0:1])

                part_imgs.append(part_img)
                part_clothes_masks.append(part_clothes_mask)
                if ii >= 6:
                    part_imgs_lower.append(part_img_lower)
                    part_clothes_masks_lower.append(part_clothes_mask_lower)

            img = np.concatenate(part_imgs, axis=2)
            img_lower = np.concatenate(part_imgs_lower, axis=2)
            clothes_masks = np.concatenate(part_clothes_masks, axis=2)
            clothes_masks_lower = np.concatenate(part_clothes_masks_lower, axis=2)
            M_invs = np.concatenate(M_invs, axis=0)

            return img, img_lower, denorm_upper_img, denorm_lower_img, M_invs, denorm_hand_masks, clothes_masks, clothes_masks_lower
        else:
            part_stickmen = list()

            denorm_upper_img = np.zeros_like(upper_img)   # [256, 256, 3]
            denorm_lower_img = np.zeros_like(upper_img)   # [256, 256, 3]
            kernel = np.ones((5, 5), np.uint8)

            for ii, bpart in enumerate(bparts):
                part_img = np.zeros((h, w, 3)).astype(np.uint8)
                part_stickman = np.zeros((h, w, 3)).astype(np.uint8)
                part_clothes_mask = np.zeros((h, w, 3)).astype(np.uint8)

                upper_M, _ = self.get_crop(upper_keypoints, bpart, order, wh, o_w, o_h, ar)
                lower_M, lower_M_inv = self.get_crop(lower_keypoints, bpart, order, wh, o_w, o_h, ar)

                if ii < 6:
                    if upper_M is not None:
                        part_img = cv2.warpPerspective(upper_img, upper_M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                        part_stickman = cv2.warpPerspective(upper_pose, upper_M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                        part_clothes_mask = cv2.warpPerspective(upper_clothes_mask, upper_M, (w, h),
                                                                borderMode=cv2.BORDER_REPLICATE)
                else:
                    if lower_M is not None:
                        part_img = cv2.warpPerspective(lower_img, lower_M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                        part_stickman = cv2.warpPerspective(lower_pose, lower_M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                        part_clothes_mask = cv2.warpPerspective(lower_clothes_mask, lower_M, (w, h),
                                                                borderMode=cv2.BORDER_REPLICATE)

                if lower_M_inv is not None:
                    denorm_patch = cv2.warpPerspective(part_img, lower_M_inv, (o_w, o_h), borderMode=cv2.BORDER_CONSTANT)
                    denorm_clothes_mask_patch = cv2.warpPerspective(part_clothes_mask, lower_M_inv, (o_w, o_h),
                                                                    borderMode=cv2.BORDER_CONSTANT)
                    if ii < 6:
                        denorm_clothes_mask_patch = cv2.erode(denorm_clothes_mask_patch, kernel, iterations=1)
                    denorm_clothes_mask_patch = denorm_clothes_mask_patch[..., 0:1]
                    denorm_clothes_mask_patch = (denorm_clothes_mask_patch == 255).astype(np.uint8)

                    if ii < 6:
                        denorm_upper_img = denorm_patch * denorm_clothes_mask_patch + denorm_upper_img * (
                                    1 - denorm_clothes_mask_patch)
                    else:
                        denorm_lower_img = denorm_patch * denorm_clothes_mask_patch + denorm_lower_img * (
                                    1 - denorm_clothes_mask_patch)

                part_imgs.append(part_img)
                part_stickmen.append(part_stickman)

            img = np.concatenate(part_imgs, axis=2)
            stickman = np.concatenate(part_stickmen, axis=2)

            return img, stickman, denorm_upper_img, denorm_lower_img

    def draw_pose_from_cords(self, pose_joints, img_size, affine_matrix=None,
                             coeffs=None, radius=2, draw_joints=True):
        colors = np.zeros(shape=img_size + (3,), dtype=np.uint8)
        # mask = np.zeros(shape=img_size, dtype=np.uint8)
        if draw_joints:
            for i, p in enumerate(limbseq):
                f, t = p[0] - 1, p[1] - 1
                from_missing = pose_joints[f][2] < 0.1  # 咩酱：关键点的置信度太低时，跳过画骨骼线
                to_missing = pose_joints[t][2] < 0.1    # 咩酱：关键点的置信度太低时，跳过画骨骼线
                if from_missing or to_missing:
                    continue
                if not affine_matrix is None:
                    pf = np.dot(affine_matrix, np.matrix([pose_joints[f][0], pose_joints[f][1], 1]).reshape(3, 1))
                    pt = np.dot(affine_matrix, np.matrix([pose_joints[t][0], pose_joints[t][1], 1]).reshape(3, 1))
                else:
                    pf = pose_joints[f][0], pose_joints[f][1]
                    pt = pose_joints[t][0], pose_joints[t][1]
                fx, fy = pf[1], pf[0]  # max(pf[1], 0), max(pf[0], 0)
                tx, ty = pt[1], pt[0]  # max(pt[1], 0), max(pt[0], 0)
                fx, fy = int(fx), int(fy)  # int(min(fx, 255)), int(min(fy, 191))
                tx, ty = int(tx), int(ty)  # int(min(tx, 255)), int(min(ty, 191))
                # xx, yy, val = line_aa(fx, fy, tx, ty)
                # colors[xx, yy] = np.expand_dims(val, 1) * kptcolors[i] # 255
                cv2.line(colors, (fy, fx), (ty, tx), kptcolors[i], 2)
                # mask[xx, yy] = 255

        for i, joint in enumerate(pose_joints):
            if pose_joints[i][2] < 0.1:  # 咩酱：关键点的置信度太低时，跳过画关键点
                continue
            if not affine_matrix is None:
                pj = np.dot(affine_matrix, np.matrix([joint[0], joint[1], 1]).reshape(3, 1))
            else:
                pj = joint[0], joint[1]
            x, y = int(pj[1]), int(pj[0])  # int(min(pj[1], 255)), int(min(pj[0], 191))
            # 关键点处画实心圆（椭圆）
            xx, yy = ellipse(x, y, r_radius=radius, c_radius=radius, shape=img_size)
            colors[xx, yy] = kptcolors[i]
            # mask[xx, yy] = 255

        # colors = colors * 1./255
        # mask = mask * 1./255

        return colors

    def get_joints(self, keypoints_path, h, w, affine_matrix=None, coeffs=None):
        with open(keypoints_path, 'r') as f:
            keypoints_data = json.load(f)
        if len(keypoints_data['people']) == 0:
            keypoints = np.zeros((18,3))
        else:
            keypoints = np.array(keypoints_data['people'][0]['pose_keypoints_2d']).reshape(-1,3)
        # joints = self.kp_to_map(img_sz=(192,256), kps=keypoints)
        color_joint = self.draw_pose_from_cords(keypoints, (h, w), affine_matrix, coeffs)
        return color_joint, keypoints

    def get_mask_from_kps(self, kps, img_h, img_w):
        rles = maskUtils.frPyObjects(kps, img_h, img_w)
        rle = maskUtils.merge(rles)
        mask = maskUtils.decode(rle)[..., np.newaxis].astype(np.float32)
        mask = mask * 255.0
        return mask

    def get_rectangle_mask(self, a, b, c, d, img_h, img_w):
        x1, y1 = a + (b - d) / 4, b + (c - a) / 4
        x2, y2 = a - (b - d) / 4, b - (c - a) / 4

        x3, y3 = c + (b - d) / 4, d + (c - a) / 4
        x4, y4 = c - (b - d) / 4, d - (c - a) / 4

        kps = [x1, y1, x2, y2]

        v0_x, v0_y = c - a, d - b
        v1_x, v1_y = x3 - x1, y3 - y1
        v2_x, v2_y = x4 - x1, y4 - y1

        cos1 = (v0_x * v1_x + v0_y * v1_y) / (
                    math.sqrt(v0_x * v0_x + v0_y * v0_y) * math.sqrt(v1_x * v1_x + v1_y * v1_y))
        cos2 = (v0_x * v2_x + v0_y * v2_y) / (
                    math.sqrt(v0_x * v0_x + v0_y * v0_y) * math.sqrt(v2_x * v2_x + v2_y * v2_y))

        if cos1 < cos2:
            kps.extend([x3, y3, x4, y4])
        else:
            kps.extend([x4, y4, x3, y3])

        kps = np.array(kps).reshape(1, -1).tolist()
        mask = self.get_mask_from_kps(kps, img_h=img_h, img_w=img_w)

        return mask

    def get_hand_mask(self, hand_keypoints, h, w):
        s_x, s_y, s_c = hand_keypoints[0]  # 咩酱：肩的x坐标、y坐标、置信度
        e_x, e_y, e_c = hand_keypoints[1]  # 咩酱：肘的x坐标、y坐标、置信度
        w_x, w_y, w_c = hand_keypoints[2]  # 咩酱：手的x坐标、y坐标、置信度

        up_mask = np.ones((h, w, 1), dtype=np.float32)
        bottom_mask = np.ones((h, w, 1), dtype=np.float32)
        if s_c > 0.1 and e_c > 0.1:  # 咩酱：肩和肘的置信度够高时，
            up_mask = self.get_rectangle_mask(s_x, s_y, e_x, e_y, h, w)
            # 对手臂的上半部分进行膨胀操作，消除两部分之间的空隙
            if h == 256:
                kernel = np.ones((25, 25), np.uint8)
            elif h == 512:
                kernel = np.ones((35, 35), np.uint8)
            else:
                raise NotImplementedError("h \'{}\' is not implemented.".format(h))
            up_mask = cv2.dilate(up_mask, kernel, iterations=1)
            up_mask = (up_mask > 0).astype(np.float32)[..., np.newaxis]
        if e_c > 0.1 and w_c > 0.1:  # 咩酱：肘和手的置信度够高时，跳过画骨骼线
            bottom_mask = self.get_rectangle_mask(e_x, e_y, w_x, w_y, h, w)
            # 对手臂的下半部分进行膨胀操作，消除两部分之间的空隙
            if h == 256:
                if self.is_train:
                    kernel = np.ones((16, 16), np.uint8)
                else:
                    kernel = np.ones((15, 15), np.uint8)
            elif h == 512:
                kernel = np.ones((20, 20), np.uint8)
            else:
                raise NotImplementedError("h \'{}\' is not implemented.".format(h))
            bottom_mask = cv2.dilate(bottom_mask, kernel, iterations=1)
            bottom_mask = (bottom_mask > 0).astype(np.float32)[..., np.newaxis]

        return up_mask, bottom_mask


    def get_palm_mask(self, hand_mask, hand_up_mask, hand_bottom_mask):
        inter_up_mask = ((hand_mask + hand_up_mask) == 2).astype(np.float32)
        hand_mask = hand_mask - inter_up_mask
        inter_bottom_mask = ((hand_mask+hand_bottom_mask) == 2).astype(np.float32)
        palm_mask = hand_mask - inter_bottom_mask

        return palm_mask

    def get_palm(self, keypoints, parsing, h, w):
        left_hand_keypoints = keypoints[[5, 6, 7], :].copy()   # 咩酱：左臂的关键点
        right_hand_keypoints = keypoints[[2, 3, 4], :].copy()  # 咩酱：右臂的关键点

        left_hand_up_mask, left_hand_botton_mask = self.get_hand_mask(left_hand_keypoints, h, w)
        right_hand_up_mask, right_hand_botton_mask = self.get_hand_mask(right_hand_keypoints, h, w)

        # 可视化。根据关键点获得：左手臂上半部分掩码、左手臂下半部分掩码、右手臂上半部分掩码、右手臂下半部分掩码。
        # cv2.imwrite('left_hand_up_mask.jpg', left_hand_up_mask * 255)
        # cv2.imwrite('left_hand_botton_mask.jpg', left_hand_botton_mask * 255)
        # cv2.imwrite('right_hand_up_mask.jpg', right_hand_up_mask * 255)
        # cv2.imwrite('right_hand_botton_mask.jpg', right_hand_botton_mask * 255)

        # mask refined by parsing
        left_hand_mask = (parsing == 14).astype(np.float32)
        right_hand_mask = (parsing == 15).astype(np.float32)

        # 可视化。语义分割图中：左手臂掩码、右手臂掩码。
        # cv2.imwrite('left_hand_mask.jpg', left_hand_mask * 255)
        # cv2.imwrite('right_hand_mask.jpg', right_hand_mask * 255)

        # 获得左手掌的掩码
        left_palm_mask = self.get_palm_mask(left_hand_mask, left_hand_up_mask, left_hand_botton_mask)
        # 获得右手掌的掩码
        right_palm_mask = self.get_palm_mask(right_hand_mask, right_hand_up_mask, right_hand_botton_mask)
        # 获得左右手掌的掩码
        palm_mask = ((left_palm_mask + right_palm_mask) > 0).astype(np.uint8)
        # 可视化。左右手掌的掩码。
        # cv2.imwrite('palm_mask.jpg', palm_mask * 255)

        return palm_mask

    def load_image(self, raw_idx):
        # load images --> range [0, 255]

        # ==================== txt文件第0列，试穿者 ====================
        fname = self.image_fnames[raw_idx]
        person_name = fname
        self.image = cv2.imread(fname)
        src_img = np.copy(self.image)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        im_shape = self.image.shape
        # padding to same size
        h, w = im_shape[0], im_shape[1]
        left_padding = (h - w) // 2
        right_padding = h - w - left_padding

        image = np.pad(self.image, ((0, 0), (left_padding, right_padding), (0, 0)), 'constant',
                       constant_values=(255, 255))

        # load keypoints --> range [0, 1]
        fname = self.kpt_fnames[raw_idx]
        pose, keypoints = self.get_joints(fname, h, w)  # self.cords_to_map(kpt, im_shape[:2])
        pose = np.pad(pose, ((0, 0), (left_padding, right_padding), (0, 0)), 'constant', constant_values=(0, 0))
        keypoints[:, 0] += left_padding  # 咩酱：x坐标加上left_padding

        # 语义分割图
        # load upper_cloth and lower body
        fname = self.parsing_fnames[raw_idx]
        parsing = cv2.imread(fname)[..., 0:1]
        parsing = np.pad(parsing, ((0, 0), (left_padding, right_padding), (0, 0)), 'constant', constant_values=(0, 0))  # [h, w=h, 1]

        # 左右手掌的掩码
        palm_mask = self.get_palm(keypoints, parsing, h, h)  # [h, w=h, 1]
        # 头的掩码
        head_mask = (parsing == 1).astype(np.uint8) + (parsing == 4).astype(np.uint8) + \
                    (parsing == 2).astype(np.uint8) + (parsing == 13).astype(np.uint8)
        # 左右鞋的掩码
        shoes_mask = (parsing == 18).astype(np.uint8) + (parsing == 19).astype(np.uint8)

        if self.is_train:
            # 不变的部分（手掌、头、鞋）的掩码
            retain_mask = palm_mask + head_mask + shoes_mask

            # 试穿者上衣的掩码
            upper_clothes_mask = (parsing == 5).astype(np.uint8) + (parsing == 6).astype(np.uint8) + \
                                 (parsing == 7).astype(np.uint8)
            # 试穿者下装（裤子、裙子等）的掩码
            lower_clothes_mask = (parsing == 9).astype(np.uint8) + (parsing == 12).astype(np.uint8)

            # 试穿者左手臂掩码、右手臂掩码
            hands_mask = (parsing == 14).astype(np.uint8) + (parsing == 15).astype(np.uint8)
            # 试穿者左腿掩码、右腿掩码
            legs_mask = (parsing == 16).astype(np.uint8) + (parsing == 17).astype(np.uint8)
            # 试穿者颈部掩码
            neck_mask = (parsing == 10).astype(np.uint8)
            gt_parsing = upper_clothes_mask * 1 + lower_clothes_mask * 2 + \
                         hands_mask * 3 + legs_mask * 4 + neck_mask * 5

            # 试穿者上衣的图片
            upper_clothes_image = upper_clothes_mask * image
            # 试穿者下装（裤子、裙子等）的图片
            lower_clothes_image = lower_clothes_mask * image

            # 试穿者上衣的掩码
            upper_clothes_mask_rgb = np.concatenate([upper_clothes_mask, upper_clothes_mask, upper_clothes_mask], axis=2)
            # 试穿者下装（裤子、裙子等）的掩码
            lower_clothes_mask_rgb = np.concatenate([lower_clothes_mask, lower_clothes_mask, lower_clothes_mask], axis=2)
            # 试穿者上衣的掩码*255
            upper_clothes_mask_rgb = upper_clothes_mask_rgb * 255
            # 试穿者下装（裤子、裙子等）的掩码*255
            lower_clothes_mask_rgb = lower_clothes_mask_rgb * 255

            norm_img, norm_img_lower, denorm_upper_img, denorm_lower_img, \
            M_invs, denorm_hand_masks, norm_clothes_masks, norm_clothes_masks_lower \
                = self.normalize(upper_clothes_image, lower_clothes_image,
                                 upper_clothes_mask_rgb, lower_clothes_mask_rgb, lower_keypoints=keypoints, box_factor=2)

            return image, pose, norm_img, norm_img_lower, denorm_upper_img, denorm_lower_img, M_invs, \
                   gt_parsing, denorm_hand_masks, norm_clothes_masks, norm_clothes_masks_lower, retain_mask
        else:
            # 试穿者下装（裤子、裙子等）的掩码
            lower_clothes_mask = (parsing == 9).astype(np.uint8) + (parsing == 12).astype(np.uint8) + \
                                 (parsing == 6).astype(np.uint8)
            # 试穿者下装（裤子、裙子等）的图片
            lower_clothes_image = lower_clothes_mask * image

            # 试穿者手掌、头、鞋的图片
            image = image * (palm_mask + head_mask + shoes_mask)

            # ==================== txt文件第1列，从这个人身上扒衣服 ====================
            fname = self.clothes_image_fnames[raw_idx]
            clothes_name = fname
            self.clothes = cv2.imread(fname)
            trg_img = np.copy(self.clothes)
            self.clothes = cv2.cvtColor(self.clothes, cv2.COLOR_BGR2RGB)
            clothes = np.pad(self.clothes, ((0, 0), (left_padding, right_padding), (0, 0)), 'constant',
                             constant_values=(255, 255))

            fname = self.clothes_kpt_fnames[raw_idx]
            clothes_pose, clothes_keypoints = self.get_joints(fname, h, w)  # self.cords_to_map(kpt, im_shape[:2])
            clothes_pose = np.pad(clothes_pose, ((0, 0), (left_padding, right_padding), (0, 0)), 'constant',
                                  constant_values=(0, 0))
            clothes_keypoints[:, 0] += left_padding

            # 被扒者语义分割图
            fname = self.clothes_parsing_fnames[raw_idx]
            clothes_parsing = cv2.imread(fname)[..., 0:1]
            clothes_parsing = np.pad(clothes_parsing, ((0, 0), (left_padding, right_padding), (0, 0)), 'constant',
                                     constant_values=(0, 0))

            # 被扒者上衣的掩码
            upper_clothes_mask = (clothes_parsing == 5).astype(np.uint8) + (clothes_parsing == 6).astype(np.uint8) + \
                                 (clothes_parsing == 7).astype(np.uint8)
            # 被扒者上衣的图片
            upper_clothes_image = upper_clothes_mask * clothes

            # 被扒者上衣的掩码
            upper_clothes_mask_rgb = np.concatenate([upper_clothes_mask, upper_clothes_mask, upper_clothes_mask], axis=2)
            # 试穿者下装（裤子、裙子等）的掩码
            lower_clothes_mask_rgb = np.concatenate([lower_clothes_mask, lower_clothes_mask, lower_clothes_mask], axis=2)
            upper_clothes_mask_rgb = upper_clothes_mask_rgb * 255
            lower_clothes_mask_rgb = lower_clothes_mask_rgb * 255

            # 被扒者骨骼图
            upper_pose = clothes_pose
            # 试穿者骨骼图
            lower_pose = pose
            # 被扒者关键点
            upper_keypoints = clothes_keypoints
            # 试穿者关键点
            lower_keypoints = keypoints

            '''
            upper_clothes_image     [256, 256, 3]    被扒者上衣的图片
            lower_clothes_image     [256, 256, 3]    试穿者下装（裤子、裙子等）的图片
            upper_clothes_mask_rgb  [256, 256, 3]    被扒者上衣的掩码
            lower_clothes_mask_rgb  [256, 256, 3]    试穿者下装（裤子、裙子等）的掩码
            upper_pose              [256, 256, 3]    被扒者骨骼图
            lower_pose              [256, 256, 3]    试穿者骨骼图
            upper_keypoints         [18, 3]    被扒者关键点
            lower_keypoints         [18, 3]    试穿者关键点
            box_factor              int        图片宽高缩小为原来的1/(2^box_factor)
            Return:
            norm_img             [64, 64, 30]
            norm_pose            [64, 64, 30]
            denorm_upper_img     [256, 256, 3]   被扒者上衣的图片（经过加工变形）
            denorm_lower_img     [256, 256, 3]   试穿者下装（裤子、裙子等）的图片（经过加工变形）
            '''
            norm_img, norm_pose, denorm_upper_img, denorm_lower_img = self.normalize(upper_clothes_image,
                                                                                     lower_clothes_image,
                                                                                     upper_clothes_mask_rgb,
                                                                                     lower_clothes_mask_rgb,
                                                                                     upper_pose, lower_pose,
                                                                                     upper_keypoints, lower_keypoints, box_factor=2)

            # 可视化。根据关键点获得：左手臂上半部分掩码、左手臂下半部分掩码、右手臂上半部分掩码、右手臂下半部分掩码。
            # cv2.imwrite('denorm_upper_img.jpg', cv2.cvtColor(denorm_upper_img, cv2.COLOR_RGB2BGR))
            # cv2.imwrite('denorm_lower_img.jpg', cv2.cvtColor(denorm_lower_img, cv2.COLOR_RGB2BGR))

            person_name = person_name.split('\\')[-1]
            clothes_name = clothes_name.split('\\')[-1]
            return image, pose, norm_img, norm_pose, denorm_upper_img, denorm_lower_img, person_name, clothes_name, src_img, trg_img

    def __getitem__(self, idx):
        if self.is_train:
            image, pose, norm_img, norm_img_lower, denorm_upper_img, denorm_lower_img, M_invs, gt_parsing, \
                denorm_hand_masks, norm_clothes_masks, norm_clothes_masks_lower, retain_mask = self.load_image(self.raw_idx[idx])

            image = image.transpose(2, 0, 1)                    # HWC => CHW
            pose = pose.transpose(2, 0, 1)                      # HWC => CHW
            norm_img = norm_img.transpose(2, 0, 1)
            norm_img_lower = norm_img_lower.transpose(2, 0, 1)
            denorm_upper_img = denorm_upper_img.transpose(2, 0, 1)
            denorm_lower_img = denorm_lower_img.transpose(2, 0, 1)

            norm_clothes_masks = norm_clothes_masks.transpose(2, 0, 1)
            norm_clothes_masks_lower = norm_clothes_masks_lower.transpose(2, 0, 1)

            # upper_clothes_mask = upper_clothes_mask.transpose(2,0,1)
            # lower_clothes_mask = lower_clothes_mask.transpose(2,0,1)
            gt_parsing = gt_parsing.transpose(2, 0, 1)

            retain_mask = retain_mask.transpose(2, 0, 1)

            # concat the pose and img since they often binded together
            # norm_img = np.concatenate((norm_img, norm_pose), axis=0)


            denorm_random_mask = np.zeros((256, 256, 1), dtype=np.uint8)
            # random.seed(5)
            random.seed(1)

            if random.random() < 0.4:
                for mask in denorm_hand_masks:
                    if random.random() < 0.5:
                        denorm_random_mask += mask
            # if random.random() < 0.4:
            #     for mask in denorm_leg_masks:
            #         if random.random() < 0.5:
            #             denorm_random_mask += mask

            if random.random() < 0.9:
                fname = self._random_mask_acgpn_fnames[self.raw_idx[idx] % self._mask_acgpn_numbers]
                random_mask = cv2.imread(fname)[..., 0:1]
                random_mask = cv2.resize(random_mask, (256, 256))[..., np.newaxis]
                denorm_random_mask += random_mask

            denorm_random_mask = (denorm_random_mask > 0).astype(np.uint8)
            denorm_random_mask = denorm_random_mask.transpose(2, 0, 1)

            denorm_upper_img_erase = denorm_upper_img * (1-denorm_random_mask)
            denorm_upper_mask = (np.sum(denorm_upper_img_erase, axis=0, keepdims=True) > 0).astype(np.uint8)
            denorm_lower_img_erase = denorm_lower_img * (1-denorm_random_mask)
            denorm_lower_mask = (np.sum(denorm_lower_img_erase, axis=0, keepdims=True) > 0).astype(np.uint8)

            # assert isinstance(image, np.ndarray)
            # assert list(image.shape) == self.image_shape
            # assert image.dtype == np.uint8

            # return image.copy(), pose.copy(), sem.copy(), norm_img.copy(), denorm_upper_img.copy(), denorm_lower_img.copy(), \
            #         M_invs.copy(), upper_clothes_mask.copy(), lower_clothes_mask.copy(), denorm_random_mask.copy(), \
            #         denorm_upper_mask.copy(), denorm_lower_mask.copy(), norm_clothes_masks.copy()
            # return image.copy(), pose.copy(), sem.copy(), norm_img.copy(), denorm_upper_img_erase.copy(), denorm_lower_img_erase.copy(), \
            #         M_invs.copy(), upper_clothes_mask.copy(), lower_clothes_mask.copy(), \
            #         denorm_upper_mask.copy(), denorm_lower_mask.copy(), norm_clothes_masks.copy(), retain_mask.copy()
            return image.copy(), pose.copy(), norm_img.copy(), norm_img_lower.copy(), denorm_upper_img_erase.copy(), denorm_lower_img_erase.copy(), \
                    M_invs.copy(), gt_parsing.copy(), denorm_upper_mask.copy(), denorm_lower_mask.copy(), \
                    norm_clothes_masks.copy(), norm_clothes_masks_lower.copy(), retain_mask.copy()
        else:
            image, pose, norm_img, norm_pose, denorm_upper_img, denorm_lower_img, person_name, clothes_name, src_img, trg_img = self.load_image(self.raw_idx[idx])

            image = image.transpose(2, 0, 1)  # HWC => CHW
            pose = pose.transpose(2, 0, 1)    # HWC => CHW
            norm_img = norm_img.transpose(2, 0, 1)
            norm_pose = norm_pose.transpose(2, 0, 1)
            denorm_upper_img = denorm_upper_img.transpose(2, 0, 1)
            denorm_lower_img = denorm_lower_img.transpose(2, 0, 1)

            # concat the pose and img since they often binded together
            norm_img = np.concatenate((norm_img, norm_pose), axis=0)

            denorm_upper_mask = (np.sum(denorm_upper_img, axis=0, keepdims=True) > 0).astype(np.uint8)
            denorm_lower_mask = (np.sum(denorm_lower_img, axis=0, keepdims=True) > 0).astype(np.uint8)

            assert isinstance(image, np.ndarray)
            assert list(image.shape) == self.image_shape
            assert image.dtype == np.uint8

            datas = {
                'image': image.copy(),
                'pose': pose.copy(),
                'norm_img': norm_img.copy(),
                'denorm_upper_img': denorm_upper_img.copy(),
                'denorm_lower_img': denorm_lower_img.copy(),
                'denorm_upper_mask': denorm_upper_mask.copy(),
                'denorm_lower_mask': denorm_lower_mask.copy(),
                'person_name': person_name,
                'clothes_name': clothes_name,
                'src_img': src_img,
                'trg_img': trg_img,
            }
            return datas

    @property
    def image_shape(self):
        return list(self.raw_shape[1:])

    def __len__(self):
        size = self.raw_idx.size
        if self.is_train:
            pass
        else:
            size = min(self.test_count, size)
        return size

    def prepare_data_infos(self, dataroot):
        pass
