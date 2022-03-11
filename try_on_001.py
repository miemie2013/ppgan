
import cv2
import os
import numpy as np

def fff(aaaaa):
    return 1


images_dir = '../UPT_256_192_my2/Zalando_256_192/images/test'

save_dataroot = '../dataset_tryon'
os.makedirs(save_dataroot, exist_ok=True)

images_folder = os.path.join(save_dataroot, 'images')
os.makedirs(images_folder, exist_ok=True)

images_test_folder = os.path.join(save_dataroot, 'images', 'test')
os.makedirs(images_test_folder, exist_ok=True)

keypoints_folder = os.path.join(save_dataroot, 'keypoints')
os.makedirs(keypoints_folder, exist_ok=True)

parsing_folder = os.path.join(save_dataroot, 'parsing')
os.makedirs(parsing_folder, exist_ok=True)

filenames = os.listdir(images_dir)



target_h = 512
target_w = 320


for name in filenames:
    path = os.path.join(images_dir, name)
    save_path = os.path.join(images_test_folder, name)
    img = cv2.imread(path)
    h, w, _ = img.shape

    im_scale = float(target_h) / float(h)
    im_scale_x = im_scale
    im_scale_y = im_scale

    img = cv2.resize(
        img,
        None,
        None,
        fx=im_scale_x,
        fy=im_scale_y,
        interpolation=cv2.INTER_LINEAR)

    h, w, _ = img.shape
    pad = (w - target_w) // 2
    img = img[:, pad:pad+target_w, :]
    cv2.imwrite(save_path, img)

print('Done.')



