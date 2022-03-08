
import cv2
from PIL import Image
import numpy as np
import os
import paddlehub as hub
from tqdm import tqdm


# model = hub.Module(name='animegan_v2_hayao_99', use_gpu=True)
model = hub.Module(name='animegan_v2_shinkai_53', use_gpu=True)
# model = hub.Module(name='animegan_v2_paprika_98', use_gpu=True)




img_path = 'D://GitHub/data/flowers_512'
save_path = 'aaaaaaaaaaaaa'
aaaaaaaaaaaaaa = os.listdir(img_path)


os.makedirs(save_path, exist_ok=True)

for aaa in aaaaaaaaaaaaaa:
    item = os.path.join(img_path, aaa)
    result = model.style_transfer(images=[cv2.imread(item)])
    result_img = result[0]
    cv2.imwrite(os.path.join(save_path, aaa), result_img)


print()






