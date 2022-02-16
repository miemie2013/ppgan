

import torch
import numpy as np
import paddle

'''


'''

dic2 = np.load('affine_grid_data.npz')


G_inv = dic2['G_inv']
images = dic2['images']
shape2 = dic2['shape']
shape = []
for sp in shape2:
    shape.append(sp)

G_inv2 = torch.as_tensor(G_inv, dtype=torch.float32)
images2 = torch.as_tensor(images, dtype=torch.float32)

grid2 = torch.nn.functional.affine_grid(theta=G_inv2[:, :2, :], size=shape, align_corners=False)
images2 = torch.nn.functional.grid_sample(input=images2, grid=grid2, mode='bilinear', padding_mode='zeros', align_corners=False)

G_inv = paddle.to_tensor(G_inv, dtype='float32')
images = paddle.to_tensor(images, dtype='float32')

grid = paddle.nn.functional.affine_grid(theta=G_inv[:, :2, :], out_shape=shape, align_corners=False)
images = paddle.nn.functional.grid_sample(images, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)


import paddle.nn.functional as F
# theta shape = [1, 2, 3]
theta = np.array([[[-0.7, -0.4, 0.3],
                   [ 0.6,  0.5, 1.5]]]).astype("float32")
theta_t = paddle.to_tensor(theta)
y_t = F.affine_grid(
        theta_t,
        [1, 2, 3, 3],
        align_corners=False)
print(y_t)

ddd = np.sum((grid2.cpu().detach().numpy() - grid.numpy())**2)
print('ddd=%.6f' % ddd)
ddd = np.sum((images2.cpu().detach().numpy() - images.numpy())**2)
print('ddd=%.6f' % ddd)

print()



