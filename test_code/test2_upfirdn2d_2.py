

import numpy as np
import torch
import paddle

'''
def upfirdn2d_setup_filter(shape, normalize=True, flip_filter=False, gain=1, separable=None): 使用到

'''

x = np.random.normal(size=[1, 64, 256, 256])
x2 = torch.Tensor(x)
batch_size2, num_channels2, in_height2, in_width2 = x2.shape
upx, upy = 2, 2
downx, downy = 1, 1
padx0, padx1, pady0, pady1 = 2, 2, 2, 2

x2 = torch.nn.functional.pad(x2, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)])
x2 = x2[:, :, max(-pady0, 0) : x2.shape[2] - max(-pady1, 0), max(-padx0, 0) : x2.shape[3] - max(-padx1, 0)]


x = paddle.to_tensor(x)
batch_size, num_channels, in_height, in_width = x.shape
upx, upy = 2, 2
downx, downy = 1, 1
padx0, padx1, pady0, pady1 = 2, 2, 2, 2


x = paddle.nn.functional.pad(x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)])
x = x[:, :, max(-pady0, 0) : x.shape[2] - max(-pady1, 0), max(-padx0, 0) : x.shape[3] - max(-padx1, 0)]

ddd = np.sum((x2.cpu().detach().numpy() - x.numpy())**2)
print('ddd=%.6f' % ddd)



print()



