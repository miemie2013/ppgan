

import numpy as np
import torch
import paddle

'''
def _conv2d_wrapper(x, w, stride=1, padding=0, groups=1, transpose=False, flip_weight=True):使用到

'''

x = np.random.normal(size=[8, 64, 5, 7])
x2 = torch.Tensor(x)

sss2 = x2.stride()[1]


x = paddle.to_tensor(x)
sss = x.shape[2] * x.shape[3]

print()



