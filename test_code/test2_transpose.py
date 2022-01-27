

import numpy as np
import torch
import paddle

'''
def conv2d_resample(x, w, filter=None, up=1, down=1, padding=0, groups=1, flip_weight=True, flip_filter=False):使用到

'''

# w = np.random.normal(size=[2, 3, 4, 5])
#
# w2 = torch.Tensor(w)
# w2 = w2.transpose(0, 1)
#
# w = paddle.to_tensor(w)
# w = w.transpose((1, 0, 2, 3))
#
# ddd = np.sum((w2.cpu().detach().numpy() - w.numpy()) ** 2)
# print('ddd=%.6f' % ddd)

# ------------------------------------

w = np.random.normal(size=[2, 3, 4, 5, 6])

w2 = torch.Tensor(w)
w2 = w2.transpose(1, 2)

w = paddle.to_tensor(w)
w = w.transpose((0, 2, 1, 3, 4))

ddd = np.sum((w2.cpu().detach().numpy() - w.numpy()) ** 2)
print('ddd=%.6f' % ddd)




print()



