

import numpy as np
import torch
import megengine as mge
import megengine.functional as F

'''
def _conv2d_wrapper(x, w, stride=1, padding=0, groups=1, transpose=False, flip_weight=True):使用到

'''

x = np.random.normal(size=[4])
f = torch.Tensor(x)

f = f.flip(list(range(f.ndim)))

f2 = mge.tensor(x)
f2 = f2[::-1]


ddd = np.sum((f.cpu().detach().numpy() - f2.numpy())**2)
print('ddd=%.6f' % ddd)




x = np.random.normal(size=[4, 4])
f = torch.Tensor(x)

f = f.flip(list(range(f.ndim)))

f2 = mge.tensor(x)
f2 = f2[::-1, ::-1]


ddd = np.sum((f.cpu().detach().numpy() - f2.numpy())**2)
print('ddd=%.6f' % ddd)





x = np.random.normal(size=[4, 4, 4])
f = torch.Tensor(x)

f = f.flip(list(range(f.ndim)))

f2 = mge.tensor(x)
f2 = f2[::-1, ::-1, ::-1]


ddd = np.sum((f.cpu().detach().numpy() - f2.numpy())**2)
print('ddd=%.6f' % ddd)




x = np.random.normal(size=[4, 4, 4, 4])
f = torch.Tensor(x)

f = f.flip(list(range(f.ndim)))

f2 = mge.tensor(x)
f2 = f2[::-1, ::-1, ::-1, ::-1]


ddd = np.sum((f.cpu().detach().numpy() - f2.numpy())**2)
print('ddd=%.6f' % ddd)






x = np.random.normal(size=[4, 4, 4, 4])
f = torch.Tensor(x)

f = f.flip([2, 3])

f2 = mge.tensor(x)
f2 = f2[:, :, ::-1, ::-1]


ddd = np.sum((f.cpu().detach().numpy() - f2.numpy())**2)
print('ddd=%.6f' % ddd)







print()



