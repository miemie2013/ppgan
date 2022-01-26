

import numpy as np
import torch
import paddle

'''
def upfirdn2d_setup_filter(shape, normalize=True, flip_filter=False, gain=1, separable=None): 使用到

'''

filter = np.random.normal(size=[4, 4])
filter2 = torch.Tensor(filter)
filter2 = filter2[np.newaxis, np.newaxis].repeat([1, 1, 2, 2])


filter = paddle.to_tensor(filter)
filter = paddle.unsqueeze(filter, [0, 1]).tile([1, 1, 2, 2])



ddd = np.sum((filter2.cpu().detach().numpy() - filter.numpy())**2)
print('ddd=%.6f' % ddd)



print()



