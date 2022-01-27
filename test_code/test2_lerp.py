

import numpy as np
import torch
import paddle

'''
class MappingNetwork(nn.Layer): 使用到
就像bn的均值和方程的更新（指数滑动平均）。

'''

w_avg_beta = 0.995
w_dim = 512


w_avg = np.random.normal(size=[w_dim, ])
temp3 = np.random.normal(size=[w_dim, ])


w_avg2 = torch.Tensor(w_avg)
temp32 = torch.Tensor(temp3)
temp32 = temp32.lerp(w_avg2, w_avg_beta)
w_avg2.copy_(temp32)


w_avg = paddle.to_tensor(w_avg)
temp3 = paddle.to_tensor(temp3)
temp3 = temp3 + w_avg_beta * (w_avg - temp3)



ddd = np.sum((temp32.cpu().detach().numpy() - temp3.numpy())**2)
print('ddd=%.6f' % ddd)

ddd = np.sum((w_avg2.cpu().detach().numpy() - temp3.numpy())**2)
print('ddd=%.6f' % ddd)


print()



