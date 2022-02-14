

import numpy as np
import torch
import paddle

'''
class FullyConnectedLayer(nn.Layer): 使用到

'''

weight = np.random.normal(size=[512, 512, 3, 3]).astype(np.float32)
styles = np.random.normal(size=[1, 512]).astype(np.float32)


weight2 = torch.Tensor(weight)
styles2 = torch.Tensor(styles)

aaa2 = weight2.norm(float('inf'), dim=[1,2,3], keepdim=True)
bbb2 = styles2.norm(float('inf'), dim=1, keepdim=True)

weight = paddle.to_tensor(weight)
styles = paddle.to_tensor(styles)

d0, d1, d2, d3 = weight.shape
weight_temp = weight.reshape((d0, d1, d2 * d3))
aaa = paddle.norm(weight_temp, p=np.inf, axis=[1, 2], keepdim=True)
aaa = aaa.reshape((d0, 1, 1, 1))
bbb = paddle.norm(styles, p=np.inf, axis=1, keepdim=True)


ddd = np.sum((aaa2.cpu().detach().numpy() - aaa.numpy())**2)
print('ddd=%.6f' % ddd)

ddd = np.sum((bbb2.cpu().detach().numpy() - bbb.numpy())**2)
print('ddd=%.6f' % ddd)


print()



