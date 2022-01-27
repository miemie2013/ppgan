

import numpy as np
import torch
import paddle

'''
class MappingNetwork(nn.Layer): 使用到
就像bn的均值和方程的更新（指数滑动平均）。

'''


x = np.random.normal(size=[1, 14, 512])


ws2 = torch.Tensor(x)
rrrrrrrrrrr2 = ws2.unbind(dim=1)
w_iter2 = iter(ws2.unbind(dim=1))

ws = paddle.to_tensor(x)
rrrrrrrrrrr = ws.unbind(axis=1)
w_iter = iter(ws.unbind(axis=1))


for i in range(14):
    bbb = next(w_iter2)
    aaa = next(w_iter)
    ddd = np.sum((bbb.cpu().detach().numpy() - aaa.numpy())**2)
    print('ddd=%.6f' % ddd)

print()



