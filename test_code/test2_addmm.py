

import numpy as np
import torch
import paddle

'''
class FullyConnectedLayer(nn.Layer): 使用到

'''

x = np.random.normal(size=[1, 512])
w = np.random.normal(size=[512, 512])
b = np.random.normal(size=[512, ])


x2 = torch.Tensor(x)
w2 = torch.Tensor(w)
b2 = torch.Tensor(b)

x2 = torch.addmm(b2.unsqueeze(0), x2, w2.t())


x = paddle.to_tensor(x)
w = paddle.to_tensor(w)
b = paddle.to_tensor(b)

x = paddle.addmm(b.unsqueeze(0), x, w.t())



ddd = np.sum((x2.cpu().detach().numpy() - x.numpy())**2)
print('ddd=%.6f' % ddd)


print()



