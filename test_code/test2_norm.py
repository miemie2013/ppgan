

import numpy as np
import torch
import paddle
import sys

'''
...

'''


def feature_normalize2(feature_in):
    feature_in_norm = torch.norm(feature_in, 2, 1, keepdim=True) + sys.float_info.epsilon
    feature_in_norm = torch.div(feature_in, feature_in_norm)
    return feature_in_norm


def feature_normalize(feature_in):
    feature_in_norm = paddle.norm(feature_in, 2, 1, keepdim=True) + sys.float_info.epsilon
    feature_in_norm = feature_in / feature_in_norm
    return feature_in_norm


x = np.random.normal(size=[8, 3, 64, 64])


ws2 = torch.Tensor(x)
rrrrrrrrrrr2 = feature_normalize2(ws2)

ws = paddle.to_tensor(x)
rrrrrrrrrrr = feature_normalize(ws)


ddd = np.sum((rrrrrrrrrrr2.cpu().detach().numpy() - rrrrrrrrrrr.numpy())**2)
print('ddd=%.6f' % ddd)


print()



