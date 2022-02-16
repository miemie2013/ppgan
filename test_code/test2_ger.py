

import torch
import paddle

'''
def upfirdn2d_setup_filter(shape, normalize=True, flip_filter=False, gain=1, separable=None): 使用到

'''


f = [1, 3, 3, 1]
fff = torch.as_tensor(f, dtype=torch.float32)

v = fff
vx = v[..., 0]; vy = v[..., 1]; vz = v[..., 2]


fff = fff.ger(fff)

aaa = list(range(fff.ndim))
fff = fff.flip(aaa)



shape = paddle.to_tensor(f, dtype='float32')
# ger()相当于向量自乘
shape = paddle.unsqueeze(shape, 1)  # [n, 1]
shape = paddle.matmul(shape, shape.transpose((1, 0)))  # [n, n]

bbb = list(range(shape.ndim))
shape = shape.flip(bbb)

print()



