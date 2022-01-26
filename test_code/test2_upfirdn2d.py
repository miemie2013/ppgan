

import numpy as np
import torch
import paddle

'''
def upfirdn2d_setup_filter(shape, normalize=True, flip_filter=False, gain=1, separable=None): 使用到

'''

x = np.random.normal(size=[1, 64, 256, 256])
x2 = torch.Tensor(x)
batch_size2, num_channels2, in_height2, in_width2 = x2.shape
upx, upy = 2, 2
downx, downy = 1, 1
padx0, padx1, pady0, pady1 = 2, 2, 2, 2

x2 = x2.reshape([batch_size2, num_channels2, in_height2, 1, in_width2, 1])
x2 = torch.nn.functional.pad(x2, [0, upx - 1, 0, 0, 0, upy - 1])  # 表示的是最后3维开始和结束处填充的数目。
x2 = x2.reshape([batch_size2, num_channels2, in_height2 * upy, in_width2 * upx])




x = paddle.to_tensor(x)
batch_size, num_channels, in_height, in_width = x.shape
upx, upy = 2, 2
downx, downy = 1, 1
padx0, padx1, pady0, pady1 = 2, 2, 2, 2

# paddle最多支持5维张量，所以分开2次pad。
# 根据data_format指定的意义填充(pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
x = x.reshape([batch_size, num_channels, in_height, 1, in_width])
x = paddle.nn.functional.pad(x, [0, 0, 0, upy - 1, 0, 0], data_format="NCDHW")
x = x.reshape([batch_size, num_channels, in_height * upy, in_width, 1])
x = paddle.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, 0], data_format="NCDHW")
x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])


ddd = np.sum((x2.cpu().detach().numpy() - x.numpy())**2)
print('ddd=%.6f' % ddd)



print()



