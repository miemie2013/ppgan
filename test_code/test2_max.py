

import torch
import numpy as np
import paddle

'''
def upfirdn2d_setup_filter(shape, normalize=True, flip_filter=False, gain=1, separable=None): 使用到

'''


x = np.random.normal(size=[2, 3, 64, 64])

f = [1, 3, 3, 1]
f2 = [2, 0.3, 7, 0.9]
images = torch.as_tensor(x, dtype=torch.float32)
fff = torch.as_tensor(f, dtype=torch.float32)
fff2 = torch.as_tensor(f2, dtype=torch.float32)

bbb = fff.max(fff2)
bbb2 = fff.min(fff2)

ttt = bbb2.ceil()
mx0, my0, mx1, my1 = ttt.to(torch.int32)

images = torch.nn.functional.pad(input=images, pad=[mx0,mx1,my0,my1], mode='reflect')



images2 = paddle.to_tensor(x, dtype='float32')
aaaaa = paddle.to_tensor(f, dtype='float32')
aaaaa2 = paddle.to_tensor(f2, dtype='float32')

ccc = paddle.maximum(aaaaa, aaaaa2)
ccc2 = paddle.minimum(aaaaa, aaaaa2)

ttt2 = ccc2.ceil()
ttt2 = paddle.cast(ttt2, dtype=paddle.int32)
mx02, my02, mx12, my12 = ttt2
# images2 = paddle.nn.functional.pad(images2, pad=[mx02,mx12,my02,my12], mode='reflect')
images2 = paddle.nn.functional.pad(images2, pad=[mx02,mx12,my02,my12], mode='reflect')


ddd = np.sum((images.cpu().detach().numpy() - images2.numpy())**2)
print('ddd=%.6f' % ddd)



print()



