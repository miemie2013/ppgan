

import numpy as np
import torch
import paddle

'''
...

'''


x = np.random.normal(size=[8, 64, 256, 256])

pred_parsing2 = torch.Tensor(x)

softmax2 = torch.nn.Softmax(dim=1)
aaaaaaaaaaaaaa2 = softmax2(pred_parsing2.detach())
bbbbbbbbbb2 = torch.argmax(aaaaaaaaaaaaaa2, dim=1)
parsing_index2 = bbbbbbbbbb2[:, None, ...].float()

upper_mask2 = (parsing_index2 == 1).float()
lower_mask2 = (parsing_index2 == 2).float()



pred_parsing = paddle.to_tensor(x)
softmax = paddle.nn.Softmax(axis=1)
aaaaaaaaaaaaaa = softmax(pred_parsing.detach())
bbbbbbbbbb = paddle.argmax(aaaaaaaaaaaaaa, axis=1)
parsing_index = bbbbbbbbbb.unsqueeze(1)
upper_mask = paddle.cast(parsing_index == 1, dtype=paddle.float32)
lower_mask = paddle.cast(parsing_index == 2, dtype=paddle.float32)


ddd = np.sum((aaaaaaaaaaaaaa2.cpu().detach().numpy() - aaaaaaaaaaaaaa.numpy())**2)
print('ddd=%.6f' % ddd)
ddd = np.sum((bbbbbbbbbb2.cpu().detach().numpy() - bbbbbbbbbb.numpy())**2)
print('ddd=%.6f' % ddd)

ddd = np.sum((upper_mask2.cpu().detach().numpy() - upper_mask.numpy())**2)
print('ddd=%.6f' % ddd)
ddd = np.sum((lower_mask2.cpu().detach().numpy() - lower_mask.numpy())**2)
print('ddd=%.6f' % ddd)



print()



