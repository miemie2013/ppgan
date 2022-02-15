

import numpy as np
import torch
import paddle

'''
self.style_mixing_prob > 0 时使用到

'''
style_mixing_prob = 0.5

ws = np.random.normal(size=[2, 3, 512])


# ws2 = torch.Tensor(ws)
# aaaaaaaaaaaaa = torch.empty([1, ], dtype=torch.int64, device=ws2.device)
# for y in range(10):
#     cutoff2 = aaaaaaaaaaaaa.random_(1, ws2.shape[1])
#     print(cutoff2)
#     bbbbbbbbbbb = torch.rand([1, ], device=ws2.device) < style_mixing_prob
#     print(bbbbbbbbbbb)
#     cutoff3 = torch.where(bbbbbbbbbbb, cutoff2, torch.full_like(cutoff2, ws.shape[1]))
#     print(cutoff3)
#     print()
#     print()



ws = paddle.to_tensor(ws)

for y in range(10):
    cutoff2 = paddle.randint(low=1, high=ws.shape[1], shape=[1, ], dtype='int64')
    print(cutoff2)
    bbbbbbbbbbb = paddle.rand([1, ], dtype='float32') < style_mixing_prob
    print(bbbbbbbbbbb)
    cutoff3 = paddle.where(bbbbbbbbbbb, cutoff2, paddle.full_like(cutoff2, ws.shape[1]))
    print(cutoff3)
    if cutoff3 == ws.shape[1]:
        print('aaaaaaaaaaaaaaaaaa')
    print()
    print()





print()



