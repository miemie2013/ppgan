

import numpy as np
import torch
import paddle

'''


'''
ws = np.random.normal(size=[2, 3])


ws2 = torch.Tensor(ws)

aaa1 = torch.exp2(ws2)

ws3 = paddle.to_tensor(ws)
aaa2 = paddle.pow(paddle.zeros_like(ws3, dtype=ws3.dtype) + 2.0, ws3)


# aaa = torch.exp2(ws2)



print()



