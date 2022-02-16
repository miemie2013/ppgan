

import numpy as np
import torch
import paddle

'''


'''
ws = np.random.normal(size=[4, 4])


margin2 = torch.Tensor(ws)

margin2 = margin2.max(1)
margin3 = margin2.values  # [x0, y0, x1, y1]

margin = paddle.to_tensor(ws)
margin4 = margin.max(1)

print()



