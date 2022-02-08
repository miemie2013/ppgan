

import numpy as np
import torch
import paddle

'''
...

'''


dic2 = np.load('ce_parsing.npz')

pred_parsing2 = dic2['pred_parsing']
gt_parsing2 = dic2['gt_parsing']
rrr2 = dic2['rrr']



# 每个类别的权重（6个类别）
class_weight = paddle.to_tensor([1., 2., 2., 3., 3., 3.])
ce_parsing = paddle.nn.CrossEntropyLoss(ignore_index=255, weight=class_weight)


pred_parsing = paddle.to_tensor(pred_parsing2)
gt_parsing = paddle.to_tensor(gt_parsing2)


aaaaaaaaaaaaa = paddle.cast(gt_parsing, dtype=paddle.int64)[:, 0, :, :]
loss = ce_parsing(pred_parsing.transpose((0, 2, 3, 1)), aaaaaaaaaaaaa)
loss = loss.numpy()


print()



