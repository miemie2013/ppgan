
import paddle
import numpy as np


ckpt_file1 = "01_19.pdparams"
ckpt_file2 = "01_19_DDP.pdparams"


state_dict1_paddle = paddle.load(ckpt_file1)

state_dict2_paddle = paddle.load(ckpt_file2)

d_value = 0.0005
print('======================== diff(weights) > d_value=%.6f ========================' % d_value)
for key, value1 in state_dict1_paddle.items():
    if '_ema' in key:
        continue
    if 'augment_pipe.' in key:
        continue
    v1 = value1.numpy()
    value2 = state_dict2_paddle[key]
    v2 = value2.numpy()
    ddd = np.sum((v1 - v2) ** 2)
    if ddd > d_value:
        print('diff=%.6f (%s)' % (ddd, key))

print()
print()
print('======================== diff(weights) <= d_value=%.6f ========================' % d_value)
for key, value1 in state_dict1_paddle.items():
    if '_ema' in key:
        continue
    if 'augment_pipe.' in key:
        continue
    v1 = value1.numpy()
    value2 = state_dict2_paddle[key]
    v2 = value2.numpy()
    ddd = np.sum((v1 - v2) ** 2)
    if ddd <= d_value:
        print('diff=%.6f (%s)' % (ddd, key))


print()
