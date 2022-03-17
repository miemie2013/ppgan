

import torch
import paddle
import numpy as np




# ckpt_file = 'styleganv2ada_32_afhqcat.pdparams'
ckpt_file = 'styleganv2ada_32_afhqcat_step19_pytorch.pdparams'
state_dict_pytorch = paddle.load(ckpt_file)


# ======================== discriminator ========================
print('======================== discriminator ========================')
model_dic_pytorch = {}
for key, value in state_dict_pytorch['discriminator'].items():
    model_dic_pytorch[key] = value.numpy()

# ckpt_file = 'D_00.pdparams'
ckpt_file = 'D_19.pdparams'
state_dict = paddle.load(ckpt_file)
model_dic_paddle = {}
for key, value in state_dict.items():
    model_dic_paddle[key] = value.numpy()


for key, value in model_dic_paddle.items():
    value2 = model_dic_pytorch[key]
    ddd = np.sum((value - value2) ** 2)
    # if ddd > 1.00001:
    #     print(key)
    print('diff=%.6f (%s)' % (ddd, key))

print('==============================================')
print()



# ======================== synthesis_ema ========================
print('======================== synthesis_ema ========================')
model_dic_pytorch = {}
for key, value in state_dict_pytorch['synthesis_ema'].items():
    model_dic_pytorch[key] = value.numpy()

ckpt_file = 'synthesis_ema_19.pdparams'
state_dict = paddle.load(ckpt_file)
model_dic_paddle = {}
for key, value in state_dict.items():
    model_dic_paddle[key] = value.numpy()


for key, value in model_dic_paddle.items():
    value2 = model_dic_pytorch[key]
    ddd = np.sum((value - value2) ** 2)
    # if ddd > 1.00001:
    #     print(key)
    print('diff=%.6f (%s)' % (ddd, key))

print('==============================================')
print()



# ======================== synthesis ========================
print('======================== synthesis ========================')
model_dic_pytorch = {}
for key, value in state_dict_pytorch['synthesis'].items():
    model_dic_pytorch[key] = value.numpy()

ckpt_file = 'synthesis_19.pdparams'
state_dict = paddle.load(ckpt_file)
model_dic_paddle = {}
for key, value in state_dict.items():
    model_dic_paddle[key] = value.numpy()


for key, value in model_dic_paddle.items():
    value2 = model_dic_pytorch[key]
    ddd = np.sum((value - value2) ** 2)
    # if ddd > 1.00001:
    #     print(key)
    print('diff=%.6f (%s)' % (ddd, key))

print('==============================================')
print()



# ======================== mapping_ema ========================
print('======================== mapping_ema ========================')
model_dic_pytorch = {}
for key, value in state_dict_pytorch['mapping_ema'].items():
    model_dic_pytorch[key] = value.numpy()

ckpt_file = 'mapping_ema_19.pdparams'
state_dict = paddle.load(ckpt_file)
model_dic_paddle = {}
for key, value in state_dict.items():
    model_dic_paddle[key] = value.numpy()


for key, value in model_dic_paddle.items():
    value2 = model_dic_pytorch[key]
    ddd = np.sum((value - value2) ** 2)
    # if ddd > 1.00001:
    #     print(key)
    print('diff=%.6f (%s)' % (ddd, key))

print('==============================================')
print()



# ======================== mapping ========================
print('======================== mapping ========================')
model_dic_pytorch = {}
for key, value in state_dict_pytorch['mapping'].items():
    model_dic_pytorch[key] = value.numpy()

ckpt_file = 'mapping_19.pdparams'
state_dict = paddle.load(ckpt_file)
model_dic_paddle = {}
for key, value in state_dict.items():
    model_dic_paddle[key] = value.numpy()


for key, value in model_dic_paddle.items():
    value2 = model_dic_pytorch[key]
    ddd = np.sum((value - value2) ** 2)
    # if ddd > 1.00001:
    #     print(key)
    print('diff=%.6f (%s)' % (ddd, key))

print('==============================================')
print()









