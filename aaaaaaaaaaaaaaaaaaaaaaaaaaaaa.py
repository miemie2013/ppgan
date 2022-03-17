

import torch
import numpy as np




ckpt_file = 'D_00.pth'
state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))

model_dic1 = {}
for key, value in state_dict.items():
    model_dic1[key] = value.data.numpy()


ckpt_file = 'D_19.pth'
state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))

model_dic2 = {}
for key, value in state_dict.items():
    model_dic2[key] = value.data.numpy()


for key, value in model_dic1.items():
    value2 = model_dic2[key]
    ddd = np.sum((value - value2) ** 2)
    if ddd > 110.00001:
        print()
    print('ddd=%.6f' % ddd)

print('==============================================')
print()

ckpt_file = 'G_00.pth'
state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))

model_dic1 = {}
for key, value in state_dict.items():
    model_dic1[key] = value.data.numpy()


ckpt_file = 'G_19.pth'
state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))

model_dic2 = {}
for key, value in state_dict.items():
    model_dic2[key] = value.data.numpy()


for key, value in model_dic1.items():
    value2 = model_dic2[key]
    ddd = np.sum((value - value2) ** 2)
    if ddd > 0.00001:
        print()
    print('ddd=%.6f' % ddd)




print()









