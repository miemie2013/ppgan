
import torch
import megengine as mge
import megengine.functional as F
import os
import numpy as np
from meg_networks import FullyConnectedLayer

in_channels = 512
w_dim = 512

# activation = 'linear'
# activation = 'lrelu'
# activation = 'relu'
# activation = 'tanh'
activation = 'sigmoid'
# activation = 'elu'
# activation = 'selu'
# activation = 'softplus'
# activation = 'swish'

model = FullyConnectedLayer(w_dim, in_channels, activation=activation, bias_init=1)
model.train()


def copy(name, w, std):
    value2 = w
    value = std[name]
    value = value * 0 + value2
    std[name] = value

fullyConnectedLayer_std = model.state_dict()

ckpt_file = 'pytorch_fullyConnectedLayer.pth'
save_name = 'pytorch_fullyConnectedLayer.pkl'
state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))


fullyConnectedLayer_dic = {}
for key, value in state_dict.items():
    fullyConnectedLayer_dic[key] = value.data.numpy()

for key in fullyConnectedLayer_dic.keys():
    name2 = key
    w = fullyConnectedLayer_dic[key]
    if '.linear.weight' in key:
        w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
    if '.noise_strength' in key:
        print()
        w = np.reshape(w, [1, ])
    print(key)
    copy(name2, w, fullyConnectedLayer_std)
model.load_state_dict(fullyConnectedLayer_std)

mge.save(fullyConnectedLayer_std, save_name)
print(mge.__version__)

