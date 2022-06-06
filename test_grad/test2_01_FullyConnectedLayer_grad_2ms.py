
import torch
import mindspore as ms
import numpy as np
from ms_networks import FullyConnectedLayer
import mindspore.context as context
context.set_context(device_target="CPU")


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
# model.train()


def copy(name, w, std):
    value = std[name]
    value2 = ms.common.parameter.Parameter(w, name=value.name)
    std[name] = value2

fullyConnectedLayer_std = model.parameters_dict()


ckpt_file = 'pytorch_fullyConnectedLayer.pth'
save_name = 'pytorch_fullyConnectedLayer.ckpt'
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
ms.load_param_into_net(model, fullyConnectedLayer_std)

ms.save_checkpoint(model, save_name)

