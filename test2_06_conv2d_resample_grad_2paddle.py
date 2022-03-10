
import paddle
import torch
import numpy as np
import paddle.nn.functional as F

from ppgan.models.generators.generator_styleganv2ada import conv2d_resample, conv2d_resample_grad


class Model(paddle.nn.Layer):
    def __init__(self, w_shape):
        super().__init__()
        self.weight = self.create_parameter(w_shape, default_initializer=paddle.nn.initializer.Normal())

    def forward(self, x, f, up, down, padding, groups, flip_weight, flip_filter):
        y = conv2d_resample(x, self.weight, f=f, up=up, down=down, padding=padding, groups=groups, flip_weight=flip_weight, flip_filter=flip_filter)
        return y


ckpt_file = 'model.pth'
save_name = 'model.pdparams'
state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))

model = Model(state_dict['weight'].shape)
model.train()

model_std = model.state_dict()


def copy(name, w, std):
    value2 = paddle.to_tensor(w)
    value = std[name]
    value = value * 0 + value2
    std[name] = value


state_dict2 = {}
for key, value in state_dict.items():
    state_dict2[key] = value.data.numpy()


name2 = 'weight'
copy(name2, state_dict2[name2], model_std)

model.set_state_dict(model_std)
paddle.save(model_std, save_name)

print()
