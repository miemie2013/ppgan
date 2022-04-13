

import torch
import paddle
import os
import numpy as np
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from ppgan.metrics import inception_pytorch
from ppgan.metrics import inception_paddle

torch.backends.cudnn.benchmark = True  # Improves training speed.
torch.backends.cuda.matmul.allow_tf32 = False  # Allow PyTorch to internally use tf32 for matmul
torch.backends.cudnn.allow_tf32 = False  # Allow PyTorch to internally use tf32 for convolutions



model = torch.load('inception-2015-12-05.pt', map_location=torch.device('cpu'))
std1 = model.state_dict()
save_name = '../inception-2015-12-05.pdparams'
save_name_pth = '../inception-2015-12-05.pth'
model.eval()


model2 = inception_pytorch.Inception_v3()
std2 = model2.state_dict()
model2.eval()

model3 = inception_paddle.Inception_v3()
std3 = model3.state_dict()
model3.eval()


already_used = []


use_gpu = True

gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()


def copy(name, w, std):
    value2 = paddle.to_tensor(w, place=place)
    value = std[name]
    value = value * 0 + value2
    std[name] = value




for key2, value2 in std2.items():
    if '.bn.num_batches_tracked' in key2:
        continue
    if '.bn.weight' in key2:
        continue
    if '.conv.weight' in key2:
        key1 = key2.replace('.conv.weight', '.weight')
        std2[key2] = std1[key1]
        copy(key2, std1[key1].data.numpy(), std3)
    if '.bn.bias' in key2:
        key1 = key2.replace('.bn.bias', '.beta')
        std2[key2] = std1[key1]
        copy(key2, std1[key1].data.numpy(), std3)
    if '.bn.running_mean' in key2:
        key1 = key2.replace('.bn.running_mean', '.mean')
        std2[key2] = std1[key1]
        copy(key2.replace('.running_mean', '._mean'), std1[key1].data.numpy(), std3)
    if '.bn.running_var' in key2:
        key1 = key2.replace('.bn.running_var', '.var')
        std2[key2] = std1[key1]
        copy(key2.replace('.running_var', '._variance'), std1[key1].data.numpy(), std3)

std2['output.weight'] = std1['output.weight']
std2['output.bias'] = std1['output.bias']
copy('output.weight', std1['output.weight'].data.numpy().transpose(1, 0), std3)
copy('output.bias', std1['output.bias'].data.numpy(), std3)

model2.load_state_dict(std2)
model3.set_state_dict(std3)
torch.save(std2, save_name_pth)
paddle.save(std3, save_name)

x_shape = [4, 3, 512, 512]
images = torch.randn(x_shape)
images2 = images.cpu().detach().numpy()
images2 = paddle.to_tensor(images2)

return_features = False
return_features = True

use_fp16 = False
# use_fp16 = True

no_output_bias = False
# no_output_bias = True

code = model.code
print(code)

features = model(images, return_features=return_features, use_fp16=use_fp16, no_output_bias=no_output_bias)
features2 = model2(images, return_features=return_features, use_fp16=use_fp16, no_output_bias=no_output_bias)
features3 = model3(images2, return_features=return_features, use_fp16=use_fp16, no_output_bias=no_output_bias)

# features = model.layers(images)
# features2 = model2.layers(images)
# features3 = model3.layers(images2)


ddd = np.sum((features2.cpu().detach().numpy() - features.cpu().detach().numpy()) ** 2)
print('diff=%.6f (%s)' % (ddd, 'features'))

ddd = np.sum((features3.numpy() - features.cpu().detach().numpy()) ** 2)
print('diff=%.6f (%s)' % (ddd, 'features'))


print()



