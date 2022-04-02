

import torch
import paddle
import numpy as np

from ppgan.metrics.inception import InceptionV3

'''
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt


'''


inception_pytorch = torch.load("inception-2015-12-05.pt", map_location="cpu")
inception_dic_pytorch = inception_pytorch.state_dict()

for key, value in inception_dic_pytorch.items():
    print(key)
    # print('diff=%.6f (%s)' % (ddd, key))

x = torch.randn([2, 3, 512, 512])
y = inception_pytorch(x)

# build inceptionv3
dims = 2048
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
inceptionv3_model = InceptionV3([block_idx], normalize_input=False)

inception_dic_paddle = inceptionv3_model.state_dict()


print()



