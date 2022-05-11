




import numpy as np
import torch
import torch.nn.functional as F


groups = 2
num_channels = 512
out_C = 32
x = np.random.normal(size=[2, num_channels, 11, 11])
w = np.random.normal(size=[out_C, num_channels // groups, 4, 4])

x2 = torch.Tensor(x)
w2 = torch.Tensor(w)

y2 = F.conv2d(x2, weight=w2, groups=groups)





import numpy as np
import megengine as mge
import megengine.functional as F



import megengine.module as M
aaaaaa = M.Conv2d(num_channels, out_C, 4, 1, 0, groups=groups, bias=True)


x = mge.tensor(x)
w = mge.tensor(w)

out_c, in_c, kH, kW = w.shape
w = w.reshape((groups, -1, in_c, kH, kW))
y = F.conv2d(x, weight=w, groups=groups)


ddd = np.sum((y2.cpu().detach().numpy() - y.numpy())**2)
print('ddd=%.6f' % ddd)


print()



