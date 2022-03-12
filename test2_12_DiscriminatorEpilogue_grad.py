
import torch
import numpy as np
from training.networks import DiscriminatorEpilogue


in_channels = 512
cmap_dim = 0
resolution = 4
img_channels = 3
architecture = 'resnet'
mbstd_group_size = None
mbstd_num_channels = 1




x_shape = [2, in_channels, 4, 4]
lr = 0.0001

model = DiscriminatorEpilogue(in_channels, cmap_dim, resolution, img_channels, architecture, mbstd_group_size, mbstd_num_channels)
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
torch.save(model.state_dict(), "12.pth")

dic = {}
for batch_idx in range(8):
    optimizer.zero_grad(set_to_none=True)

    x = torch.randn(x_shape)
    x.requires_grad_(True)

    y = model(x, None, None)
    dy_dx = torch.autograd.grad(outputs=[y.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dy_dx'%batch_idx] = dy_dx.cpu().detach().numpy()
    dic['batch_%.3d.output'%batch_idx] = y.cpu().detach().numpy()
    dic['batch_%.3d.input0'%batch_idx] = x.cpu().detach().numpy()

    loss = dy_dx.sum() + y.sum()
    # loss = y.sum()
    loss.backward()
    optimizer.step()
np.savez('12', **dic)
print()
