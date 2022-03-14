
import torch
import numpy as np
from training.networks import Conv2dLayer


in_channels = 256
out_channels = 256
kernel_size = 3
bias = True
activation = 'lrelu'
up = 1
down = 1
resample_filter = [1, 3, 3, 1]
conv_clamp = 256
channels_last = False
trainable = True




x_shape = [2, in_channels, 64, 64]
lr = 0.0001

model = Conv2dLayer(in_channels, out_channels, kernel_size, bias, activation, up, down, resample_filter, conv_clamp, channels_last, trainable)
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
torch.save(model.state_dict(), "11.pth")

dic = {}
for batch_idx in range(8):
    optimizer.zero_grad(set_to_none=True)

    x = torch.randn(x_shape)
    x.requires_grad_(True)

    y = model(x)
    dy_dx = torch.autograd.grad(outputs=[y.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dy_dx'%batch_idx] = dy_dx.cpu().detach().numpy()
    dic['batch_%.3d.output'%batch_idx] = y.cpu().detach().numpy()
    dic['batch_%.3d.input0'%batch_idx] = x.cpu().detach().numpy()

    loss = dy_dx.sum() + y.sum()
    # loss = y.sum()
    loss.backward()
    optimizer.step()
np.savez('11', **dic)
print()
