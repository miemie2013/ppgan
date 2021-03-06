
import torch
import numpy as np
from training.networks import ToRGBLayer


x_shape = [1, 512, 4, 4]
w_shape = [1, 512]
in_channels = 512
out_channels = 3
w_dim = 512
kernel_size = 1
conv_clamp = 256
channels_last = False

x_shape = [1, 512, 8, 8]
w_shape = [1, 512]
in_channels = 512
out_channels = 3
w_dim = 512
kernel_size = 1
conv_clamp = 256
channels_last = False

x_shape = [1, 512, 16, 16]
w_shape = [1, 512]
in_channels = 512
out_channels = 3
w_dim = 512
kernel_size = 1
conv_clamp = 256
channels_last = False

x_shape = [1, 512, 32, 32]
w_shape = [1, 512]
in_channels = 512
out_channels = 3
w_dim = 512
kernel_size = 1
conv_clamp = 256
channels_last = False

x_shape = [1, 512, 64, 64]
w_shape = [1, 512]
in_channels = 512
out_channels = 3
w_dim = 512
kernel_size = 1
conv_clamp = 256
channels_last = False

x_shape = [1, 256, 128, 128]
w_shape = [1, 512]
in_channels = 256
out_channels = 3
w_dim = 512
kernel_size = 1
conv_clamp = 256
channels_last = False

x_shape = [1, 128, 256, 256]
w_shape = [1, 512]
in_channels = 128
out_channels = 3
w_dim = 512
kernel_size = 1
conv_clamp = 256
channels_last = False

x_shape = [1, 64, 512, 512]
w_shape = [1, 512]
in_channels = 64
out_channels = 3
w_dim = 512
kernel_size = 1
conv_clamp = None
channels_last = False







batch_size = 2
fused_modconv = False
lr = 0.0001

model = ToRGBLayer(in_channels, out_channels, w_dim, kernel_size, conv_clamp, channels_last)
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
torch.save(model.state_dict(), "51.pth")

dic = {}
for batch_idx in range(8):
    optimizer.zero_grad(set_to_none=True)

    w_shape[0] = batch_size
    x_shape[0] = batch_size
    ws = torch.randn(w_shape)
    ws.requires_grad_(True)
    x = torch.randn(x_shape)
    x.requires_grad_(True)

    y = model(x, ws, fused_modconv=fused_modconv)
    dy_dx = torch.autograd.grad(outputs=[y.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]
    dy_dws = torch.autograd.grad(outputs=[y.sum()], inputs=[ws], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dy_dx'%batch_idx] = dy_dx.cpu().detach().numpy()
    dic['batch_%.3d.dy_dws'%batch_idx] = dy_dws.cpu().detach().numpy()
    dic['batch_%.3d.output'%batch_idx] = y.cpu().detach().numpy()
    dic['batch_%.3d.input0'%batch_idx] = x.cpu().detach().numpy()
    dic['batch_%.3d.input1'%batch_idx] = ws.cpu().detach().numpy()

    # loss = dy_dx.sum() + dy_dws.sum() + y.sum()
    # loss = dy_dx.sum() + y.sum()
    loss = dy_dws.sum() + y.sum()
    # loss = y.sum()
    loss.backward()
    optimizer.step()
np.savez('51', **dic)
print()
