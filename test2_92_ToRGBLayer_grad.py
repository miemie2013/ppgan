
import torch
import numpy as np
from training.networks import ToRGBLayer


w_dim = 512
out_channels = 512
img_channels = 3
conv_clamp = 256
channels_last = False
fused_modconv = False
batch_size = 2
lr = 0.0001

model = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                   conv_clamp=conv_clamp, channels_last=channels_last)
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
torch.save(model.state_dict(), "pytorch_toRGBLayer.pth")

dic = {}
for batch_idx in range(20):
    optimizer.zero_grad(set_to_none=True)

    ws = torch.randn([batch_size, 512])
    ws.requires_grad_(True)
    x = torch.randn([batch_size, 512, 16, 16])
    x.requires_grad_(True)

    y = model(x, ws, fused_modconv=fused_modconv)
    dy_dx = torch.autograd.grad(outputs=[y.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]
    dy_dws = torch.autograd.grad(outputs=[y.sum()], inputs=[ws], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dy_dx'%batch_idx] = dy_dx.cpu().detach().numpy()
    dic['batch_%.3d.dy_dws'%batch_idx] = dy_dws.cpu().detach().numpy()
    dic['batch_%.3d.output'%batch_idx] = y.cpu().detach().numpy()
    dic['batch_%.3d.input0'%batch_idx] = x.cpu().detach().numpy()
    dic['batch_%.3d.input1'%batch_idx] = ws.cpu().detach().numpy()

    loss = dy_dx.sum() + dy_dws.sum() + y.sum()
    loss.backward()
    optimizer.step()
np.savez('02_toRGBLayer_grad', **dic)
print()
