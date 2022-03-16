
import torch
import numpy as np
from training.networks import SynthesisNetwork


w_dim = 512
# img_resolution = 512
img_resolution = 128
# img_resolution = 32
img_channels = 3
channel_base = 32768
channel_max = 512
num_fp16_res = 4
conv_clamp = 256




# 需要强制设置SynthesisLayer的self.use_noise = False


synthesis_cfg = dict(
    w_dim=w_dim,
    img_resolution=img_resolution,
    img_channels=img_channels,
    channel_base=channel_base,
    channel_max=channel_max,
    num_fp16_res=num_fp16_res,
    conv_clamp=conv_clamp,
)

batch_size = 1
w_shape = [batch_size, w_dim]
lr = 0.0001

model = SynthesisNetwork(**synthesis_cfg)
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
torch.save(model.state_dict(), "53.pth")

dic = {}
for batch_idx in range(8):
    optimizer.zero_grad(set_to_none=True)

    w = torch.randn(w_shape)
    ws = w.unsqueeze(1).repeat([1, model.num_ws, 1])
    ws.requires_grad_(True)

    y = model(ws)
    dy_dws = torch.autograd.grad(outputs=[y.sum()], inputs=[ws], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dy_dws'%batch_idx] = dy_dws.cpu().detach().numpy()
    dic['batch_%.3d.output'%batch_idx] = y.cpu().detach().numpy()
    dic['batch_%.3d.input1'%batch_idx] = ws.cpu().detach().numpy()

    # loss = dy_dws.sum() + y.sum()
    loss = y.sum()
    loss.backward()
    optimizer.step()
np.savez('53', **dic)
print()
