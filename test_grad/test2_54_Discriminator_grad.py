
import torch
import numpy as np
from training.networks import Discriminator


c_dim = 0
w_dim = 512
# img_resolution = 512
# img_resolution = 128
img_resolution = 32
img_channels = 3
channel_base = 32768
channel_max = 512
num_fp16_res = 4
conv_clamp = 256
epilogue_kwargs = dict(
    mbstd_group_size=8,
)



batch_size = 2
x_shape = [batch_size, img_channels, img_resolution, img_resolution]
lr = 0.0001

model = Discriminator(c_dim=c_dim,
                                            img_resolution=img_resolution,
                                            img_channels=img_channels,
                                            channel_base=channel_base,
                                            channel_max=channel_max,
                                            num_fp16_res=num_fp16_res,
                                            conv_clamp=conv_clamp,
                                            block_kwargs={},
                                            mapping_kwargs={},
                                            epilogue_kwargs=epilogue_kwargs,
                                            )
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
torch.save(model.state_dict(), "54.pth")

dic = {}
for batch_idx in range(8):
    optimizer.zero_grad(set_to_none=True)

    x = torch.randn(x_shape)
    x.requires_grad_(True)

    y = model(x, None)
    dy_dx = torch.autograd.grad(outputs=[y.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dy_dx'%batch_idx] = dy_dx.cpu().detach().numpy()
    dic['batch_%.3d.output'%batch_idx] = y.cpu().detach().numpy()
    dic['batch_%.3d.input0'%batch_idx] = x.cpu().detach().numpy()

    loss = dy_dx.sum() + y.sum()
    # loss = y.sum()
    loss.backward()
    optimizer.step()
np.savez('54', **dic)
print()
