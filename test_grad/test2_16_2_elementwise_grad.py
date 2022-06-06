
import torch
import numpy as np

import torch.nn.functional as F







dic = {}
for batch_idx in range(8):
    x = torch.randn([2, 512, 16, 16])
    styles = torch.randn([2, 512])
    weight = torch.randn([512, 512, 3, 3])


    weight.requires_grad_(True)
    x.requires_grad_(True)
    styles.requires_grad_(True)
    batch_size = 2

    out_C, in_C, kH, kW = weight.shape

    w = weight.unsqueeze(0) # [1, out_C, in_C, kH, kW]
    w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [N, out_C, in_C, kH, kW]
    dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [N, out_C]
    w = w * dcoefs.reshape((batch_size, -1, 1, 1, 1))  # [N, out_C, in_C, kH, kW]
    out = w
    loss = torch.square(out)

    dloss_dstyles = torch.autograd.grad(outputs=[loss.sum()], inputs=[styles], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dloss_dstyles'%batch_idx] = dloss_dstyles.cpu().detach().numpy()
    dic['batch_%.3d.out'%batch_idx] = out.cpu().detach().numpy()
    dic['batch_%.3d.styles'%batch_idx] = styles.cpu().detach().numpy()
    dic['batch_%.3d.weight'%batch_idx] = weight.cpu().detach().numpy()
    dic['batch_%.3d.x'%batch_idx] = x.cpu().detach().numpy()
np.savez('16_grad', **dic)
print()
