
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

    x_mul_styles = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
    rrrrrrrrrr = torch.sigmoid(x_mul_styles)
    out = rrrrrrrrrr * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
    loss = torch.square(out)

    dloss_dstyles = torch.autograd.grad(outputs=[loss.sum()], inputs=[styles], create_graph=True, only_inputs=True)[0]
    dloss_dx = torch.autograd.grad(outputs=[loss.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dloss_dstyles'%batch_idx] = dloss_dstyles.cpu().detach().numpy()
    dic['batch_%.3d.dloss_dx'%batch_idx] = dloss_dx.cpu().detach().numpy()
    dic['batch_%.3d.out'%batch_idx] = out.cpu().detach().numpy()
    dic['batch_%.3d.styles'%batch_idx] = styles.cpu().detach().numpy()
    dic['batch_%.3d.weight'%batch_idx] = weight.cpu().detach().numpy()
    dic['batch_%.3d.x'%batch_idx] = x.cpu().detach().numpy()
np.savez('10_grad', **dic)
print()
