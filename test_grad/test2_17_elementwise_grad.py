
import torch
import numpy as np

import torch.nn.functional as F







dic = {}
for batch_idx in range(8):
    styles = torch.randn([2, 512, 512])
    styles.requires_grad_(True)
    batch_size = 2

    w = styles * 11
    dcoefs = w.sum(dim=[2,])  # [N, out_C]
    out = w * dcoefs.reshape((batch_size, -1, 1))  # [N, out_C, in_C]
    loss = torch.square(out)

    dloss_dstyles = torch.autograd.grad(outputs=[loss.sum()], inputs=[styles], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dloss_dstyles'%batch_idx] = dloss_dstyles.cpu().detach().numpy()
    dic['batch_%.3d.out'%batch_idx] = out.cpu().detach().numpy()
    dic['batch_%.3d.styles'%batch_idx] = styles.cpu().detach().numpy()
np.savez('17_grad', **dic)
print()
