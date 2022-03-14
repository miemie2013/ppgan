
import torch
import numpy as np

import torch.nn.functional as F







dic = {}
for batch_idx in range(8):
    x = torch.randn([4, 512, 4, 4])
    x.requires_grad_(True)

    # N, C, H, W = x.shape
    # G = N
    # F = 512
    # c = C // F

    # N, C, H, W = x.shape
    # G = N
    # F = 256
    # c = C // F

    # N, C, H, W = x.shape
    # G = 1
    # F = 512
    # c = C // F

    # N, C, H, W = x.shape
    # G = 1
    # F = 256
    # c = C // F

    N, C, H, W = x.shape
    G = 2
    F = 256
    c = C // F

    y = x.reshape(G, -1, F, c, H, W)  # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
    y = y - y.mean(dim=0)  # [GnFcHW] Subtract mean over group.
    y = y.square().mean(dim=0)  # [nFcHW]  Calc variance over group.
    y = (y + 1e-8).sqrt()  # [nFcHW]  Calc stddev over group.
    y = y.mean(dim=[2, 3, 4])  # [nF]     Take average over channels and pixels.
    y = y.reshape(-1, F, 1, 1)  # [nF11]   Add missing dimensions.
    y = y.repeat(G, 1, H, W)  # [NFHW]   Replicate over group and pixels.
    out = torch.cat([x, y], dim=1)  # [NCHW]   Append to input as new channels.

    dloss_dx = torch.autograd.grad(outputs=[out.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dloss_dx'%batch_idx] = dloss_dx.cpu().detach().numpy()
    dic['batch_%.3d.out'%batch_idx] = out.cpu().detach().numpy()
    dic['batch_%.3d.x'%batch_idx] = x.cpu().detach().numpy()
np.savez('10_grad', **dic)
print()
