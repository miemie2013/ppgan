import paddle
import numpy as np
import paddle.nn.functional as F
from ppgan.models.generators.generator_styleganv2ada import GridSample



model = GridSample(mode='bilinear', padding_mode='zeros', align_corners=False)
# model = GridSample(mode='bilinear', padding_mode='zeros', align_corners=True)


dic2 = np.load('14_grad.npz')
for batch_idx in range(8):
    print('======================== batch_%.3d ========================' % batch_idx)

    dloss_dx_pytorch = dic2['batch_%.3d.dloss_dx' % batch_idx]
    y_pytorch = dic2['batch_%.3d.out' % batch_idx]
    x = dic2['batch_%.3d.x' % batch_idx]
    x = paddle.to_tensor(x)
    x.stop_gradient = False
    grid = dic2['batch_%.3d.grid' % batch_idx]
    grid = paddle.to_tensor(grid)

    y = model(x, grid)


    # dloss_dx = paddle.grad(outputs=[y.sum()], inputs=[x], create_graph=False)[0]
    dloss_dy = paddle.ones(y.shape, dtype=paddle.float32)
    dloss_dx = model.grad_layer(dloss_dy)


    y_paddle = y.numpy()
    ddd = np.mean((y_pytorch - y_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    dloss_dx_paddle = dloss_dx.numpy()
    ddd = np.mean((dloss_dx_pytorch - dloss_dx_paddle) ** 2)
    print('ddd=%.6f' % ddd)
print()
