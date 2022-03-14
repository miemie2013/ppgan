
import paddle
import numpy as np
import paddle.nn.functional as F

from ppgan.models.generators.generator_styleganv2ada import upfirdn2d, upfirdn2d_grad


dic2 = np.load('05_grad.npz')
for batch_idx in range(8):
    print('======================== batch_%.3d ========================'%batch_idx)
    x_shape = [1, 512, 9, 9]
    f_shape = [4, 4]
    up = 1
    down = 1
    padding = [1, 1, 1, 1]
    flip_filter = False
    gain = 4

    # x_shape = [1, 3, 4, 4]
    # f_shape = [4, 4]
    # up = 2
    # down = 1
    # padding = [2, 1, 2, 1]
    # flip_filter = False
    # gain = 4

    # x_shape = [1, 512, 17, 17]
    # f_shape = [4, 4]
    # up = 1
    # down = 1
    # padding = [1, 1, 1, 1]
    # flip_filter = False
    # gain = 4

    # x_shape = [1, 3, 8, 8]
    # f_shape = [4, 4]
    # up = 2
    # down = 1
    # padding = [2, 1, 2, 1]
    # flip_filter = False
    # gain = 4

    # x_shape = [1, 512, 33, 33]
    # f_shape = [4, 4]
    # up = 1
    # down = 1
    # padding = [1, 1, 1, 1]
    # flip_filter = False
    # gain = 4

    # x_shape = [1, 3, 16, 16]
    # f_shape = [4, 4]
    # up = 2
    # down = 1
    # padding = [2, 1, 2, 1]
    # flip_filter = False
    # gain = 4

    # x_shape = [1, 512, 65, 65]
    # f_shape = [4, 4]
    # up = 1
    # down = 1
    # padding = [1, 1, 1, 1]
    # flip_filter = False
    # gain = 4

    # x_shape = [1, 3, 32, 32]
    # f_shape = [4, 4]
    # up = 2
    # down = 1
    # padding = [2, 1, 2, 1]
    # flip_filter = False
    # gain = 4

    # x_shape = [1, 256, 129, 129]
    # f_shape = [4, 4]
    # up = 1
    # down = 1
    # padding = [1, 1, 1, 1]
    # flip_filter = False
    # gain = 4

    # x_shape = [1, 3, 64, 64]
    # f_shape = [4, 4]
    # up = 2
    # down = 1
    # padding = [2, 1, 2, 1]
    # flip_filter = False
    # gain = 4

    # x_shape = [1, 128, 257, 257]
    # f_shape = [4, 4]
    # up = 1
    # down = 1
    # padding = [1, 1, 1, 1]
    # flip_filter = False
    # gain = 4

    # x_shape = [1, 3, 128, 128]
    # f_shape = [4, 4]
    # up = 2
    # down = 1
    # padding = [2, 1, 2, 1]
    # flip_filter = False
    # gain = 4

    # x_shape = [1, 64, 513, 513]
    # f_shape = [4, 4]
    # up = 1
    # down = 1
    # padding = [1, 1, 1, 1]
    # flip_filter = False
    # gain = 4

    # x_shape = [1, 3, 256, 256]
    # f_shape = [4, 4]
    # up = 2
    # down = 1
    # padding = [2, 1, 2, 1]
    # flip_filter = False
    # gain = 4

    x_shape = [1, 512, 8, 8]
    f_shape = [4, 4]
    up = 1
    down = 2
    padding = [1, 1, 1, 1]
    flip_filter = False
    gain = 1




    dy_dx_pytorch = dic2['batch_%.3d.dy_dx'%batch_idx]
    y_pytorch = dic2['batch_%.3d.y'%batch_idx]
    f = dic2['batch_%.3d.f'%batch_idx]
    x = dic2['batch_%.3d.x'%batch_idx]
    x = paddle.to_tensor(x)
    x.stop_gradient = False
    f = paddle.to_tensor(f)

    y = upfirdn2d(x, f, up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain)
    # dy_dx = paddle.grad(outputs=[y.sum()], inputs=[x], create_graph=True)[0]
    dysum_dy = paddle.ones(y.shape, dtype=paddle.float32)
    dy_dx = upfirdn2d_grad(dysum_dy, x, f, up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain)

    aaaaaa = y.numpy()
    ddd = np.sum((y_pytorch - aaaaaa) ** 2)
    print('ddd=%.6f' % ddd)

    dy_dx_paddle = dy_dx.numpy()
    ddd = np.sum((dy_dx_pytorch - dy_dx_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    # aaaaaa = dy_dw.numpy()
    # ddd = np.sum((dy_dw_pytorch - aaaaaa) ** 2)
    # print('ddd=%.6f' % ddd)
print()
