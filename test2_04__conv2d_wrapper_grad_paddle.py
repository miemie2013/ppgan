
import paddle
import numpy as np
import paddle.nn.functional as F

from ppgan.models.generators.generator_styleganv2ada import _conv2d_wrapper, _conv2d_wrapper_grad


dic2 = np.load('04_grad.npz')
for batch_idx in range(8):
    print('======================== batch_%.3d ========================'%batch_idx)
    x_shape = [1, 8, 1, 1]
    w_shape = [4, 8, 1, 1]
    stride = 1
    padding = [0, 0]
    groups = 1
    transpose = False
    flip_weight = False

    # x_shape = [1, 8, 1, 1]
    # w_shape = [16, 8, 1, 1]
    # stride = 1
    # padding = [0, 0]
    # groups = 1
    # transpose = False
    # flip_weight = True

    # x_shape = [1, 512, 4, 4]
    # w_shape = [512, 512, 3, 3]
    # stride = 1
    # padding = [1, 1]
    # groups = 1
    # transpose = False
    # flip_weight = True

    # x_shape = [1, 512, 4, 4]
    # w_shape = [3, 512, 1, 1]
    # stride = 1
    # padding = [0, 0]
    # groups = 1
    # transpose = False
    # flip_weight = True

    # x_shape = [1, 512, 4, 4]
    # w_shape = [512, 512, 3, 3]
    # stride = 2
    # padding = [0, 0]
    # groups = 1
    # transpose = True
    # flip_weight = True

    x_shape = [1, 512, 8, 8]
    w_shape = [512, 512, 3, 3]
    stride = 1
    padding = [1, 1]
    groups = 1
    transpose = False
    flip_weight = True

    # x_shape = [1, 512, 8, 8]
    # w_shape = [3, 512, 1, 1]
    # stride = 1
    # padding = [0, 0]
    # groups = 1
    # transpose = False
    # flip_weight = True

    # x_shape = [1, 512, 8, 8]
    # w_shape = [512, 512, 3, 3]
    # stride = 2
    # padding = [0, 0]
    # groups = 1
    # transpose = True
    # flip_weight = True

    # x_shape = [1, 512, 16, 16]
    # w_shape = [512, 512, 3, 3]
    # stride = 1
    # padding = [1, 1]
    # groups = 1
    # transpose = False
    # flip_weight = True

    # x_shape = [1, 512, 16, 16]
    # w_shape = [3, 512, 1, 1]
    # stride = 1
    # padding = [0, 0]
    # groups = 1
    # transpose = False
    # flip_weight = True

    # x_shape = [1, 512, 16, 16]
    # w_shape = [512, 512, 3, 3]
    # stride = 2
    # padding = [0, 0]
    # groups = 1
    # transpose = True
    # flip_weight = True

    # x_shape = [1, 512, 32, 32]
    # w_shape = [512, 512, 3, 3]
    # stride = 1
    # padding = [1, 1]
    # groups = 1
    # transpose = False
    # flip_weight = True

    # x_shape = [1, 512, 32, 32]
    # w_shape = [3, 512, 1, 1]
    # stride = 1
    # padding = [0, 0]
    # groups = 1
    # transpose = False
    # flip_weight = True

    # x_shape = [1, 512, 32, 32]
    # w_shape = [512, 512, 3, 3]
    # stride = 2
    # padding = [0, 0]
    # groups = 1
    # transpose = True
    # flip_weight = True

    # x_shape = [1, 512, 64, 64]
    # w_shape = [512, 512, 3, 3]
    # stride = 1
    # padding = [1, 1]
    # groups = 1
    # transpose = False
    # flip_weight = True

    # x_shape = [1, 512, 64, 64]
    # w_shape = [3, 512, 1, 1]
    # stride = 1
    # padding = [0, 0]
    # groups = 1
    # transpose = False
    # flip_weight = True

    # x_shape = [1, 512, 64, 64]
    # w_shape = [512, 256, 3, 3]
    # stride = 2
    # padding = [0, 0]
    # groups = 1
    # transpose = True
    # flip_weight = True

    # x_shape = [1, 256, 128, 128]
    # w_shape = [256, 256, 3, 3]
    # stride = 1
    # padding = [1, 1]
    # groups = 1
    # transpose = False
    # flip_weight = True

    # x_shape = [1, 256, 128, 128]
    # w_shape = [3, 256, 1, 1]
    # stride = 1
    # padding = [0, 0]
    # groups = 1
    # transpose = False
    # flip_weight = True

    # x_shape = [1, 256, 128, 128]
    # w_shape = [256, 128, 3, 3]
    # stride = 2
    # padding = [0, 0]
    # groups = 1
    # transpose = True
    # flip_weight = True

    # x_shape = [1, 128, 256, 256]
    # w_shape = [128, 128, 3, 3]
    # stride = 1
    # padding = [1, 1]
    # groups = 1
    # transpose = False
    # flip_weight = True

    # x_shape = [1, 128, 256, 256]
    # w_shape = [3, 128, 1, 1]
    # stride = 1
    # padding = [0, 0]
    # groups = 1
    # transpose = False
    # flip_weight = True

    # x_shape = [1, 128, 256, 256]
    # w_shape = [128, 64, 3, 3]
    # stride = 2
    # padding = [0, 0]
    # groups = 1
    # transpose = True
    # flip_weight = True

    # x_shape = [1, 64, 512, 512]
    # w_shape = [64, 64, 3, 3]
    # stride = 1
    # padding = [1, 1]
    # groups = 1
    # transpose = False
    # flip_weight = True

    # x_shape = [1, 64, 512, 512]
    # w_shape = [3, 64, 1, 1]
    # stride = 1
    # padding = [0, 0]
    # groups = 1
    # transpose = False
    # flip_weight = True




    dy_dx_pytorch = dic2['batch_%.3d.dy_dx'%batch_idx]
    dy_dw_pytorch = dic2['batch_%.3d.dy_dw'%batch_idx]
    y_pytorch = dic2['batch_%.3d.y'%batch_idx]
    w = dic2['batch_%.3d.w'%batch_idx]
    x = dic2['batch_%.3d.x'%batch_idx]
    x = paddle.to_tensor(x)
    x.stop_gradient = False
    w = paddle.to_tensor(w)
    w.stop_gradient = False

    y = _conv2d_wrapper(x=x, w=w, stride=stride, padding=padding, groups=groups, transpose=transpose, flip_weight=flip_weight)

    # dy_dx = paddle.grad(outputs=[y.sum()], inputs=[x], create_graph=True)[0]
    # dy_dw = paddle.grad(outputs=[y.sum()], inputs=[w], create_graph=True)[0]
    dysum_dy = paddle.ones(y.shape, dtype=paddle.float32)
    # dy_dx, dy_dw = _conv2d_wrapper_grad(dysum_dy, x=x, w=w, stride=stride, padding=padding, groups=groups, transpose=transpose, flip_weight=flip_weight)
    dy_dx = _conv2d_wrapper_grad(dysum_dy, x=x, w=w, stride=stride, padding=padding, groups=groups, transpose=transpose, flip_weight=flip_weight)

    aaaaaa = y.numpy()
    ddd = np.sum((y_pytorch - aaaaaa) ** 2)
    print('ddd=%.6f' % ddd)

    dy_dx_paddle = dy_dx.numpy()
    ddd = np.sum((dy_dx_pytorch - dy_dx_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    # dy_dw_paddle = dy_dw.numpy()
    # ddd = np.sum((dy_dw_pytorch - dy_dw_paddle) ** 2)
    # print('ddd=%.6f' % ddd)
print()
