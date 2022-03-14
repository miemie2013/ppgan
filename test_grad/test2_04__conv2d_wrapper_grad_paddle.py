
import paddle
import numpy as np
import paddle.nn.functional as F

from ppgan.models.generators.generator_styleganv2ada import _conv2d_wrapper, _conv2d_wrapper_grad


class Model(paddle.nn.Layer):
    def __init__(self, w_shape):
        super().__init__()
        self.weight = self.create_parameter(w_shape, default_initializer=paddle.nn.initializer.Normal())

    def forward(self, x, stride, padding, groups, transpose, flip_weight):
        y = _conv2d_wrapper(x=x, w=self.weight, stride=stride, padding=padding, groups=groups, transpose=transpose, flip_weight=flip_weight)
        return y


lr = 0.0001
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

    x_shape = [1, 8, 1, 1]
    w_shape = [16, 8, 1, 1]
    stride = 1
    padding = [0, 0]
    groups = 1
    transpose = False
    flip_weight = True

    x_shape = [1, 8, 1, 1]
    w_shape = [16, 4, 1, 1]
    stride = 1
    padding = [0, 0]
    groups = 2
    transpose = False
    flip_weight = False

    x_shape = [1, 512, 4, 4]
    w_shape = [512, 512, 3, 3]
    stride = 1
    padding = [1, 1]
    groups = 1
    transpose = False
    flip_weight = True

    x_shape = [1, 512, 4, 4]
    w_shape = [3, 512, 1, 1]
    stride = 1
    padding = [0, 0]
    groups = 1
    transpose = False
    flip_weight = True

    x_shape = [1, 512, 4, 4]
    w_shape = [512, 512, 3, 3]
    stride = 2
    padding = [0, 0]
    groups = 1
    transpose = True
    flip_weight = True

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

    x_shape = [1, 256, 128, 128]
    w_shape = [256, 128, 3, 3]
    stride = 2
    padding = [0, 0]
    groups = 1
    transpose = True
    flip_weight = True

    x_shape = [1, 128, 256, 256]
    w_shape = [128, 128, 3, 3]
    stride = 1
    padding = [1, 1]
    groups = 1
    transpose = False
    flip_weight = True

    x_shape = [1, 128, 256, 256]
    w_shape = [3, 128, 1, 1]
    stride = 1
    padding = [0, 0]
    groups = 1
    transpose = False
    flip_weight = True

    x_shape = [1, 128, 256, 256]
    w_shape = [128, 64, 3, 3]
    stride = 2
    padding = [0, 0]
    groups = 1
    transpose = True
    flip_weight = True

    x_shape = [1, 64, 512, 512]
    w_shape = [64, 64, 3, 3]
    stride = 1
    padding = [1, 1]
    groups = 1
    transpose = False
    flip_weight = True

    x_shape = [1, 64, 512, 512]
    w_shape = [3, 64, 1, 1]
    stride = 1
    padding = [0, 0]
    groups = 1
    transpose = False
    flip_weight = True

    x_shape = [1, 512, 9, 9]
    w_shape = [512, 512, 3, 3]
    stride = 2
    padding = 0
    groups = 1
    transpose = False
    flip_weight = True




    dloss_dx_pytorch = dic2['batch_%.3d.dloss_dx'%batch_idx]
    dloss_dw_pytorch = dic2['batch_%.3d.dloss_dw'%batch_idx]
    y_pytorch = dic2['batch_%.3d.y'%batch_idx]
    x = dic2['batch_%.3d.x'%batch_idx]
    x = paddle.to_tensor(x)
    x.stop_gradient = False

    if batch_idx == 0:
        model = Model(w_shape)
        model.train()
        optimizer = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=lr, momentum=0.9)
        model.set_state_dict(paddle.load("model.pdparams"))
    optimizer.clear_gradients()


    y = model(x=x, stride=stride, padding=padding, groups=groups, transpose=transpose, flip_weight=flip_weight)

    # loss = paddle.nn.functional.sigmoid(y)
    # loss = paddle.tanh(y)
    loss = y

    # 总结：和梯度相乘的临时前向张量loss_clone一定要是本尊，不能加.detach()
    # 正确的
    loss_clone = loss
    # loss_clone = loss.clone()

    # 错误的
    # loss_clone = loss.detach()
    # loss_clone = loss.numpy()
    # loss_clone = paddle.to_tensor(loss_clone)
    # loss_clone = paddle.ones(loss.shape, dtype=paddle.float32)
    # loss_clone.set_value(loss)

    # dloss_dx = paddle.grad(outputs=[loss.sum()], inputs=[x], create_graph=True)[0]
    # dloss_dw = paddle.grad(outputs=[loss.sum()], inputs=[model.weight], create_graph=True)[0]
    # dloss_dx, dloss_dw = paddle.grad(outputs=[loss.sum()], inputs=[x, model.weight], create_graph=True)
    dloss_dloss = paddle.ones(loss.shape, dtype=paddle.float32)
    # dloss_dy = dloss_dloss * loss_clone * (1.0 - loss_clone)   # 总结：和梯度相乘的临时前向张量loss_clone一定要是本尊，不能加.detach()
    dloss_dy = dloss_dloss
    dloss_dx, dloss_dw = _conv2d_wrapper_grad(dloss_dy, x=x, w=model.weight, stride=stride, padding=padding, groups=groups, transpose=transpose, flip_weight=flip_weight)

    y_paddle = y.numpy()
    ddd = np.sum((y_pytorch - y_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    dloss_dx_paddle = dloss_dx.numpy()
    ddd = np.sum((dloss_dx_pytorch - dloss_dx_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    dloss_dw_paddle = dloss_dw.numpy()
    ddd = np.sum((dloss_dw_pytorch - dloss_dw_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    # loss = loss.sum() + dloss_dx.sum() + dloss_dw.sum()   # 使用paddle.grad()时，和pytorch获得不一样的结果xxx
    loss = loss.sum() + dloss_dx.sum()   # 使用paddle.grad()时，和pytorch获得不一样的结果xxx
    # loss = loss.sum() + dloss_dw.sum()     # 使用paddle.grad()时，可以和pytorch获得一样的结果。
    loss.backward()
    optimizer.step()
    optimizer.clear_gradients()
print()
