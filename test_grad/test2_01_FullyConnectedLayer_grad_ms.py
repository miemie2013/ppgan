
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as P
import numpy as np
from ms_networks import FullyConnectedLayer
import mindspore.context as context
context.set_context(device_target="CPU")

batch_size = 2
in_channels = 512
w_dim = 512
lr = 0.1
# activation = 'linear'
# activation = 'lrelu'
# activation = 'relu'
# activation = 'tanh'
activation = 'sigmoid'
# activation = 'elu'
# activation = 'selu'
# activation = 'softplus'
# activation = 'swish'

model = FullyConnectedLayer(w_dim, in_channels, activation=activation, bias_init=1)
# model.train()
optimizer = nn.Momentum(params=model.trainable_params(), learning_rate=lr, momentum=0.9)
ms.load_param_into_net(model, ms.load_checkpoint("pytorch_fullyConnectedLayer.ckpt"))
model = ms.Model(model, optimizer)


dic2 = np.load('01_fullyConnectedLayer_grad.npz')
for batch_idx in range(20):
    print('======================== batch_%.3d ========================'%batch_idx)
    # optimizer.clear_gradients()
    ws = dic2['batch_%.3d.input'%batch_idx]
    styles_pytorch = dic2['batch_%.3d.output'%batch_idx]
    dstyles2_dws_pytorch = dic2['batch_%.3d.dstyles2_dws'%batch_idx]
    ws = ms.Tensor(ws)
    ws.requ = False
    styles = model(ws)

    styles2 = P.Sigmoid()(styles)

    # 总结：和梯度相乘的临时前向张量styles2_clone一定要是本尊，不能加.detach()
    # 正确的
    styles2_clone = styles2
    # styles2_clone = styles2.clone()

    # 错误的
    # styles2_clone = styles2.detach()
    # styles2_clone = styles2.numpy()
    # styles2_clone = paddle.to_tensor(styles2_clone)
    # styles2_clone = paddle.ones(styles2.shape, dtype=paddle.float32)
    # styles2_clone.set_value(styles2)


    # dstyles2_dws = paddle.grad(outputs=[styles2.sum()], inputs=[ws], create_graph=True)[0]
    # dstyles2_dstyles2 = paddle.ones(styles2.shape, dtype=paddle.float32)
    # dstyles2_dstyles = dstyles2_dstyles2 * styles2_clone * (1.0 - styles2_clone)  # 总结：和梯度相乘的临时前向张量styles2_clone一定要是本尊，不能加.detach()
    # dstyles2_dws = model.grad_layer(dstyles2_dstyles)

    aaaaaa = styles.numpy()
    ddd = np.sum((styles_pytorch - aaaaaa) ** 2)
    print('ddd=%.6f' % ddd)

    # aaaaaa = dstyles2_dws.numpy()
    # ddd = np.sum((dstyles2_dws_pytorch - aaaaaa) ** 2)
    # print('ddd=%.6f' % ddd)

    # loss = dstyles2_dws.sum() + styles2.sum()
    loss = styles2.sum()
    loss.backward()
    # optimizer.step()
print()
