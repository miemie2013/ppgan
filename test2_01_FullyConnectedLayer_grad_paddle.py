
import paddle
import paddle.nn.functional as F
import numpy as np
from ppgan.models.generators.generator_styleganv2ada import FullyConnectedLayer

batch_size = 2
in_channels = 512
w_dim = 512
lr = 0.0001
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
model.train()
optimizer = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=lr, momentum=0.9)
model.set_state_dict(paddle.load("pytorch_fullyConnectedLayer.pdparams"))


dic2 = np.load('01_fullyConnectedLayer_grad.npz')
for batch_idx in range(20):
    print('======================== batch_%.3d ========================'%batch_idx)
    optimizer.clear_gradients()
    ws = dic2['batch_%.3d.input'%batch_idx]
    styles_pytorch = dic2['batch_%.3d.output'%batch_idx]
    dstyles2_dws_pytorch = dic2['batch_%.3d.dstyles2_dws'%batch_idx]
    ws = paddle.to_tensor(ws)
    ws.stop_gradient = False
    styles = model(ws)

    styles2 = F.sigmoid(styles)

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
    dstyles2_dstyles2 = paddle.ones(styles2.shape, dtype=paddle.float32)
    dstyles2_dstyles = dstyles2_dstyles2 * styles2_clone * (1.0 - styles2_clone)  # 总结：和梯度相乘的临时前向张量styles2_clone一定要是本尊，不能加.detach()
    dstyles2_dws = model.grad_layer(dstyles2_dstyles)

    aaaaaa = styles.numpy()
    ddd = np.sum((styles_pytorch - aaaaaa) ** 2)
    print('ddd=%.6f' % ddd)

    aaaaaa = dstyles2_dws.numpy()
    ddd = np.sum((dstyles2_dws_pytorch - aaaaaa) ** 2)
    print('ddd=%.6f' % ddd)

    loss = dstyles2_dws.sum() + styles2.sum()
    loss.backward()
    optimizer.step()
print()
