import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np
from megengine.autodiff import GradManager
from meg_networks import FullyConnectedLayer



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
optimizer = mge.optimizer.SGD(model.parameters(), lr=lr, momentum=0.9)
# model.set_state_dict(paddle.load("pytorch_fullyConnectedLayer.pdparams"))


dic2 = np.load('01_fullyConnectedLayer_grad.npz')
for batch_idx in range(20):
    print('======================== batch_%.3d ========================'%batch_idx)
    optimizer.zero_grad()
    ws = dic2['batch_%.3d.input'%batch_idx]
    styles_pytorch = dic2['batch_%.3d.output'%batch_idx]
    dstyles2_dws_pytorch = dic2['batch_%.3d.dstyles2_dws'%batch_idx]
    ws = mge.tensor(ws)
    ws.stop_gradient = False


    with GradManager() as gm:
        gm.attach(ws)

        styles = model(ws)
        styles2 = F.sigmoid(styles)

        dstyles2dstyles2 = F.ones_like(styles2)
        gm.backward(styles2, dstyles2dstyles2)  # doesn't need x, already known via attach()
        dstyles2_dws = ws.grad


        aaaaaa = styles.numpy()
        ddd = np.sum((styles_pytorch - aaaaaa) ** 2)
        print('ddd=%.6f' % ddd)

        aaaaaa = dstyles2_dws.numpy()
        ddd = np.sum((dstyles2_dws_pytorch - aaaaaa) ** 2)
        print('ddd=%.6f' % ddd)


        loss = dstyles2_dws.sum() + styles2.sum()
        gm.backward(loss)
    optimizer.step()
print()
