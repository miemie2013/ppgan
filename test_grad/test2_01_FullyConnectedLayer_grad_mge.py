import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np
from megengine.autodiff import GradManager
from meg_networks import FullyConnectedLayer, mge_grad



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
model.train()
optimizer = mge.optimizer.SGD(model.parameters(), lr=lr, momentum=0.9)
model.load_state_dict(mge.load("pytorch_fullyConnectedLayer.pkl", map_location="cpu"))


dic2 = np.load('01_fullyConnectedLayer_grad.npz')
for batch_idx in range(20):
    print('======================== batch_%.3d ========================'%batch_idx)
    optimizer.clear_grad()
    ws = dic2['batch_%.3d.input'%batch_idx]
    styles_pytorch = dic2['batch_%.3d.output'%batch_idx]
    dstyles2_dws_pytorch = dic2['batch_%.3d.dstyles2_dws'%batch_idx]
    ws = mge.tensor(ws)
    ws.stop_gradient = False

    gm = GradManager()
    gm2 = GradManager()
    gm.attach([ws])
    gm2.attach([ws])
    gm.attach(model.parameters())
    gm2.attach(model.parameters())


    with gm:
        with gm2:
            styles = model(ws)
            styles2 = F.sigmoid(styles)

            dstyles2_dws = mge_grad(gm2, output=styles2, inputs=[ws], clear_grad_nets=[model])[0]

            styles_mge = styles.numpy()
            ddd = np.sum((styles_pytorch - styles_mge) ** 2)
            print('ddd=%.6f' % ddd)

            dstyles2_dws_mge = dstyles2_dws.numpy()
            ddd = np.sum((dstyles2_dws_pytorch - dstyles2_dws_mge) ** 2)
            print('ddd=%.6f' % ddd)

            loss = dstyles2_dws.sum() + styles2.sum()
            gm.backward(loss)
    optimizer.step()
print()
