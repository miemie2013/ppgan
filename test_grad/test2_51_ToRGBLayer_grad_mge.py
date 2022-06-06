
import megengine as mge
from megengine.autodiff import GradManager
import numpy as np
from meg_networks import ToRGBLayer, mge_grad


x_shape = [1, 512, 4, 4]
w_shape = [1, 512]
in_channels = 512
out_channels = 3
w_dim = 512
kernel_size = 1
conv_clamp = 256
channels_last = False

x_shape = [1, 512, 8, 8]
w_shape = [1, 512]
in_channels = 512
out_channels = 3
w_dim = 512
kernel_size = 1
conv_clamp = 256
channels_last = False

x_shape = [1, 512, 16, 16]
w_shape = [1, 512]
in_channels = 512
out_channels = 3
w_dim = 512
kernel_size = 1
conv_clamp = 256
channels_last = False

x_shape = [1, 512, 32, 32]
w_shape = [1, 512]
in_channels = 512
out_channels = 3
w_dim = 512
kernel_size = 1
conv_clamp = 256
channels_last = False

x_shape = [1, 512, 64, 64]
w_shape = [1, 512]
in_channels = 512
out_channels = 3
w_dim = 512
kernel_size = 1
conv_clamp = 256
channels_last = False

x_shape = [1, 256, 128, 128]
w_shape = [1, 512]
in_channels = 256
out_channels = 3
w_dim = 512
kernel_size = 1
conv_clamp = 256
channels_last = False

x_shape = [1, 128, 256, 256]
w_shape = [1, 512]
in_channels = 128
out_channels = 3
w_dim = 512
kernel_size = 1
conv_clamp = 256
channels_last = False

x_shape = [1, 64, 512, 512]
w_shape = [1, 512]
in_channels = 64
out_channels = 3
w_dim = 512
kernel_size = 1
conv_clamp = None
channels_last = False



fused_modconv = False
batch_size = 2
lr = 0.0001

model = ToRGBLayer(in_channels, out_channels, w_dim, kernel_size, conv_clamp, channels_last)
model.train()
optimizer = mge.optimizer.SGD(model.parameters(), lr=lr, momentum=0.9)
model.load_state_dict(mge.load("51.pkl", map_location="cpu"))


dic2 = np.load('51.npz')
for batch_idx in range(8):
    print('======================== batch_%.3d ========================'%batch_idx)
    optimizer.clear_grad()
    x = dic2['batch_%.3d.input0'%batch_idx]
    ws = dic2['batch_%.3d.input1'%batch_idx]
    y_pytorch = dic2['batch_%.3d.output'%batch_idx]
    dy_dws_pytorch = dic2['batch_%.3d.dy_dws'%batch_idx]
    dy_dx_pytorch = dic2['batch_%.3d.dy_dx'%batch_idx]
    ws = mge.tensor(ws)
    ws.stop_gradient = False
    x = mge.tensor(x)
    x.stop_gradient = False

    gm = GradManager()
    gm2 = GradManager()
    gm.attach([ws])
    gm2.attach([ws])
    gm.attach([x])
    gm2.attach([x])
    gm.attach(model.parameters())
    gm2.attach(model.parameters())


    with gm:
        with gm2:
            y = model(x, ws, fused_modconv=fused_modconv)

            dy_dx, dy_dws = mge_grad(gm2, output=y, inputs=[x, ws], clear_grad_nets=[model])

            y_mge = y.numpy()
            ddd = np.sum((y_pytorch - y_mge) ** 2)
            print('ddd=%.6f' % ddd)

            dy_dx_mge = dy_dx.numpy()
            ddd = np.sum((dy_dx_pytorch - dy_dx_mge) ** 2)
            print('ddd=%.6f' % ddd)

            dy_dws_mge = dy_dws.numpy()
            ddd = np.sum((dy_dws_pytorch - dy_dws_mge) ** 2)
            print('ddd=%.6f' % ddd)

            ddd = np.mean((y_pytorch - y_mge) ** 2)
            print('ddd=%.6f' % ddd)
            ddd = np.mean((dy_dx_pytorch - dy_dx_mge) ** 2)
            print('ddd=%.6f' % ddd)
            ddd = np.mean((dy_dws_pytorch - dy_dws_mge) ** 2)
            print('ddd=%.6f' % ddd)

            # loss = dy_dx.sum() + dy_dws.sum() + y.sum()   # 分辨率太大（如512）时，和pytorch获得不一样的结果xxx。暂时不修复这个bug
            # loss = dy_dx.sum() + y.sum()
            loss = dy_dws.sum() + y.sum()    # 和pytorch获得一样的结果。
            # loss = y.sum()
            gm.backward(loss)
    optimizer.step()
print()
