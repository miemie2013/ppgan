
import paddle
import numpy as np
from paddle_networks2 import ToRGBLayer

w_dim = 512
out_channels = 512
img_channels = 3
conv_clamp = 256
channels_last = False
fused_modconv = False

toRGBLayer = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                        conv_clamp=conv_clamp, channels_last=channels_last)
toRGBLayer.train()
toRGBLayer.set_state_dict(paddle.load("pytorch_toRGBLayer.pdparams"))


pre_name = 'toRGBLayer'
dic2 = np.load('toRGBLayer_grad.npz')
x = dic2[pre_name + '.input0']
ws = dic2[pre_name + '.input1']
y_pytorch = dic2[pre_name + '.output']
dy_dx_pytorch = dic2[pre_name + '.dy_dx']
dy_dws_pytorch = dic2[pre_name + '.dy_dws']
x = paddle.to_tensor(x)
x.stop_gradient = False
ws = paddle.to_tensor(ws)
ws.stop_gradient = False


# y = toRGBLayer(x, ws, dic2, pre_name + '.toRGBLayer', fused_modconv=fused_modconv)
y = toRGBLayer(x, ws, fused_modconv=fused_modconv)

dy_dx = paddle.grad(outputs=[y.sum()], inputs=[x], create_graph=True)[0]
dy_dws = paddle.grad(outputs=[y.sum()], inputs=[ws], create_graph=True)[0]

y_paddle = y.numpy()
ddd = np.sum((y_pytorch - y_paddle)**2)
print('ddd=%.6f' % ddd)


dy_dx_paddle = dy_dx.numpy()
ddd = np.sum((dy_dx_pytorch - dy_dx_paddle)**2)
print('ddd=%.6f' % ddd)


dy_dws_paddle = dy_dws.numpy()
ddd = np.sum((dy_dws_pytorch - dy_dws_paddle)**2)
print('ddd=%.6f' % ddd)


print()
