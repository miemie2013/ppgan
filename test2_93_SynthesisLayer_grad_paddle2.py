
import paddle
import numpy as np
from paddle_networks2 import SynthesisLayer

w_dim = 512
in_channels = 512
out_channels = 512
conv_clamp = 256
channels_last = False
fused_modconv = False
layer_kwargs = {}


resolution = 4
synthesisLayer = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
                                conv_clamp=conv_clamp, channels_last=channels_last, **layer_kwargs)

# resample_filter = [1, 3, 3, 1]
# resolution = 8
# synthesisLayer = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution,
#                                 up=2, resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=channels_last, **layer_kwargs)

synthesisLayer.train()
synthesisLayer.set_state_dict(paddle.load("pytorch_synthesisLayer.pdparams"))


pre_name = 'synthesisLayer'
dic2 = np.load('synthesisLayer_grad.npz')
x = dic2[pre_name + '.input0']
ws = dic2[pre_name + '.input1']
y_pytorch = dic2[pre_name + '.output']
dy_dx_pytorch = dic2[pre_name + '.dy_dx']
dy_dws_pytorch = dic2[pre_name + '.dy_dws']
x = paddle.to_tensor(x)
x.stop_gradient = False
ws = paddle.to_tensor(ws)
ws.stop_gradient = False


y, img2, styles = synthesisLayer(x, ws, dic2, pre_name + '.synthesisLayer', fused_modconv=fused_modconv)
# y = synthesisLayer(x, ws, fused_modconv=fused_modconv)

dysum_dy = paddle.grad(outputs=[y.sum()], inputs=[y], create_graph=True)[0]
dy_dx, dy_dws = synthesisLayer.get_grad(dysum_dy, y, img2, styles, x, ws)

# dy_dx = paddle.grad(outputs=[y.sum()], inputs=[x], create_graph=True)[0]
# dy_dws = paddle.grad(outputs=[y.sum()], inputs=[ws], create_graph=True)[0]

y_paddle = y.numpy()
ddd = np.sum((y_pytorch - y_paddle)**2)
print('ddd=%.6f' % ddd)


dy_dx_paddle = dy_dx.numpy()
ddd = np.sum((dy_dx_pytorch - dy_dx_paddle)**2)
print('ddd=%.6f' % ddd)


dy_dws_paddle = dy_dws.numpy()
ddd = np.sum((dy_dws_pytorch - dy_dws_paddle)**2)
print('dy_dws_diff=%.6f' % ddd)


print()
