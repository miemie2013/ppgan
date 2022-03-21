
import torch
import numpy as np
from training.networks import FullyConnectedLayer


class Line(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, x, b):
        ctx.save_for_backward(w, x, b)
        return w * x + b

    @staticmethod
    def backward(ctx, grad_out):
        w, x, b = ctx.saved_tensors
        grad_w = grad_out * x
        grad_x = grad_out * w
        grad_b = grad_out
        return grad_w, grad_x, grad_b


class MyTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.tanh(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        y, = ctx.saved_tensors
        # dx = dy * (1 - torch.square(y))
        dx = MyTanhGrad.apply(dy, y)
        return dx

class MyTanhGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        y = x * (1 - torch.square(w))
        ctx.save_for_backward(x, w)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        dx = dy * (1 - torch.square(w))
        dw = dy * x * -2 * w
        return dx, dw





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
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
model.load_state_dict(torch.load("19.pth", map_location="cpu"))

dic2 = np.load('19.npz')
for batch_idx in range(20):
    print('======================== batch_%.3d ========================'%batch_idx)
    optimizer.zero_grad(set_to_none=True)
    ws = dic2['batch_%.3d.input'%batch_idx]
    styles_pytorch = dic2['batch_%.3d.output'%batch_idx]
    dstyles2_dws_pytorch = dic2['batch_%.3d.dstyles2_dws'%batch_idx]
    ws = torch.Tensor(ws)
    ws.requires_grad_(True)

    styles = model(ws)
    aaa = MyTanh.apply(styles)
    # aaa = torch.tanh(styles)
    styles2 = torch.sigmoid(aaa)
    dstyles2_dws = torch.autograd.grad(outputs=[styles2.sum()], inputs=[ws], create_graph=True, only_inputs=True)[0]

    styles_pytorch2 = styles.cpu().detach().numpy()
    ddd = np.sum((styles_pytorch - styles_pytorch2) ** 2)
    print('ddd=%.6f' % ddd)

    dstyles2_dws_pytorch2 = dstyles2_dws.cpu().detach().numpy()
    ddd = np.sum((dstyles2_dws_pytorch - dstyles2_dws_pytorch2) ** 2)
    print('ddd=%.6f' % ddd)

    loss = dstyles2_dws.sum() + styles2.sum()
    loss.backward()
    optimizer.step()
print()
