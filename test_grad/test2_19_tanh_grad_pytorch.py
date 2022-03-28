
import torch
import numpy as np
from training.networks import FullyConnectedLayer



class MyTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.tanh(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        y, = ctx.saved_tensors
        # dloss_dx = dloss_dy * (1 - torch.square(y))
        dx = MyTanhGrad.apply(dy, y)
        return dx

class MyTanhGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dy, y):
        dx = dy * (1 - torch.square(y))
        ctx.save_for_backward(dy, y)
        return dx

    @staticmethod
    def backward(ctx, ddx):
        dy, y = ctx.saved_tensors
        ddy = ddx * (1 - torch.square(y))
        dy_new = ddx * dy * -2 * y
        # dy_new = None
        return ddy, dy_new



class MyRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.relu(x)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        # dx = dy * torch.where(x > 0., 1., 0.)
        dx = MyReluGrad.apply(dy, x)
        return dx

class MyReluGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dy, x):
        dx = dy * torch.where(x > 0., 1., 0.)
        ctx.save_for_backward(dy, x)
        return dx

    @staticmethod
    def backward(ctx, ddx):
        dy, x = ctx.saved_tensors
        ddy = ddx * torch.where(x > 0., 1., 0.)
        dx = None
        return ddy, dx





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
    # aaa = MyRelu.apply(styles)
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
