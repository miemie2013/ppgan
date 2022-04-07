
import torch
import torch.nn.functional as F
import numpy as np


class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            out = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            r = x.matmul(w.t())
            r += b.unsqueeze(0)
            out = torch.sigmoid(r)
        return out




batch_size = 16
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
torch.save(model.state_dict(), "01_00.pth")

dic = {}
for batch_idx in range(20):
    optimizer.zero_grad(set_to_none=True)

    ws = torch.randn([batch_size, 512])
    ws.requires_grad_(True)

    styles = model(ws)
    styles2 = torch.sigmoid(styles)
    dstyles2_dws = torch.autograd.grad(outputs=[styles2.sum()], inputs=[ws], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dstyles2_dws'%batch_idx] = dstyles2_dws.cpu().detach().numpy()
    dic['batch_%.3d.output'%batch_idx] = styles.cpu().detach().numpy()
    dic['batch_%.3d.input'%batch_idx] = ws.cpu().detach().numpy()

    loss = dstyles2_dws.sum() + styles2.sum()
    # loss = styles2.sum()
    loss.backward()
    optimizer.step()
np.savez('01', **dic)
torch.save(model.state_dict(), "01_19.pth")
print()
