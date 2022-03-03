
import torch
import numpy as np
from training.networks import FullyConnectedLayer


batch_size = 2
in_channels = 512
w_dim = 512
lr = 0.0001
activation = 'linear'
activation = 'lrelu'
activation = 'relu'
# activation = 'tanh'
# activation = 'sigmoid'
# activation = 'elu'
# activation = 'selu'
# activation = 'softplus'
# activation = 'swish'


model = FullyConnectedLayer(w_dim, in_channels, activation=activation, bias_init=1)
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
torch.save(model.state_dict(), "pytorch_fullyConnectedLayer.pth")

dic = {}
for batch_idx in range(20):
    optimizer.zero_grad(set_to_none=True)

    ws = torch.randn([batch_size, 512])
    ws.requires_grad_(True)

    styles = model(ws)
    dstyles_dws = torch.autograd.grad(outputs=[styles.sum()], inputs=[ws], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dstyles_dws'%batch_idx] = dstyles_dws.cpu().detach().numpy()
    dic['batch_%.3d.output'%batch_idx] = styles.cpu().detach().numpy()
    dic['batch_%.3d.input'%batch_idx] = ws.cpu().detach().numpy()

    loss = dstyles_dws.sum() + styles.sum()
    loss.backward()
    optimizer.step()
np.savez('01_fullyConnectedLayer_grad', **dic)
print()
