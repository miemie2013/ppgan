
import torch
import numpy as np
from training.networks import FullyConnectedLayer


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
torch.save(model.state_dict(), "19.pth")

dic = {}
for batch_idx in range(20):
    optimizer.zero_grad(set_to_none=True)

    ws = torch.randn([batch_size, 512])
    ws.requires_grad_(True)

    styles = model(ws)
    aaa = torch.tanh(styles)
    styles2 = torch.sigmoid(aaa)
    dstyles2_dws = torch.autograd.grad(outputs=[styles2.sum()], inputs=[ws], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dstyles2_dws'%batch_idx] = dstyles2_dws.cpu().detach().numpy()
    dic['batch_%.3d.output'%batch_idx] = styles.cpu().detach().numpy()
    dic['batch_%.3d.input'%batch_idx] = ws.cpu().detach().numpy()

    loss = dstyles2_dws.sum() + styles2.sum()
    loss.backward()
    optimizer.step()
np.savez('19', **dic)
print()
