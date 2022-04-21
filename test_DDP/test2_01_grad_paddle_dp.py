
import paddle
import paddle.nn.functional as F
import numpy as np
from paddle.distributed import ParallelEnv
from paddle.io import DistributedBatchSampler


class FullyConnectedLayer(paddle.nn.Layer):
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
        self.weight = self.create_parameter([out_features, in_features],
                                            default_initializer=paddle.nn.initializer.Normal(mean=0.0, std=1.0 / lr_multiplier))
        self.bias = self.create_parameter([out_features], is_bias=True,
                                          default_initializer=paddle.nn.initializer.Constant(bias_init)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = paddle.cast(self.weight, dtype=x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = paddle.cast(b, dtype=x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            # out = paddle.addmm(b.unsqueeze(0), x, w.t())   # 因为paddle.addmm()没有实现二阶梯度，所以用其它等价实现。
            out = paddle.matmul(x, w, transpose_y=True) + b.unsqueeze(0)
        else:
            r = x.matmul(w.t())
            r += b.unsqueeze(0)
            out = F.sigmoid(r)

        styles = out

        styles2 = F.sigmoid(styles)
        dstyles2_dws = paddle.grad(outputs=[styles2.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]

        loss = dstyles2_dws.sum() + styles2.sum()
        return loss

class MyDataset(paddle.io.Dataset):
    def __init__(self, npz_path, batch_size, steps):
        self.dic2 = np.load(npz_path)
        self.batch_size = batch_size
        self.steps = steps

    def __len__(self):
        # size = len(self.seeds)
        return self.batch_size * self.steps

    def __getitem__(self, idx):
        batch_idx = idx // self.batch_size
        i = idx % self.batch_size

        ws = self.dic2['batch_%.3d.input' % batch_idx]
        styles_pytorch = self.dic2['batch_%.3d.output' % batch_idx]
        dstyles2_dws_pytorch = self.dic2['batch_%.3d.dstyles2_dws' % batch_idx]

        # miemieGAN中验证集的写法
        # datas = {
        #     'ws': ws[i],
        #     'styles_pytorch': styles_pytorch[i],
        #     'dstyles2_dws_pytorch': dstyles2_dws_pytorch[i],
        # }
        # miemieGAN中训练集的写法
        datas = (ws[i], styles_pytorch[i], dstyles2_dws_pytorch[i])
        return datas


class IterLoader:
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 1

    @property
    def epoch(self):
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)


if __name__ == '__main__':
    # 代码参考自PaddleGAN
    # 不可以少了这句
    paddle.set_device('gpu')

    local_rank = ParallelEnv().local_rank  # 咩酱：当前训练进程的编号。
    world_size = ParallelEnv().nranks  # 咩酱：参与训练进程的数量，一般也是训练所使用GPU卡的数量。
    is_distributed = world_size > 1

    batch_size = 16
    steps = 20
    batch_gpu = batch_size // world_size
    in_channels = 2
    w_dim = 2
    lr = 0.1

    # "双卡+每卡批大小b//2",如果要对齐"单卡+每卡批大小b"的训练过程,
    # loss或学习率lr需要乘以显卡数量get_world_size()
    # 因为多卡训练时,每一张卡上的梯度是求平均值而并不是求和
    # 即 paddle的DP 和 pytorch的DDP 在这里是一样的，都是每一张卡上的梯度求平均值
    if is_distributed:
        lr *= world_size



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
    if world_size > 1:
        paddle.distributed.init_parallel_env()
        model = paddle.DataParallel(model, find_unused_parameters=True)
    model.train()

    optimizer = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=lr, momentum=0.9)
    model.set_state_dict(paddle.load("01_00.pdparams"))

    is_train = True
    num_workers = 0
    use_shared_memory = False
    train_dataset = MyDataset('01.npz', batch_size, steps)
    sampler = DistributedBatchSampler(train_dataset,
                                      batch_size=batch_gpu,  # 注意，这里一定要是batch_gpu。即PaddleGAN中yaml配置文件的批大小其实是batch_gpu。
                                      # shuffle=True if is_train else False,
                                      shuffle=False,  # 不打乱顺序
                                      drop_last=True if is_train else False)
    train_dataloader = paddle.io.DataLoader(train_dataset,
                                            batch_sampler=sampler,
                                            num_workers=num_workers,
                                            use_shared_memory=use_shared_memory)
    iter_loader = IterLoader(train_dataloader)
    iters_per_epoch = len(train_dataloader)
    print('iters_per_epoch =', iters_per_epoch)
    # dic2 = np.load('01.npz')
    for batch_idx in range(20):
        print('======================== batch_%.3d ========================'%batch_idx)
        optimizer.clear_gradients()
        # ws = dic2['batch_%.3d.input'%batch_idx]
        # styles_pytorch = dic2['batch_%.3d.output'%batch_idx]
        # dstyles2_dws_pytorch = dic2['batch_%.3d.dstyles2_dws'%batch_idx]
        ws, styles_pytorch, dstyles2_dws_pytorch = next(iter_loader)
        print('ws.shape =', ws.shape)
        ws.stop_gradient = False

        # 多卡训练 且 调用的是模型的forward()方法 时，不需要调用model_ = model.module
        # forward()方法不要return任何不计算loss的变量！
        loss = model(ws)

        # "双卡+每卡批大小b//2",如果要对齐"单卡+每卡批大小b"的训练过程,
        # loss或学习率lr需要乘以显卡数量get_world_size()
        # 因为多卡训练时,每一张卡上的梯度是求平均值而并不是求和
        loss.backward()

        if is_distributed:
            w_grad = model._layers.weight.gradient()
            b_grad = model._layers.bias.gradient()
        else:
            w_grad = model.weight.gradient()
            b_grad = model.bias.gradient()

        print(w_grad)
        print(b_grad)
        optimizer.step()
    paddle.save(model.state_dict(), "01_19_DDP.pdparams")
    print()
