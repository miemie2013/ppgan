
learning_rate = 0.002
beta1 = 0.0
beta2 = 0.99


G_reg_interval = 4
D_reg_interval = 16

for name, reg_interval in [('G', G_reg_interval), ('D', D_reg_interval)]:
    if reg_interval is None:
        pass
        # opt = dnnlib.util.construct_class_by_name(params=module.parameters(),
        #                                           **opt_kwargs)  # subclass of torch.optim.Optimizer
        # phases += [dnnlib.EasyDict(name=name + 'both', module=module, opt=opt, interval=1)]
    else:  # Lazy regularization.
        mb_ratio = reg_interval / (reg_interval + 1)
        new_lr = learning_rate * mb_ratio
        new_beta1 = beta1 ** mb_ratio
        new_beta2 = beta2 ** mb_ratio
    print(new_lr)
    print(new_beta1)
    print(new_beta2)
    print()




