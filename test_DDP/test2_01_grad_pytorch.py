
import os
from datetime import timedelta
import sys
import torch.distributed as dist
import torch.multiprocessing as mp
import cv2
import random
import torch
import torch.nn.functional as F
import numpy as np

import warnings
import torch.backends.cudnn as cudnn
import subprocess

import functools
import os
import pickle
import time
from contextlib import contextmanager
from loguru import logger

import numpy as np

import torch
from torch import distributed as dist


_LOCAL_PROCESS_GROUP = None


def get_num_devices():
    gpu_list = os.getenv('CUDA_VISIBLE_DEVICES', None)
    if gpu_list is not None:
        return len(gpu_list.split(','))
    else:
        devices_list_info = os.popen("nvidia-smi -L")
        devices_list_info = devices_list_info.read().strip().split("\n")
        return len(devices_list_info)


@contextmanager
def wait_for_the_master(local_rank: int):
    """
    Make all processes waiting for the master to do some task.
    """
    if local_rank > 0:
        dist.barrier()
    yield
    if local_rank == 0:
        if not dist.is_available():
            return
        if not dist.is_initialized():
            return
        else:
            dist.barrier()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group, i.e. the number of processes per machine.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
    return get_rank() == 0


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device)
        for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros(
            (max_size - local_size,), dtype=torch.uint8, device=tensor.device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
        for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group=group) == 1:
        return [data]
    rank = dist.get_rank(group=group)

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    # receiving Tensor from all ranks
    if rank == dst:
        max_size = max(size_list)
        tensor_list = [
            torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
            for _ in size_list
        ]
        dist.gather(tensor, tensor_list, dst=dst, group=group)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        return data_list
    else:
        dist.gather(tensor, [], dst=dst, group=group)
        return []


def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
            If workers need a shared RNG, they can use this shared seed to
            create one.
    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2 ** 31)
    all_ints = all_gather(ints)
    return all_ints[0]


def time_synchronized():
    """pytorch-accurate time"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()







def get_num_devices():
    gpu_list = os.getenv('CUDA_VISIBLE_DEVICES', None)
    if gpu_list is not None:
        return len(gpu_list.split(','))
    else:
        devices_list_info = os.popen("nvidia-smi -L")
        devices_list_info = devices_list_info.read().strip().split("\n")
        return len(devices_list_info)


DEFAULT_TIMEOUT = timedelta(minutes=30)

def _find_free_port():
    """
    Find an available port of current machine / node.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    backend,
    dist_url,
    args,
    timeout=DEFAULT_TIMEOUT,
):
    assert (
        torch.cuda.is_available()
    ), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    logger.info("Rank {} initialization finished.".format(global_rank))
    try:
        dist.init_process_group(
            backend=backend,
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
            timeout=timeout,
        )
    except Exception:
        logger.error("Process group URL: {}".format(dist_url))
        raise

    # Setup the local process group (which contains ranks within the same machine)
    global _LOCAL_PROCESS_GROUP
    assert _LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(
            range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine)
        )
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            _LOCAL_PROCESS_GROUP = pg

    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    synchronize()

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    main_func(*args)

def launch(
    main_func,
    num_gpus_per_machine,
    num_machines=1,
    machine_rank=0,
    backend="nccl",
    dist_url=None,
    args=(),
    timeout=DEFAULT_TIMEOUT,
):
    """
    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine (one per machine)
        dist_url (str): url to connect to for distributed training, including protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to auto to automatically select a free port on localhost
        args (tuple): arguments passed to main_func
    """
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        # https://github.com/pytorch/pytorch/pull/14391
        # TODO prctl in spawned processes

        if dist_url == "auto":
            assert (
                num_machines == 1
            ), "dist_url=auto cannot work with distributed training."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"

        start_method = "spawn"
        # cache = vars(args[1]).get("cache", False)
        cache = False

        # To use numpy memmap for caching image into RAM, we have to use fork method
        if cache:
            assert sys.platform != "win32", (
                "As Windows platform doesn't support fork method, "
                "do not add --cache in your training command."
            )
            start_method = "fork"

        mp.start_processes(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(
                main_func,
                world_size,
                num_gpus_per_machine,
                machine_rank,
                backend,
                dist_url,
                args,
            ),
            daemon=False,
            start_method=start_method,
        )
    else:
        main_func(*args)



def configure_nccl():
    """Configure multi-machine environment variables of NCCL."""
    os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
    os.environ["NCCL_IB_HCA"] = subprocess.getoutput(
        "pushd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; "
        "do cat $i/ports/1/gid_attrs/types/* 2>/dev/null "
        "| grep v >/dev/null && echo $i ; done; popd > /dev/null"
    )
    os.environ["NCCL_IB_GID_INDEX"] = "3"
    os.environ["NCCL_IB_TC"] = "106"


def configure_omp(num_threads=1):
    """
    If OMP_NUM_THREADS is not configured and world_size is greater than 1,
    Configure OMP_NUM_THREADS environment variables of NCCL to `num_thread`.

    Args:
        num_threads (int): value of `OMP_NUM_THREADS` to set.
    """
    # We set OMP_NUM_THREADS=1 by default, which achieves the best speed on our machines
    # feel free to change it for better performance.
    if "OMP_NUM_THREADS" not in os.environ and get_world_size() > 1:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        if is_main_process():
            logger.info(
                "\n***************************************************************\n"
                "We set `OMP_NUM_THREADS` for each process to {} to speed up.\n"
                "please further tune the variable for optimal performance.\n"
                "***************************************************************".format(
                    os.environ["OMP_NUM_THREADS"]
                )
            )


def configure_module(ulimit_value=8192):
    """
    Configure pytorch module environment. setting of ulimit and cv2 will be set.

    Args:
        ulimit_value(int): default open file number on linux. Default value: 8192.
    """
    # system setting
    try:
        import resource

        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (ulimit_value, rlimit[1]))
    except Exception:
        # Exception might be raised in Windows OS or rlimit reaches max limit number.
        # However, set rlimit value might not be necessary.
        pass

    # cv2
    # multiprocess might be harmful on performance of torch dataloader
    os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
    try:
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        # cv2 version mismatch might rasie exceptions.
        pass



from torch.nn.parallel import DistributedDataParallel as DDP


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)



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

        styles = out

        styles2 = torch.sigmoid(styles)
        dstyles2_dws = torch.autograd.grad(outputs=[styles2.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]

        loss = dstyles2_dws.sum() + styles2.sum()
        # return styles, styles2, dstyles2_dws, loss
        return loss


class MyDataset(torch.utils.data.Dataset):
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

import uuid

def worker_init_reset_seed(worker_id):
    seed = uuid.uuid4().int % 2**32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)


import itertools
from typing import Optional
from torch.utils.data.sampler import Sampler

class InfiniteSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(
        self,
        size: int,
        shuffle: bool = True,
        seed: Optional[int] = 0,
        rank=0,
        world_size=1,
    ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        self._seed = int(seed)

        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size
        )

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)

    def __len__(self):
        return self._size // self._world_size



class StyleGANv2ADADataPrefetcher:
    """
    xxxDataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = StyleGANv2ADADataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.ws, self.styles_pytorch, self.dstyles2_dws_pytorch = next(self.loader)
        except StopIteration:
            self.ws = None
            self.styles_pytorch = None
            self.dstyles2_dws_pytorch = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.styles_pytorch = self.styles_pytorch.cuda(non_blocking=True)
            self.dstyles2_dws_pytorch = self.dstyles2_dws_pytorch.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        ws = self.ws
        styles_pytorch = self.styles_pytorch
        dstyles2_dws_pytorch = self.dstyles2_dws_pytorch
        if ws is not None:
            self.record_stream(ws)
        if styles_pytorch is not None:
            styles_pytorch.record_stream(torch.cuda.current_stream())
        if dstyles2_dws_pytorch is not None:
            dstyles2_dws_pytorch.record_stream(torch.cuda.current_stream())
        self.preload()
        return ws, styles_pytorch, dstyles2_dws_pytorch

    def _input_cuda_for_image(self):
        self.ws = self.ws.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())


def main(seed, args):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True


    is_distributed = get_world_size() > 1
    rank = get_rank()
    local_rank = get_local_rank()
    device = "cuda:{}".format(local_rank)

    batch_size = 16
    steps = 20
    in_channels = 2
    w_dim = 2
    lr = 0.1

    # "双卡+每卡批大小b//2",如果要对齐"单卡+每卡批大小b"的训练过程,
    # loss或学习率lr需要乘以显卡数量get_world_size()
    # 因为多卡训练时,每一张卡上的梯度是求平均值而并不是求和
    if is_distributed:
        lr *= get_world_size()


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
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.load_state_dict(torch.load("01_00.pth", map_location="cpu"))

    # miemieGAN中验证集的写法
    # train_dataset = MyDataset('01.npz', batch_size, steps)
    # if is_distributed:
    #     batch_gpu = batch_size // dist.get_world_size()
    #     sampler = torch.utils.data.distributed.DistributedSampler(
    #         train_dataset, shuffle=False
    #     )
    # else:
    #     batch_gpu = batch_size
    #     sampler = torch.utils.data.SequentialSampler(train_dataset)
    #
    # dataloader_kwargs = {
    #     "num_workers": 0,
    #     "pin_memory": True,
    #     "sampler": sampler,
    # }
    # dataloader_kwargs["batch_size"] = batch_gpu
    # train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_kwargs)

    # miemieGAN中训练集的写法
    local_rank = get_local_rank()
    with wait_for_the_master(local_rank):
        train_dataset = MyDataset('01.npz', batch_size, steps)

    if is_distributed:
        batch_size = batch_size // dist.get_world_size()

    sampler = InfiniteSampler(len(train_dataset), shuffle=False, seed=seed if seed else 0)

    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler=sampler,
        batch_size=batch_size,
        drop_last=True,
    )

    dataloader_kwargs = {"num_workers": 0, "pin_memory": True}
    dataloader_kwargs["batch_sampler"] = batch_sampler

    # Make sure each process has different random seed, especially for 'fork' method.
    # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
    dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_kwargs)
    logger.info("init prefetcher, this might take one minute or less...")
    prefetcher = StyleGANv2ADADataPrefetcher(train_loader)


    if is_distributed:
        # model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=True, find_unused_parameters=True)

    logger.info("Training start...")
    # miemieGAN中验证集的写法
    # for batch_idx, data in enumerate(train_loader):
    # miemieGAN中训练集的写法
    for batch_idx in range(20):
        # miemieGAN中验证集的写法
        # for k, v in data.items():
        #     data[k] = v.cuda()
        # ws = data['ws']
        # styles_pytorch = data['styles_pytorch']
        # dstyles2_dws_pytorch = data['dstyles2_dws_pytorch']
        # styles_pytorch = styles_pytorch.cpu().detach().numpy()
        # dstyles2_dws_pytorch = dstyles2_dws_pytorch.cpu().detach().numpy()

        # miemieGAN中训练集的写法
        ws, styles_pytorch, dstyles2_dws_pytorch = prefetcher.next()


        print('======================== batch_%.3d ========================' % batch_idx)
        optimizer.zero_grad(set_to_none=True)
        ws.requires_grad_(True)

        # 多卡训练 且 调用的是模型的forward()方法 时，不需要调用model_ = model.module
        # forward()方法不要return任何不计算loss的变量！
        loss = model(ws)

        # "双卡+每卡批大小b//2",如果要对齐"单卡+每卡批大小b"的训练过程,
        # loss或学习率lr需要乘以显卡数量get_world_size()
        # 因为多卡训练时,每一张卡上的梯度是求平均值而并不是求和
        loss.backward()
        if is_distributed:
            w_grad = model.module.weight.grad
            b_grad = model.module.bias.grad
        else:
            w_grad = model.weight.grad
            b_grad = model.bias.grad

        print(w_grad)
        print(b_grad)
        optimizer.step()

    # 多卡训练 且 保存模型 时，需要调用model_ = model.module
    if is_distributed:
        torch.save(model.module.state_dict(), "01_19_DDP.pth")
    else:
        torch.save(model.state_dict(), "01_19_DDP.pth")
    print()


if __name__ == "__main__":
    seed = 0
    args = None
    dist_backend = "nccl"
    num_machines = 1
    machine_rank = 0

    # 1 ji 1 ka
    num_gpu = 1
    dist_url = "auto"

    # 1 ji 2 ka
    num_gpu = 2
    dist_url = "auto"

    # 2 ji 2 ka
    # num_gpu = 1
    # dist_url = "tcp://192.168.0.104:12312"
    # num_machines = 2
    # machine_rank = 0


    assert num_gpu <= get_num_devices()
    launch(
        main,
        num_gpu,
        num_machines,
        machine_rank,
        backend=dist_backend,
        dist_url=dist_url,
        args=(seed, args),
    )







