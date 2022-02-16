

import numpy as np
import torch
import paddle

'''


'''
ppppppppppp = 0.5
batch_size = 2

'''
I_3 = torch.eye(3)
device = I_3.device
G_inv = I_3

_constant_cache = dict()


def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor

def matrix(*rows, device=None):
    assert all(len(row) == len(rows[0]) for row in rows)
    elems = [x for row in rows for x in row]
    ref = [x for x in elems if isinstance(x, torch.Tensor)]
    if len(ref) == 0:
        return constant(np.asarray(rows), device=device)
    assert device is None or device == ref[0].device
    elems = [x if isinstance(x, torch.Tensor) else constant(x, shape=ref[0].shape, device=ref[0].device) for x in elems]
    bbbbbbb = torch.stack(elems, dim=-1)
    bbbbbbbb = bbbbbbb.reshape(ref[0].shape + (len(rows), -1))
    return bbbbbbbb

def scale2d(sx, sy, **kwargs):
    return matrix(
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1],
        **kwargs)

def scale2d_inv(sx, sy, **kwargs):
    return scale2d(1 / sx, 1 / sy, **kwargs)


debug_percentile = None

www = torch.rand([batch_size], device=device)
i = torch.floor(www * 2)


i = torch.where(torch.rand([batch_size], device=device) < ppppppppppp, i, torch.zeros_like(i))
if debug_percentile is not None:
    i = torch.full_like(i, torch.floor(debug_percentile * 2))

aaaaaaa = scale2d_inv(1 - 2 * i, 1)
G_inv = G_inv @ aaaaaaa
'''



I_3 = paddle.eye(3)
G_inv = I_3

_constant_cache = dict()


def constant(value, shape=None, dtype=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = paddle.get_default_dtype()

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        # tensor = paddle.to_tensor(value.copy(), dtype=dtype)
        tensor = paddle.ones(shape, dtype=dtype) * value
        _constant_cache[key] = tensor
    return tensor

def matrix(*rows):
    assert all(len(row) == len(rows[0]) for row in rows)
    elems = [x for row in rows for x in row]
    ref = [x for x in elems if isinstance(x, paddle.Tensor)]
    if len(ref) == 0:
        return constant(np.asarray(rows))
    elems = [x if isinstance(x, paddle.Tensor) else constant(x, shape=ref[0].shape) for x in elems]
    bbbbbbb = paddle.stack(elems, axis=-1)
    bbbbbbbb = bbbbbbb.reshape((tuple(ref[0].shape) + (len(rows), -1)))
    return bbbbbbbb

def scale2d(sx, sy, **kwargs):
    return matrix(
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1],
        **kwargs)

def scale2d_inv(sx, sy, **kwargs):
    return scale2d(1 / sx, 1 / sy, **kwargs)


debug_percentile = None

www = paddle.rand([batch_size], dtype=paddle.float32)
i = paddle.floor(www * 2)


i = paddle.where(paddle.rand([batch_size], dtype=paddle.float32) < ppppppppppp, i, paddle.zeros_like(i))
if debug_percentile is not None:
    i = paddle.full_like(i, paddle.floor(debug_percentile * 2))

aaaaaaa = scale2d_inv(1 - 2 * i, 1)
G_inv = G_inv @ aaaaaaa




print()



