from typing import Dict, BinaryIO
import subprocess
import tempfile

import numpy as np
import torch
from torch import nn
import mlx.core as mx

TEST_BIN = 'build/test_mamba'

Tensors = Dict[str, torch.Tensor]


def torch2gguf(tensors: Tensors) -> tempfile.NamedTemporaryFile:
    with torch.no_grad():
        mx_arrays = {k: mx.array(v.numpy()) for k, v in tensors.items()}
        tmp = tempfile.NamedTemporaryFile(suffix='.gguf')
        mx.save_gguf(tmp.name, mx_arrays)
        return tmp


def gguf2torch(file: BinaryIO) -> Tensors:
    mx_arrays = mx.load(file.name)
    return {k: torch.tensor(np.array(v))
            for k, v in mx_arrays.items()}


def execute_test_fn(name: str, tensors: Tensors) -> Tensors:
    with torch2gguf(tensors) as input:
        with tempfile.NamedTemporaryFile(suffix='.gguf') as output:
            proc = subprocess.run(
                [TEST_BIN, name, input.name, output.name], capture_output=True)
            if proc.stdout != b'':
                print('stdout', proc.stdout.decode('utf-8'))
            if proc.stderr != b'':
                print('stderr', proc.stderr.decode('utf-8'))
            assert proc.returncode == 0
            return gguf2torch(output)


def test_sanity():
    x = torch.randn(30, 20, 10)
    res = execute_test_fn('sanity', {'x': x})
    assert torch.equal(res['y'], x)


def test_linear_layer():
    layer = nn.Linear(30, 20)
    x = torch.randn(30)
    y_expected = layer(x)
    res = execute_test_fn(
        'linear_layer', {'weight': layer.weight, 'bias': layer.bias, 'x': x})
    assert torch.allclose(res['y'], y_expected)


def test_mamba_block_discretize():
    l, d, n = 20, 8, 4
    u = torch.randn(l, d)
    delta = torch.randn(l, d)
    A = torch.randn(d, n)
    B = torch.randn(l, n)
    deltaA_expected = torch.exp(torch.einsum('ld, dn -> ldn', delta, A))
    deltaB_u_expected = torch.einsum('ld, ln, ld -> ldn', delta, B, u)
    res = execute_test_fn('mamba_block_discretize', {
                          'u': u, 'delta': delta, 'A': A, 'B': B})
    assert torch.allclose(res['deltaA'], deltaA_expected)
    assert torch.allclose(res['deltaB_u'], deltaB_u_expected)
