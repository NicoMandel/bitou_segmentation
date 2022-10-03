# to test imports
import pytest
import warnings


def test_torch():
    import torch

def test_flash():
    import flash 

def test_cuda():
    import torch
    assert torch.cuda.is_available()

def test_cuda_version():
    import torch
    cuda_v = float(torch.version.cuda)
    assert cuda_v > 11.0
    

def test_arch():
    import torch
    rtx_arch = 'sm_86'
    archlist = torch.cuda.get_arch_list()
    assert rtx_arch in archlist

def test_device():
    import torch
    torch.cuda.get_device_name(0)

