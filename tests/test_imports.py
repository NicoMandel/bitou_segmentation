# to test imports
import pytest
import torch
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    def test_flash():
        import flash 

    def test_cuda():
        assert torch.cuda.is_available()

    def test_cuda_version():
        cuda_v = float(torch.version.cuda)
        assert cuda_v > 11.0
        

    def test_arch():
        rtx_arch = 'sm_86'
        archlist = torch.cuda.get_arch_list()
        assert rtx_arch in archlist

    def test_device():
        torch.cuda.get_device_name(0)

