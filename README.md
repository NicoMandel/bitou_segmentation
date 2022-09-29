# Bitou Segmentation

Project folder for the bitou bush segmentation project.
This project is following tips and guidelines from the [Good Research Code Handbook](https://goodresearch.dev/setup.html).

The main package that is used in this project is [Pytorch lightning flash](https://lightning-flash.readthedocs.io/en/latest/installation.html) and its associated package [lightning bolts](https://www.pytorchlightning.ai/bolts)


## Folder Structure
```
bitou_segmentation
|   .gitignore
|   conda_env_linux_gpu.yml
|   README.md
|   setup.py
|
└─── data - where pytorchs dataloader expects data - mainly gitignored
|       └─── images
|       |       
|       └─── masks
|               | 
|
└─── results - intermediate files and results. mostly gitignored
|       |   
|       |   
|
└─── scripts - executable scripts that are running things
|       |   
|       |   
|
└─── csupl - importable code. Should be installed using pip - see [below](###-Installing-src-files-in-csupl)
|       |    \__init__.py 

```


## Installing src files in csupl
run `pip install -e .` from base folder. Installs current directory with editable configuration. Can now be imported anywhere

alternative is defining a .ENV file for vscode

## Environment files
Environments can be recreated on the native host using `conda create --name plenv --file <filename>.txt`
there are currently two environments available, both working with RTX GPUs (Cuda 11.X), see [pytorch documentation](https://discuss.pytorch.org/t/nvidia-geforce-rtx-3090-with-cuda-capability-sm-86-is-not-compatible-with-the-current-pytorch-installation/141940)
1. [csuflash.txt](csuflash.txt) - created running `conda install -c conda-forge lightning-flash` as [lightning flash install](https://lightning-flash.readthedocs.io/en/latest/installation.html) guide proposes. As a follow-up, cudatoolkit is replaced by adding `conda install cudatoolkit=11.6`
2. [csuplflash.txt](csuplflash.txt) - created through subsequent installation and cloning of:
    1. `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
    2. `conda install pytorch-lightning -c conda-forge`
    3. `conda install -c conda-forge lightning-flash`

### Testing
Running the following code inside the conda environment will test validity. If `torch.cuda.get_device_name(0)` does not return with a warning and `torch.cuda.get_arch_list()` contains `smi_86` things should work.
Expected output:
```
>>> import torch
>>> torch.cuda.is_available()
True
>>> torch.version.cuda
'11.3'
>>> torch.__version__
'1.12.1'
>>> torch.cuda.get_arch_list()
['sm_37', 'sm_50', 'sm_60', 'sm_61', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'compute_37']
>>> torch.cuda.get_device_name(0)
'NVIDIA GeForce RTX 3060 Laptop GPU'
```


## TODOs
