# Bitou Segmentation

Project folder for the bitou bush segmentation project.
This project is following tips and guidelines from the [Good Research Code Handbook](https://goodresearch.dev/setup.html).

The main package that is used in this project is [Pytorch lightning flash](https://lightning-flash.readthedocs.io/en/latest/installation.html) and its associated package [lightning bolts](https://www.pytorchlightning.ai/bolts)

possible segmentation models available from [qbvel](https://github.com/qubvel/segmentation_models.pytorch)
consider background randomization - see Julian's code and [this article](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0)
or [this guy](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0)

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
|       |   base_task.py - file that is analogous to flash's segmentation task - base file to start from
|       |   generate_masks.py - file that generates binary masks from a folder
|
└─── csupl - importable code. Should be installed using pip - see [below](###-Installing-src-files-in-csupl)
|       |    \__init__.py 
|
└─── tests - tests to check imports and proper function. can be run using `pytest .` inside the directory
|       |   test_imports.py - checks if import and GPU functions are available
|       |   test_segmentation.py - checks if the segmentation models are available
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
#### Environment
Running `pytest .` inside the `tests` directory will ensure the packages are configured correctly

#### Flash functionality
running `scripts/base_task.py` runs the [default Semantic Segmentation task](https://lightning-flash.readthedocs.io/en/latest/reference/semantic_segmentation.html)

## Generating masks from Json
`.json` files should be created using VGG Image Annotation tool.
`generate_masks.py`[scripts/generate_masks.py] needs a .json file with the image annotations, an input directory for the original files and an output directory for the masks.