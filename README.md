# Bitou Segmentation

Project folder for the bitou bush segmentation project.
This project is following tips and guidelines from the [Good Research Code Handbook](https://goodresearch.dev/setup.html).

The main package that is used in this project is [Pytorch lightning flash](https://lightning-flash.readthedocs.io/en/latest/installation.html) and its associated package [lightning bolts](https://www.pytorchlightning.ai/bolts)


## Folder Structure
'''
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

'''

### Installing src files in csupl
run `pip install -e .` from base folder. Installs current directory with editable configuration. Can now be imported anywhere

alternative is defining a .ENV file for vscode

## TODOs
