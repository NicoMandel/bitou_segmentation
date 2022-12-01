# Bitou Segmentation

Project folder for the bitou bush segmentation project.
This project is following tips and guidelines from the [Good Research Code Handbook](https://goodresearch.dev/setup.html) for structure and setup

The main packages that this project relies on are :
* [VGG Image Annotation Tool (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/) for labelling masks - outputs to a json file, which can be read by utils functions
    * Can be downloaded as a html file and run in every browser
* [Pytorch Lightning](https://www.pytorchlightning.ai/) for the training and testing pipelines and loops
* [Segmentation Models Pytorch](https://smp.readthedocs.io/en/latest/index.html) for getting:
    * [SoA models](https://smp.readthedocs.io/en/latest/models.html)
    * [Losses](https://smp.readthedocs.io/en/latest/losses.html)
    * [Metrics](https://smp.readthedocs.io/en/latest/metrics.html)
* [Albumentations](https://albumentations.ai/) for image transforms during training and testing

<!-- ! this example is for writing custom transforms - background randomization https://docs.fast.ai/tutorial.albumentations.html -->
## Folder Structure
```
bitou_segmentation
|   .gitignore - top folder gitignore
|   csupl.txt - the file for creating the environment currently necessary to run the code
|   README.md
|   setup.py - a python package setup file
|
└─── config - where configuration files are sitting that define settings
|       |   colour_code.json -  a file that defines the colours for classes by number. Used for mask decoding
|       |   via_bitou_test1_220928.json - a config file created using VIA
|       |   kmeans_K-3_scale-20_hsv_full.pkl - **gitignored** for file size reasons - current classifier used for k_means prediction
|
└─── data - where pytorchs dataloader expects data - mainly gitignored. Is normally a collection of multiple 
|           folders with the same substructure. This can be defined in the dataloader. Important is the same name in the <masks>
|           and <image> directory
|       |
|       └─── images
|       |       | <files with a name>.<and an extension>
|       |       
|       └─── masks
|               | <masks with the **exact same name as the image**>.<and some extension>
|
└─── results - intermediate files and results. mostly gitignored, see below
|       |   
|
└─── scripts - executable scripts that are intended to run programs. Mainly contain an argument parser and lots of includes from the src directory.
|               can always be run with python <script_name>.py --help to see information
|       |   generate_masks.py - file to create masks from a config file and a directory of base images. Needs output, otherwise will plot images
|       |   train_kmeans.py - file to train a k-means classifier, which can be used for watershed pre-labelling
|       |   predict_kmeans.py - file that will run the prediction from a specified k_means classifier on a specified image or directory
|       |   k_means.sh - shell file to run multiple combinations for the train_kmeans.py version
|       |   predict.py - file for running prediction on a few images and plotting a model - currently not executable with argument parsing
|       |   train_binary_model.py - file for training a binary model. Is split from normal model because of setting differences
|       |   train_model.py - file for training a multiclass model. Standard file to run for training
|       |   rysny_dataset_*.sh - files for synchronising folders to remote connections. Have safeguards in them to protect overwriting
            remote connections without authorisation 
|
└─── src - importable code. Should be installed using pip - see [below](###-Installing-src-files-in-csupl). Can be used by placing
            <import csupl.<filename>> at the top of a file
|       └─── csupl
|               |   __init__.py - necessary for installation
|               |   generate_masks.py - functions that are necessary for generating masks - includes cropping etc.
|               |   k_means.py - functions and classes necessary for running k_means prediction on images
|               |   watershed.py - functions and classes necessary for running watershed prediction - requires k_means.py
|               |   utils.py - general utilities that are used across everywhere. should be first point to import, for consistency across scripts. 
                                includes a lot of plotting files and formats for conversion and type checking
|               |   model.py - model definitions and steps that are used in lightning training
|               |   dataloader.py - dataset and data loading tools for use with pytorch pipeline
|               |   train_utils.py - extra utilities during training. Mostly for logging images and other things
|
└─── tests - tests to check imports and proper function. can be run using `pytest .` inside the directory
|       |   pytest.ini - configuration file for the tests. Used to suppress DeprecationWarnings
|       |   test_imports.py - checks if import and GPU functions are available
|       |   test_dataloader.py - checks to see if the dataloading functions do what they need to
```

### Gitignored files and folders:
```
bitou_segmentation
|
└─── config
|       |   kmeans_K-3_scale-20_hsv_full.pkl - **gitignored** for file size reasons - current classifier used for k_means prediction
|
└─── .vscode - configuration and compilation settings for vscode
|       |   launch.json -  custom launch configurations    
|       |   settings.json - custom settings
|
└─── lightning_logs - checkpoint files created by lightning during training.
|
└─── logs      
|
└─── results
|       └─── tmp - temporary results that are not thought for publication
|              |    <models>.pt - collection of default models
|              |    <k_means_classif>.pkl - collection of classifiers for different k_means settings
|              |    <images> - collections of images created during process
|
└─── tmp - temporary scripts that are used for testing and probing scripts and functions during development
|       |   base_task.py - file that is analogous to flash's segmentation task - base file to start from
|       |   generate_masks.py - file that generates binary masks from a folder
```

## Installing src files in csupl
run `pip install -e .` from base folder. Installs current directory with editable configuration. Can now be imported anywhere

alternative is defining a .ENV file for vscode

## Environment files
Environments can be recreated on the native host using `conda create --name <environment_name> --file <filename>.txt`
The current file to use is: [csupl.txt](csupl.txt)
### Caveats:
* There are known issues with pytorch versions and  are currently two environments available, both working with RTX GPUs (Cuda 11.X), see [pytorch documentation](https://discuss.pytorch.org/t/nvidia-geforce-rtx-3090-with-cuda-capability-sm-86-is-not-compatible-with-the-current-pytorch-installation/141940)

### Testing
#### Environment
Running `pytest .` inside the `tests` directory will ensure the packages are configured correctly

## Generating masks from Json
`.json` files should be created using VGG Image Annotation tool.
[`generate_masks.py`](scripts/generate_masks.py) needs a .json file with the image annotations, an input directory for the original files and an output directory for the masks.

## Training a Model
A model can be trained using the [`train_model.py`](scripts/train_model.py) file.
A model needs specificiations what kind of model it should be. Segmentation models consist of an encoder and a decoder, where smp provides certain default architectures and combinations, see [smp documentation](https://smp.readthedocs.io/en/latest/models.html). It also needs to know how many classes are in the dataset. And it needs to know where the masks and images are stored that contain the masks and images for training.

## Running predictions with a model
A model first needs to be trained, before prediction can happen. Best practice to store in `results` folder
