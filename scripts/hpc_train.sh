#!/bin/bash -l

# These are the limits for the hpc script
memory_limit="10000mb"
time_limit="4:00:00"
# cputype="6140"
gputype="A100" # Other types: V100 - 16 / 32 GB, P100 - 16 GB, T4 - 16GB, A100 - 40 GB - 
header_1="#!/bin/bash -l"
header_2="#PBS -l walltime=${time_limit}"
header_3="#PBS -l mem=${memory_limit}"
# header_4 = "#PBS -l cputype=${cputype}"
header_5="#PBS -l ngpus=1"
header_6="#PBS -l ncpus=4"

# Body for the hpc script
# conda and directory
body_1="conda activate csupl"
body_2="cd /work/quteagles/bitou_segmentation"


## Actual python settings below
# Model Settings
classes=3		# number of classes in test set
model_name="FPN"
encoder_name="resnet34"
encoder_weights="imagenet"

height=512
width=512

# Training settings
workers=4		# default = 4 * nGPUS
batch=16		# default batch size
epochs=30
freeze_bool=true
freeze=''
if [ ${freeze_bool} = true ]
    freeze="--freeze"
else
    freeze=''
fi

# Dataset Settings
data_dir="/work/quteagles/data_bitou/multiclass_1"

# Output Settings
output_dir = "results/"

this_script_file="bitou_${model_name}_${encoder_name}.sh"
# HPC required header
echo ${header_1} >> ${this_script_file}
echo "#PBS -N bitou_segm-${model_name}" >> ${this_script_file}
echo ${header_2} >> ${this_script_file}
echo ${header_3} >> ${this_script_file}
echo ${header_5} >> ${this_script_file}
echo ${header_6} >> ${this_script_file}

# body - loading conda and directory
echo ${body_1} >> ${this_script_file}
echo ${body_2} >> ${this_script_file}

# Actual command to execute
echo "python scripts/train_model.py -c ${classes} -m ${model_name} --encoder ${encoder_name} --weights ${weights}\
--width ${width} --height ${height}
-workers ${workers} -b ${batch} -e ${epochs} ${freeze}\
-i ${data_dir} -o ${output_dir}" >> ${this_script_file}
qsub ${this_script_file}

#cleanup
# rm ${this_script_file}

