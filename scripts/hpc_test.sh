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
ncpus=4
header_5="#PBS -l ngpus=1"
header_6="#PBS -l ncpus=${ncpus}"

# Body for the hpc script
# conda and directory
body_1="conda activate csupl"
body_2="cd /work/quteagles/bitou_segmentation"

## Actual python settings below
# Model Settings
model_file="results/model_filename.pt"
height=512
width=512

# Dataloader settings
workers=$ncpus	# default = 4 * nGPUS
batch=16		# default batch size

# Dataset Settings
test_dir="/work/quteagles/data_bitou/multiclass_1"

this_script_file="bitou_test_${model_file}.sh"
# HPC required header
echo ${header_1} >> ${this_script_file}
echo "#PBS -N bitou_test-${model_file}" >> ${this_script_file}
echo ${header_2} >> ${this_script_file}
echo ${header_3} >> ${this_script_file}
echo ${header_5} >> ${this_script_file}
echo ${header_6} >> ${this_script_file}

# body - loading conda and directory
echo ${body_1} >> ${this_script_file}
echo ${body_2} >> ${this_script_file}

# Actual command to execute
echo "python scripts/train_model.py -m ${model_file}\
-w ${width} -h ${height}\
--workers ${workers} -b ${batch}\
-i ${data_dir}" >> ${this_script_file}
qsub ${this_script_file}

#cleanup
# rm ${this_script_file}

