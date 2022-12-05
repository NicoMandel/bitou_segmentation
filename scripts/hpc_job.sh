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

# Model Settings
classes=3		# number of classes in test set
model_name="FPN"
encoder_name="resnet34"
encoder_weights="imagenet"

# Training settings
workers=4		# default = 4 * nGPUS
batch=16		# default batch size

# Dataset Settings
data_dir="/work/quteagles/data_bitou"

# Output Settings
output_dir = "results/"

for m in 1 2
do
	this_script_file="bitou_seg_model-${m}.sh"
	# HPC required header
	echo ${header_1} >> ${this_script_file}
	echo "#PBS -N bitou_segm-${m}" >> ${this_script_file}
	echo ${header_2} >> ${this_script_file}
	echo ${header_3} >> ${this_script_file}
	echo ${header_5} >> ${this_script_file}
	echo ${header_6} >> ${this_script_file}
	
	# body - loading conda and directory
	echo ${body_1} >> ${this_script_file}
	echo ${body_2} >> ${this_script_file}
	
	# Actual command to execute
	echo "python scripts/train_model.py -c ${classes} -w ${workers} -b ${batch} -m ${m} -s -e 25" >> ${this_script_file}
	qsub ${this_script_file}
	
	#cleanup
	# rm ${this_script_file}
done

