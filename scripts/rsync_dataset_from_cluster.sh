#!/bin/bash


# USAGE FUNCTION
print_usage() {
    printf "\nThis bash script uses rsync to copy a dataset from the
    QCR's GPU Cluster Mercury PC to your machine. The script will only 
    copy the contents of the manual_labels folder across.

Example Usage:
    ./rsync_dataset_from_cluster.sh  -d <dataset> -u <username> -n <dataset_name>

Input Arguments
    -d <dataset>        the dataset location on the local machine.
    -u <username>       your QUT username.
    -n <dataset_name>   the unique name given to this dataset.
                            
"
}


# PARSE ARGUMENTS
dataset_path=''
username=''
dataset_name=''
while getopts 'd:u:n:' flag; do
    case "${flag}" in
        d) dataset_path="${OPTARG}" ;;
        u) username="${OPTARG}" ;;
        n) dataset_name="${OPTARG}" ;;
        *) print_usage
            exit -1 ;;
    esac
done

if [ -z "${dataset_path}" ]; then
    echo "The script requires an input dataset."
    print_usage
    exit -1
fi

if [ -z "${username}" ]; then
    echo "The script requires an username."
    print_usage
    exit -1
fi

if [ -z "${dataset_name}" ]; then
    echo "The script requires the name of the dataset."
    print_usage
    exit -1
fi


# Setup Computer and directory
cluster_pc="${username}@venus.qut.edu.au"
cluster_dir="/home/${username}/agkelpie/datasets/${dataset_name}"

# Create rsync command
command="rsync -azP --exclude images --exclude auto_masks ${cluster_pc}:${cluster_dir}/images/ ${dataset_path}/images/"
echo "Running Command: ${command}"
eval ${command}