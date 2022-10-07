#!/bin/bash


# USAGE FUNCTION
print_usage() {
    printf "\nThis bash script uses rsync to copy a dataset from your machine to
    QCR's GPU Cluster Mercury PC. will only copy the contents of the <dataset>/images 
    folder across.

Example Usage:
    ./rsync_dataset_to_cluster.sh  -d <dataset> -u <username> -n <dataset_name>

Input Arguments
    -d <dataset>        the dataset, in the ground-classification dataset format, to be copied across.
    -u <username>       your QUT username.
    -n <dataset_name>   the unique name given to this dataset.
    -N                  used for new datasets only. Only to be used by James
                            
"
}


# PARSE ARGUMENTS
dataset_path=''
username=''
dataset_name=''
new_dataset=false
while getopts 'Nd:u:n:' flag; do
    case "${flag}" in
        d) dataset_path="${OPTARG}" ;;
        u) username="${OPTARG}" ;;
        n) dataset_name="${OPTARG}" ;;
        N) new_dataset=true ;;
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

# See if directory exists on cluster maching - if not can do straight copy
if ssh ${cluster_pc} '[ -d ${cluster_dir}]'; then
    # get images on cluster and local machines, use these to check if correct dataset name was entered
    local_images=("${dataset_path}/images/images"/*)
    cluster_images=($(ssh ${cluster_pc} 'ls '${cluster_dir}'/images/images/*'))

    local_img="$(basename -- ${local_images[0]})"
    cluster_img="$(basename -- ${cluster_images[0]})"
    if [ ${local_img} != ${cluster_img} ] ; then
        echo "The local dataset ${dataset_path} and the cluster dataset ${cluster_dir} are not the same. Make sure you are specifying the correct local and cluster dataset variables."
        echo "First image on local machine ${local_img}"
        echo "First image on cluster machine ${cluster_img}"
        exit -1
    fi
elif [ "${new_dataset}" = false ]; then # dataset doesn't exist, only run rsync command if -N flag is set
    echo "The ${dataset_name} does not exist on the cluster. If attempting to copy a new dataset make sure the -N flag is set."
    exit -1
fi

# Create rsync command
command="rsync -azP --exclude pointclouds --exclude data_files --exclude .rosdata --exclude .label_ground ${dataset_path}/ ${cluster_pc}:${cluster_dir}"
echo "Running Command: ${command}"
eval ${command}