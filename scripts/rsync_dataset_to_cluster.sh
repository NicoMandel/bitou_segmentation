#!/bin/bash


# USAGE FUNCTION
print_usage() {
    printf "\nThis bash script uses rsync to copy a dataset from your machine to
    QUTs HPC cluster. Requires the names of the subfolder for images and masks,
    as well as the dataset name
    The data will land in /work/quteagles/data_bitou/<name>, with subfolders <images> and <masks>
    Please ensure you have access to the /quteagles workspace by requesting it from the HPC team.
    To overwrite a folder, 
    

Example Usage:
    ./rsync_dataset_to_cluster.sh  -d <dataset> -u <username> -n <dataset_name>

Input Arguments
    -u <username>       your QUT username for HPC access
    -m <masks>          the masks folder
    -i <input>          the input folder of raw images
    -n <name>           the dataset name - which is the name of the subfolder
    -N                  flag for new dataset creation
"
}


# PARSE ARGUMENTS
username=''
image_path=''
mask_path=''
dataset_name=''
new_dataset=false
while getopts 'u:m:i:n:N' flag; do
    case "${flag}" in
        u) username="${OPTARG}" ;;
        i) image_path="${OPTARG}" ;;
        m) mask_path="${OPTARG}" ;;
        n) dataset_name="${OPTARG}" ;;
        N) new_dataset=true ;;
        *) print_usage
            exit -1 ;;
    esac
done

# Checks on Syntax
if [ -z "${image_path}" ]; then
    echo "The script requires an input image folder."
    print_usage
    exit -1
fi

if [ -z "${username}" ]; then
    echo "The script requires an username."
    print_usage
    exit -1
fi

if [ -z "${mask_path}" ]; then
    echo "The script requires an input mask folder."
    print_usage
    exit -1
fi

if [ -z "${dataset_name}" ]; then
    echo "The script requires a dataset name."
    print_usage
    exit -1
fi

# Setup Computer and directory
cluster_pc="${username}@lyra.qut.edu.au"
cluster_dir="/work/quteagles/data_bitou/${dataset_name}"
# Setup directories
cluster_mask_dir="/work/quteagles/data_bitou/${dataset_name}/masks"
cluster_img_dir="/work/quteagles/data_bitou/${dataset_name}/images"

check_images() {
    local local_images=("${image_path}/*")
    local local_masks=("${mask_path}"/*)

    local local_img="$(basename -- ${local_images[0]})"
    local cluster_img="$(basename -- ${cluster_images[0]})"
    local local_mask="$(basename -- ${local_masks[0]})"

    local cluster_images=($(ssh -q ${cluster_pc} 'ls '${cluster_dir}'/images'/*))
    local cluster_masks=($(ssh -q ${cluster_pc} 'ls '${cluster_dir}'/masks'/*))

    # check if local image name and mask image name are the same
    if [ ${local_img} != ${local_mask} ] ; then
        echo "Images and Masks do not have the same names."
        echo "Image 1 name: ${local_img}    Mask 1 name: ${cluster_img}"
        exit -1
    else
        echo "Images and Masks have same name. Continuing"
    fi
    
    # check if local images and cluster images are the same
    if [ ${local_img} != ${cluster_img} ] ; then
        echo "Local dataset ${image_path} and cluster dataset ${cluster_dir} are not the same."
        echo "Local Image 1 name: ${local_img}      Cluster Image 1 name: ${cluster_img}"
        exit -1
    else
        echo "Local Image Directory and Cluster Image Directory have same first file. Continuing."
    fi
}

# See if directory exists on cluster machine - if not can do straight copy
if ssh -q ${cluster_pc} "[ -d ${cluster_dir} ]"; then
    echo "Directory ${cluster_dir} exists. Checking image similarity"
    check_images
elif [ "${new_dataset}" = false ]; then # dataset doesn't exist, only run rsync command if -N flag is set
    echo "The ${dataset_name} does not exist on the cluster. If attempting to copy a new dataset make sure the -N flag is set."
    exit -1
elif [ "${new_dataset}" = true ]; then
    echo "Directory ${cluster_dir} does not exist. Creating"
    ssh -q ${cluster_pc} "mkdir -p ${cluster_dir}"
    echo "Created directory ${cluster_dir} on HPC"
fi

# Create rsync command
command_imgs="rsync -azP --no-motd ${image_path}/ ${cluster_pc}:${cluster_img_dir}"
echo "Syncing Images with Command: ${command_imgs}"
eval ${command_imgs}

command_masks="rsync -azP --no-motd ${mask_path}/ ${cluster_pc}:${cluster_mask_dir}"
echo "Syncing Masks with Command: ${command_masks}"
eval ${command_masks}