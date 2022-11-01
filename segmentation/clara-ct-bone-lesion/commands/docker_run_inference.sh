#!/bin/bash

DOCKER_IMAGE=nvcr.io/nvidia/clara-train-sdk:v4.1

DATASET_JSON="/home/mip/clara-models/clara-ct-bone-lesion/config/docker_run_inference.json"
DATASET_LOCATION="/home/mip/clara-models/bone_data"
ENV_FILE="config/environment.json"

docker run --rm -it --network=host --runtime=nvidia --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
       -w /mmar \
       -e NVIDIA_VISIBLE_DEVICES=2 \
       -e CONFIG_FILE=config/config_inference.json \
       -e DATASET_JSON= \
       -e ENVIRONMENT_FILE=${ENV_FILE}\
       -e DATA_ROOT=/dataset \
       -e MMAR_ROOT=/mmar \
       -v /home/mip/clara-models/clara-ct-bone-lesion:/mmar \
       -v ${DATASET_LOCATION}:/dataset \
       ${DOCKER_IMAGE} \
       bash -c 'python3 -u  -m medl.apps.evaluate \
                -m $MMAR_ROOT \
                -c $CONFIG_FILE \
                -e $ENVIRONMENT_FILE \
                --set \
                print_conf=True \
                multi_gpu=False \
                dont_load_ts_model=True \
                dont_load_ckpt_model=False'
