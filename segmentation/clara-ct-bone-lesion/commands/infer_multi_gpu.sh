#!/usr/bin/env bash

my_dir="$(dirname "$0")"
. $my_dir/set_env.sh

echo "MMAR_ROOT set to $MMAR_ROOT"

CONFIG_FILE=config/config_inference.json
ENVIRONMENT_FILE=config/environment.json
NUM_GPUS=2

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --nnodes=1 --node_rank=0 \
    --master_addr="localhost" --master_port=1234 \
    -m medl.apps.evaluate \
    -m $MMAR_ROOT \
    -c $CONFIG_FILE \
    -e $ENVIRONMENT_FILE \
    --set \
    print_conf=True \
    multi_gpu=True \
    dont_load_ts_model=False \
    dont_load_ckpt_model=True
