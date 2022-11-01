#!/usr/bin/env bash

my_dir="$(dirname "$0")"
. $my_dir/set_env.sh

echo "MMAR_ROOT set to $MMAR_ROOT"
additional_options="$*"

CONFIG_FILE=config/config_train.json
ENVIRONMENT_FILE=config/environment.json
NUM_GPUS=2

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --nnodes=1 --node_rank=0 \
    --master_addr="localhost" --master_port=1234 \
    -m medl.apps.train \
    -m $MMAR_ROOT \
    -c $CONFIG_FILE \
    -e $ENVIRONMENT_FILE \
    --write_train_stats \
    --set \
    print_conf=True \
    epochs=500 \
    learning_rate=0.00001 \
    num_interval_per_valid=20 \
    multi_gpu=True \
    dont_load_ckpt_model=False \
    ${additional_options}
