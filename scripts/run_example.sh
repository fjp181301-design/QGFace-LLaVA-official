#!/bin/bash

# Example running script for QGFace-LLaVA
# This script shows the basic command format for running experiments.

DATASET="utkface"
METHOD="qgface_llava"
METADATA_SETTING="clean"
CONFIG="configs/utkface.yaml"

python src/train.py \
  --dataset ${DATASET} \
  --method ${METHOD} \
  --metadata_setting ${METADATA_SETTING} \
  --config ${CONFIG}
