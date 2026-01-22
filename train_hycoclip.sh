#!/bin/bash
export PYTHONPATH=$(pwd)
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=3
export GLOG_minloglevel=3

torchrun --nproc_per_node=4 train/train_hycoclip.py --json_root /SHARE_ST/icl/hyperbolic/ShareGPT4V/LayoutSAM/json --image_root /SHARE_ST/icl/hyperbolic/ShareGPT4V/data/sam/images 

