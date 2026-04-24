#!/usr/bin/env bash
set -euo pipefail

export ***REMOVED***
export ***REMOVED***
export ***REMOVED***
export ***REMOVED***

export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export MASTER_ADDR=10.40.1.8
export MASTER_PORT=29500

torchrun \
  --standalone \
  --nproc_per_node=8 \
  --master_addr=10.40.1.8 \
  --master_port=29500 \
  --local-addr=10.40.1.8 \
  train.py \
  --run_name "mha" \
  --attention_type "mha"