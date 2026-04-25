#!/usr/bin/env bash
set -euo pipefail

# Example usage:
# bash ./train.sh --run_name mha --attention_type mha

# Load secrets / local config from .env if present.
# Anything defined there (MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD,
# MLFLOW_TRACKING_URI, WANDB_API_KEY, HF_TOKEN, ...) is auto-exported to the
# environment that torchrun and train.py inherit.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
else
  echo "warning: $ENV_FILE not found — continuing without it" >&2
fi

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
  "$@"