#!/usr/bin/env bash
set -euo pipefail

python -m src.train.train_finetune --config src/config/finetune_bert.yaml "$@"

