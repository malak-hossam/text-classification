#!/usr/bin/env bash
set -euo pipefail

python -m src.train.train_rnn --config src/config/embeddings_rnn.yaml "$@"

