# Experiments

Use this page to track reproducible runs and compare pipelines.

## Experiment Log Template

| Run ID | Date | Pipeline | Backbone/Variant | Stopwords | Freeze Backbone | Epochs | LR | Val Acc | Test Acc | Notes | Artifacts |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|
| exp_001 | YYYY-MM-DD | embeddings_rnn | bi_lstm | true | n/a | 10 | 1e-3 | TBD | TBD | baseline | outputs/models/bi_lstm_best.pt |
| exp_002 | YYYY-MM-DD | finetune | arabert | true | true | 2 | 2e-5 | TBD | TBD | head-only ft | outputs/hf_runs/arabert_finetune/best_model |
| exp_003 | YYYY-MM-DD | finetune | mbert | true | true | 2 | 2e-5 | TBD | TBD | baseline | outputs/hf_runs/mbert_finetune/best_model |

## Suggested Comparison Dimensions

- AraBERT embeddings + BiLSTM vs full AraBERT fine-tuning.
- Frozen backbone vs unfrozen backbone.
- Effect of stopword removal on each pipeline.
- Error concentration from confusion matrix.

## Minimal Reproducibility Checklist

- Config file committed.
- Seed fixed and logged.
- Metrics JSON saved.
- Model path and tokenizer path recorded.
- Dataset source/version documented.

