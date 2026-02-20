# System Design

## Objective

Build a reproducible Arabic sentiment classification system with two interchangeable modeling paths:

1. Feature-based modeling via AraBERT CLS embeddings + recurrent classifier.
2. End-to-end transformer fine-tuning with Hugging Face Trainer.

## Components

- `src/data/`: dataset load, cleaning, normalization, split.
- `src/embeddings/`: batched CLS embedding extraction from AraBERT.
- `src/models/`: recurrent classifier, HF dataset wrapper, metrics.
- `src/train/`: pipeline entrypoints for RNN and fine-tune workflows.
- `src/infer/`: unified prediction CLI for both pipelines.
- `src/utils/`: paths, logging, seeding.

## Data Flow

1. Load CSV from `data/raw/arabic_sentiment_reviews.csv`.
2. Validate required columns (`content`, `label`).
3. Preprocess text:
   - stopword removal (optional via config)
   - Arabic-only cleaning
   - Arabic normalization
4. Stratified train/test split (default 70/30, random_state=42).
5. Train pipeline A or B.
6. Save artifacts to `outputs/`.

## Pipeline A: Embeddings + RNN Family

1. Build embeddings from `last_hidden_state[:, 0, :]` using AraBERT.
2. Train variants:
   - RNN
   - LSTM
   - BiRNN
   - BiLSTM
3. Evaluate on test split.
4. Save:
   - model weights (`outputs/models/*.pt`)
   - metrics JSON (`outputs/metrics/`)
   - confusion matrix plot (`outputs/plots/`)
   - metadata sidecar for inference reproducibility.

## Pipeline B: Transformer Fine-Tuning

1. Build tokenized datasets with `TextClassificationDataset`.
2. Select matching tokenizer+model:
   - AraBERT: `aubmindlab/bert-base-arabertv02`
   - mBERT: `bert-base-multilingual-cased`
3. Optional `freeze_backbone=true` to train only classifier head.
4. Train/evaluate with `Trainer`.
5. Save:
   - checkpoints and best model (`outputs/hf_runs/<run_name>/`)
   - metrics and inference config.

## Inference Design

`src.infer.predict` supports:

- `--pipeline embeddings_rnn`:
  - loads `.pt` model + sidecar metadata
  - preprocesses input text consistently
  - rebuilds CLS embeddings
  - outputs `pred_label`, `pred_prob`

- `--pipeline finetune`:
  - loads HF model folder
  - tokenizes and runs logits forward pass
  - applies softmax and writes predictions

## Reproducibility

- Seeded RNG for Python, NumPy, Torch.
- cuDNN deterministic configuration enabled.
- All key parameters controlled via YAML config.
- Label mappings persisted with artifacts.

## MLOps-Lite Practices

- Scripted CLI entrypoints.
- CI with import and unit smoke tests.
- Dockerized environment for repeatable execution.
- Structured artifact directories for experiment auditability.

## Risks and Tradeoffs

- Freezing backbone is faster but may underfit domain nuances.
- Arabic normalization can reduce sparsity but may remove stylistic cues.
- CLS-only embeddings are efficient but may lose token-level detail.
- Heavy transformer runs require GPU for practical training time.

