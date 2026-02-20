# Model Card

## Models in This Repository

1. Embedding-based recurrent classifiers:
   - RNN
   - LSTM
   - BiRNN
   - BiLSTM
2. Fine-tuned transformers:
   - AraBERT (`aubmindlab/bert-base-arabertv02`)
   - Optional mBERT baseline (`bert-base-multilingual-cased`)

## Intended Use

- Arabic sentiment classification on short to medium user-generated text.
- Benchmarking feature-based vs fine-tuning-based pipelines.

## Inputs

- UTF-8 Arabic text in CSV column `content`.

## Outputs

- `pred_label`: predicted sentiment class.
- `pred_prob`: max softmax confidence for predicted class.

## Training Setup

- Shared preprocessing pipeline with configurable stopword removal.
- Train/test split: 70/30 stratified.
- Metrics: accuracy, classification report, confusion matrix.

## Fine-Tuning Notes

- Tokenizer/model are intentionally matched by selected backbone.
- Optional backbone freezing to train classifier head only.

## Performance Reporting

Populate experiment tables in `docs/experiments.md` for:

- run config
- validation/test metrics
- artifact paths

## Risks

- Class imbalance can bias toward majority sentiment classes.
- Confidence is not calibrated probability by default.
- Domain shift can degrade performance significantly.

## Ethical and Operational Constraints

- Do not use as a sole decision-maker in sensitive workflows.
- Add human oversight and bias audits before deployment.

