from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from src.data.load_data import load_raw_dataset, save_processed_dataset
from src.data.preprocess import preprocess_dataframe
from src.data.split import split_train_test
from src.models.hf_dataset import TextClassificationDataset
from src.models.metrics import hf_compute_metrics, save_json
from src.utils.logging import get_logger, setup_logging
from src.utils.paths import CONFIG_DIR, HF_RUNS_DIR, ensure_project_dirs
from src.utils.seed import set_seed

LOGGER = get_logger(__name__)

MODEL_ALIASES = {
    "arabert": "aubmindlab/bert-base-arabertv02",
    "mbert": "bert-base-multilingual-cased",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune AraBERT or mBERT for sentiment classification.")
    parser.add_argument("--config", type=str, default="src/config/finetune_bert.yaml")
    parser.add_argument("--data_csv", type=str, default=None, help="Optional override for raw dataset path.")
    return parser.parse_args()


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str) -> dict[str, Any]:
    base_cfg = yaml.safe_load((CONFIG_DIR / "default.yaml").read_text(encoding="utf-8")) or {}
    run_cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    return _merge_dict(base_cfg, run_cfg)


def freeze_backbone_parameters(model: AutoModelForSequenceClassification) -> None:
    backbone = getattr(model, model.base_model_prefix, None)
    if backbone is None:
        backbone = model.base_model
    for parameter in backbone.parameters():
        parameter.requires_grad = False


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    ensure_project_dirs()
    set_seed(int(config["seed"]), deterministic=True)

    data_path = args.data_csv or config["data"]["raw_csv"]
    LOGGER.info("Loading raw dataset from %s", data_path)
    df = load_raw_dataset(data_path)

    remove_stop_words = bool(config["preprocess"]["remove_stopwords"])
    df_processed = preprocess_dataframe(df, remove_stop_words=remove_stop_words)
    processed_path = save_processed_dataset(df_processed, Path(config["data"]["processed_csv"]).name)
    LOGGER.info("Saved processed dataset to %s", processed_path)

    train_df, test_df = split_train_test(
        df_processed,
        test_size=float(config["split"]["test_size"]),
        random_state=int(config["split"]["random_state"]),
        stratify_col="label",
    )

    label_values = train_df["label"].astype(str).tolist()
    label_names = sorted(set(label_values))
    label2id = {label: idx for idx, label in enumerate(label_names)}
    id2label = {idx: label for label, idx in label2id.items()}

    train_labels = train_df["label"].astype(str).map(label2id).to_list()
    test_labels = test_df["label"].astype(str).map(label2id).to_list()
    train_texts = train_df["content"].astype(str).to_list()
    test_texts = test_df["content"].astype(str).to_list()

    model_key = str(config["model"]["name"])
    model_name = MODEL_ALIASES.get(model_key, model_key)
    max_length = int(config["model"]["max_length"])
    run_name = str(config["run_name"])
    run_dir = HF_RUNS_DIR / run_name
    best_model_dir = run_dir / "best_model"
    run_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Using model/tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )
    if bool(config["model"].get("freeze_backbone", True)):
        freeze_backbone_parameters(model)
        LOGGER.info("Backbone parameters frozen (classifier head fine-tuning only).")

    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length=max_length)
    eval_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_length=max_length)

    trainer_cfg = dict(config["trainer"])
    trainer_cfg["output_dir"] = str(run_dir)
    trainer_cfg["seed"] = int(config["seed"])
    trainer_cfg["remove_unused_columns"] = False
    training_args = TrainingArguments(**trainer_cfg)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=hf_compute_metrics,
    )

    LOGGER.info("Starting fine-tuning.")
    trainer.train()
    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="eval")
    train_metrics = trainer.evaluate(eval_dataset=train_dataset, metric_key_prefix="train")

    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    metadata = {
        "pipeline": "finetune",
        "base_model_name": model_name,
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "preprocess": {"remove_stopwords": remove_stop_words},
        "max_length": max_length,
    }
    (best_model_dir / "inference_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    metrics = {"train": train_metrics, "eval": eval_metrics}
    save_json(metrics, run_dir / "metrics.json")
    LOGGER.info("Saved best model and tokenizer to %s", best_model_dir)
    LOGGER.info("Saved metrics to %s", run_dir / "metrics.json")


if __name__ == "__main__":
    main()
