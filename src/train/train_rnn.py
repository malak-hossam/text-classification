from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.load_data import load_raw_dataset, save_processed_dataset
from src.data.preprocess import preprocess_dataframe
from src.data.split import split_train_test
from src.embeddings.build_bert_embeddings import build_embeddings, resolve_device, save_embeddings
from src.models.metrics import compute_metrics_bundle, save_confusion_matrix_plot, save_json
from src.models.rnn_classifier import EmbeddingDataset, RNNClassifier
from src.utils.logging import get_logger, setup_logging
from src.utils.paths import CONFIG_DIR, EMBEDDINGS_DIR, METRICS_DIR, MODELS_DIR, PLOTS_DIR, ensure_project_dirs
from src.utils.seed import set_seed

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RNN/LSTM classifiers on AraBERT CLS embeddings.")
    parser.add_argument("--config", type=str, default="src/config/embeddings_rnn.yaml")
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


def encode_labels(labels: pd.Series) -> tuple[np.ndarray, dict[str, int], dict[str, str]]:
    label_strings = labels.astype(str)
    unique_labels = sorted(label_strings.unique().tolist())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {str(idx): label for label, idx in label2id.items()}
    encoded = label_strings.map(label2id).to_numpy(dtype=np.int64)
    return encoded, label2id, id2label


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
    desc: str = "train",
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    iterator = tqdm(loader, desc=desc, leave=False)
    for batch_x, batch_y in iterator:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            if is_train:
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_loss += float(loss.item())
        total_correct += int((preds == batch_y).sum().item())
        total_count += int(batch_y.size(0))

    avg_loss = total_loss / max(len(loader), 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc


@torch.inference_mode()
def predict_labels(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch_x, batch_y in loader:
        logits = model(batch_x.to(device))
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        y_pred.extend(preds)
        y_true.extend(batch_y.numpy().tolist())
    return np.array(y_true, dtype=np.int64), np.array(y_pred, dtype=np.int64)


def train_variant(
    variant_cfg: dict[str, Any],
    common_model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
) -> tuple[Path, float]:
    variant_name = variant_cfg["name"]
    model = RNNClassifier(
        input_dim=x_train.shape[1],
        hidden_dim=int(common_model_cfg["hidden_dim"]),
        output_dim=len(np.unique(y_train)),
        model_type=str(variant_cfg["model_type"]),
        num_layers=int(common_model_cfg["num_layers"]),
        bidirectional=bool(variant_cfg["bidirectional"]),
        dropout=float(common_model_cfg["dropout"]),
    ).to(device)

    train_loader = DataLoader(
        EmbeddingDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(
        EmbeddingDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    epochs = int(train_cfg["num_epochs"])

    best_acc = -1.0
    best_path = MODELS_DIR / f"{variant_name}_best.pt"

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer, desc=f"{variant_name} train {epoch}/{epochs}"
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device, optimizer=None, desc=f"{variant_name} val   {epoch}/{epochs}"
        )
        LOGGER.info(
            "%s | epoch=%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            variant_name,
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)

    return best_path, best_acc


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]), deterministic=True)
    ensure_project_dirs()

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

    y_train, label2id, id2label = encode_labels(train_df["label"])
    y_test = test_df["label"].astype(str).map(label2id).to_numpy(dtype=np.int64)
    x_train_text = train_df["content"].astype(str).tolist()
    x_test_text = test_df["content"].astype(str).tolist()

    emb_cfg = config["embeddings"]
    LOGGER.info("Building AraBERT CLS embeddings for train/test split.")
    x_train_emb = build_embeddings(
        x_train_text,
        model_name=str(emb_cfg["model_name"]),
        batch_size=int(emb_cfg["batch_size"]),
        max_length=int(emb_cfg["max_length"]),
        device=str(config["device"]),
    )
    x_test_emb = build_embeddings(
        x_test_text,
        model_name=str(emb_cfg["model_name"]),
        batch_size=int(emb_cfg["batch_size"]),
        max_length=int(emb_cfg["max_length"]),
        device=str(config["device"]),
    )

    save_embeddings(x_train_emb, EMBEDDINGS_DIR / "X_train_embeddings.npy")
    save_embeddings(x_test_emb, EMBEDDINGS_DIR / "X_test_embeddings.npy")
    np.save(EMBEDDINGS_DIR / "y_train.npy", y_train)
    np.save(EMBEDDINGS_DIR / "y_test.npy", y_test)

    device = resolve_device(str(config["device"]))
    summary_rows: list[dict[str, Any]] = []

    for variant in config["model"]["variants"]:
        variant_name = str(variant["name"])
        LOGGER.info("Training variant: %s", variant_name)
        best_model_path, best_val_acc = train_variant(
            variant_cfg=variant,
            common_model_cfg=config["model"],
            train_cfg=config["train"],
            x_train=x_train_emb,
            y_train=y_train,
            x_val=x_test_emb,
            y_val=y_test,
            device=device,
        )

        model = RNNClassifier(
            input_dim=x_train_emb.shape[1],
            hidden_dim=int(config["model"]["hidden_dim"]),
            output_dim=len(label2id),
            model_type=str(variant["model_type"]),
            num_layers=int(config["model"]["num_layers"]),
            bidirectional=bool(variant["bidirectional"]),
            dropout=float(config["model"]["dropout"]),
        ).to(device)
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        test_loader = DataLoader(
            EmbeddingDataset(torch.tensor(x_test_emb, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
            batch_size=int(config["train"]["batch_size"]),
            shuffle=False,
        )
        y_true, y_pred = predict_labels(model, test_loader, device)
        metrics = compute_metrics_bundle(y_true, y_pred)
        metrics["best_val_accuracy"] = float(best_val_acc)
        metrics["variant"] = variant_name

        plot_path = save_confusion_matrix_plot(
            metrics["confusion_matrix"],
            PLOTS_DIR / f"{variant_name}_confusion_matrix.png",
            class_names=[id2label[str(i)] for i in range(len(id2label))],
        )
        if plot_path:
            metrics["confusion_matrix_plot"] = str(plot_path)

        metrics_path = METRICS_DIR / f"{variant_name}_metrics.json"
        save_json(metrics, metrics_path)

        metadata = {
            "pipeline": "embeddings_rnn",
            "embedding_model_name": str(emb_cfg["model_name"]),
            "embedding_max_length": int(emb_cfg["max_length"]),
            "remove_stopwords": remove_stop_words,
            "label2id": label2id,
            "id2label": id2label,
            "model_params": {
                "input_dim": int(x_train_emb.shape[1]),
                "hidden_dim": int(config["model"]["hidden_dim"]),
                "output_dim": len(label2id),
                "model_type": str(variant["model_type"]),
                "num_layers": int(config["model"]["num_layers"]),
                "bidirectional": bool(variant["bidirectional"]),
                "dropout": float(config["model"]["dropout"]),
            },
        }
        metadata_path = best_model_path.with_suffix(".meta.json")
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        summary_rows.append(
            {
                "variant": variant_name,
                "best_val_accuracy": best_val_acc,
                "test_accuracy": metrics["accuracy"],
                "model_path": str(best_model_path),
                "metrics_path": str(metrics_path),
            }
        )
        LOGGER.info(
            "Finished %s | best_val_accuracy=%.4f test_accuracy=%.4f",
            variant_name,
            best_val_acc,
            metrics["accuracy"],
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("test_accuracy", ascending=False)
    summary_path = METRICS_DIR / "rnn_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    LOGGER.info("Saved model summary to %s", summary_path)


if __name__ == "__main__":
    main()

