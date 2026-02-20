from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.preprocess import preprocess_texts
from src.embeddings.build_bert_embeddings import build_embeddings, resolve_device
from src.models.rnn_classifier import RNNClassifier
from src.utils.logging import get_logger, setup_logging

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sentiment inference.")
    parser.add_argument("--pipeline", choices=["embeddings_rnn", "finetune"], required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def _load_input(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "content" not in df.columns:
        raise ValueError("Input CSV must contain a `content` column.")
    return df


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _lookup_label(id2label: Any, idx: int) -> str:
    if isinstance(id2label, dict):
        if idx in id2label:
            return str(id2label[idx])
        if str(idx) in id2label:
            return str(id2label[str(idx)])
    return str(idx)


@torch.inference_mode()
def run_embeddings_rnn_inference(
    model_path: str | Path,
    input_df: pd.DataFrame,
    batch_size: int = 32,
    device: str = "auto",
) -> pd.DataFrame:
    model_path = Path(model_path)
    meta_path = model_path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Expected embeddings metadata file next to model: {meta_path}")
    metadata = _read_json(meta_path)

    model_params = metadata["model_params"]
    torch_device = resolve_device(device)

    model = RNNClassifier(
        input_dim=int(model_params["input_dim"]),
        hidden_dim=int(model_params["hidden_dim"]),
        output_dim=int(model_params["output_dim"]),
        model_type=str(model_params["model_type"]),
        num_layers=int(model_params["num_layers"]),
        bidirectional=bool(model_params["bidirectional"]),
        dropout=float(model_params["dropout"]),
    ).to(torch_device)
    model.load_state_dict(torch.load(model_path, map_location=torch_device))
    model.eval()

    processed_texts = preprocess_texts(
        input_df["content"].astype(str).tolist(),
        remove_stop_words=bool(metadata.get("remove_stopwords", True)),
    )
    embeddings = build_embeddings(
        processed_texts,
        model_name=str(metadata["embedding_model_name"]),
        batch_size=batch_size,
        max_length=int(metadata.get("embedding_max_length", 200)),
        device=device,
    )

    probs_list: list[float] = []
    pred_ids: list[int] = []

    for start in tqdm(range(0, len(embeddings), batch_size), desc="Predict embeddings_rnn", unit="batch"):
        batch = torch.tensor(embeddings[start : start + batch_size], dtype=torch.float32).to(torch_device)
        logits = model(batch)
        probs = softmax(logits, dim=-1)
        pred_prob, pred_idx = torch.max(probs, dim=-1)
        probs_list.extend(pred_prob.cpu().tolist())
        pred_ids.extend(pred_idx.cpu().tolist())

    id2label = metadata.get("id2label", {})
    pred_labels = [_lookup_label(id2label, idx) for idx in pred_ids]

    out = input_df.copy()
    out["pred_label"] = pred_labels
    out["pred_prob"] = probs_list
    return out


@torch.inference_mode()
def run_finetune_inference(
    model_path: str | Path,
    input_df: pd.DataFrame,
    batch_size: int = 32,
    max_length: int = 128,
    device: str = "auto",
) -> pd.DataFrame:
    model_dir = Path(model_path)
    infer_cfg_path = model_dir / "inference_config.json"
    infer_cfg = _read_json(infer_cfg_path) if infer_cfg_path.exists() else {}
    remove_stopwords = bool(infer_cfg.get("preprocess", {}).get("remove_stopwords", True))
    max_length = int(infer_cfg.get("max_length", max_length))

    processed_texts = preprocess_texts(input_df["content"].astype(str).tolist(), remove_stop_words=remove_stopwords)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    torch_device = resolve_device(device)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(torch_device)
    model.eval()
    pred_ids: list[int] = []
    pred_probs: list[float] = []

    for start in tqdm(range(0, len(processed_texts), batch_size), desc="Predict finetune", unit="batch"):
        batch_texts = processed_texts[start : start + batch_size]
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(torch_device) for k, v in tokens.items()}
        logits = model(**tokens).logits
        probs = softmax(logits, dim=-1)
        best_prob, best_idx = torch.max(probs, dim=-1)
        pred_ids.extend(best_idx.cpu().tolist())
        pred_probs.extend(best_prob.cpu().tolist())

    id2label = getattr(model.config, "id2label", None) or infer_cfg.get("id2label", {})
    pred_labels = [_lookup_label(id2label, idx) for idx in pred_ids]

    out = input_df.copy()
    out["pred_label"] = pred_labels
    out["pred_prob"] = pred_probs
    return out


def main() -> None:
    setup_logging()
    args = parse_args()
    input_df = _load_input(args.input_csv)

    if args.pipeline == "embeddings_rnn":
        result_df = run_embeddings_rnn_inference(
            model_path=args.model_path,
            input_df=input_df,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        result_df = run_finetune_inference(
            model_path=args.model_path,
            input_df=input_df,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved predictions to %s", output_path)


if __name__ == "__main__":
    main()
