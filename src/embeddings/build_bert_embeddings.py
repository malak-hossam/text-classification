from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def build_embeddings(
    texts: Sequence[str],
    model_name: str = "aubmindlab/bert-base-arabertv02",
    batch_size: int = 16,
    max_length: int = 200,
    device: str = "auto",
    show_progress: bool = True,
) -> np.ndarray:
    torch_device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(torch_device)
    model.eval()

    all_embeddings: list[np.ndarray] = []
    batch_iterator = range(0, len(texts), batch_size)
    if show_progress:
        batch_iterator = tqdm(batch_iterator, desc="Building CLS embeddings", unit="batch")

    for start in batch_iterator:
        batch_texts = list(texts[start : start + batch_size])
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(torch_device) for k, v in tokens.items()}
        outputs = model(**tokens)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        all_embeddings.append(cls_embeddings)

    if not all_embeddings:
        return np.empty((0, 768), dtype=np.float32)
    return np.vstack(all_embeddings).astype(np.float32)


def save_embeddings(array: np.ndarray, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    return path

