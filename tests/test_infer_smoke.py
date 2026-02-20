import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.infer import predict
from src.models.rnn_classifier import RNNClassifier


def test_predict_embeddings_rnn_smoke(tmp_path: Path, monkeypatch) -> None:
    input_csv = tmp_path / "tiny_input.csv"
    output_csv = tmp_path / "tiny_output.csv"
    model_path = tmp_path / "tiny_rnn.pt"

    pd.DataFrame({"content": ["هذا جيد", "هذا سيئ"]}).to_csv(input_csv, index=False, encoding="utf-8")

    model = RNNClassifier(
        input_dim=4,
        hidden_dim=2,
        output_dim=2,
        model_type="RNN",
        num_layers=1,
        bidirectional=False,
        dropout=0.0,
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    torch.save(model.state_dict(), model_path)

    metadata = {
        "pipeline": "embeddings_rnn",
        "embedding_model_name": "dummy-model",
        "embedding_max_length": 32,
        "remove_stopwords": False,
        "label2id": {"neg": 0, "pos": 1},
        "id2label": {"0": "neg", "1": "pos"},
        "model_params": {
            "input_dim": 4,
            "hidden_dim": 2,
            "output_dim": 2,
            "model_type": "RNN",
            "num_layers": 1,
            "bidirectional": False,
            "dropout": 0.0,
        },
    }
    model_path.with_suffix(".meta.json").write_text(json.dumps(metadata), encoding="utf-8")

    def fake_build_embeddings(texts, **kwargs):  # noqa: ANN001, ANN003
        return np.ones((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr(predict, "build_embeddings", fake_build_embeddings)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict.py",
            "--pipeline",
            "embeddings_rnn",
            "--model_path",
            str(model_path),
            "--input_csv",
            str(input_csv),
            "--output_csv",
            str(output_csv),
            "--device",
            "cpu",
        ],
    )
    predict.main()

    assert output_csv.exists()
    out_df = pd.read_csv(output_csv)
    assert {"pred_label", "pred_prob"}.issubset(out_df.columns)
    assert len(out_df) == 2

