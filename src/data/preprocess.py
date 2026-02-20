from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

import nltk
import pandas as pd
from nltk.corpus import stopwords

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)
ARABIC_KEEP_RE = re.compile(r"[^\u0600-\u06FF\s]")
WHITESPACE_RE = re.compile(r"\s+")
DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670]")


@lru_cache(maxsize=1)
def get_arabic_stopwords() -> set[str]:
    try:
        return set(stopwords.words("arabic"))
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words("arabic"))
        except Exception as exc:
            LOGGER.warning("Could not load NLTK Arabic stopwords; continuing without them: %s", exc)
            return set()


def remove_stopwords(content: str, stop_words: set[str]) -> str:
    return " ".join([word for word in str(content).split() if word not in stop_words])


def clean_content(content: str) -> str:
    text = ARABIC_KEEP_RE.sub(" ", str(content))
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_arabic(text: str) -> str:
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ى", "ي", text)
    text = DIACRITICS_RE.sub("", text)
    return text


def preprocess_text(
    text: str,
    remove_stop_words: bool = True,
    stop_words: set[str] | None = None,
) -> str:
    stop_words = stop_words if stop_words is not None else get_arabic_stopwords()
    processed = str(text)
    if remove_stop_words:
        processed = remove_stopwords(processed, stop_words)
    processed = clean_content(processed)
    processed = normalize_arabic(processed)
    return processed


def preprocess_dataframe(df: pd.DataFrame, remove_stop_words: bool = True) -> pd.DataFrame:
    if "content" not in df.columns:
        raise ValueError("Expected a `content` column for preprocessing.")
    stop_words = get_arabic_stopwords() if remove_stop_words else set()
    out = df.copy()
    out["content"] = out["content"].astype(str).apply(
        lambda text: preprocess_text(text, remove_stop_words=remove_stop_words, stop_words=stop_words)
    )
    return out


def preprocess_texts(texts: Iterable[str], remove_stop_words: bool = True) -> list[str]:
    stop_words = get_arabic_stopwords() if remove_stop_words else set()
    return [
        preprocess_text(text, remove_stop_words=remove_stop_words, stop_words=stop_words)
        for text in texts
    ]

