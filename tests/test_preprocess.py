from src.data.preprocess import clean_content, normalize_arabic, preprocess_text


def test_clean_content_keeps_arabic_only() -> None:
    text = "Hello! هذا نص 123 !!!"
    cleaned = clean_content(text)
    assert "Hello" not in cleaned
    assert "123" not in cleaned
    assert cleaned == "هذا نص"


def test_normalize_arabic_maps_characters_and_removes_diacritics() -> None:
    text = "إأآا ة ى سَلَامٌ"
    normalized = normalize_arabic(text)
    assert normalized == "اااا ه ي سلام"


def test_preprocess_text_stopwords_toggle() -> None:
    text = "هذا فيلم رائع"
    custom_stopwords = {"هذا"}
    with_stopwords_removed = preprocess_text(
        text,
        remove_stop_words=True,
        stop_words=custom_stopwords,
    )
    without_stopwords_removed = preprocess_text(
        text,
        remove_stop_words=False,
        stop_words=custom_stopwords,
    )
    assert with_stopwords_removed == "فيلم رائع"
    assert without_stopwords_removed == "هذا فيلم رائع"
