# Data Card

## Dataset Name

330K Arabic Sentiment Reviews

## Source

Kaggle: https://www.kaggle.com/datasets/abdallaellaithy/330k-arabic-sentiment-reviews

## File Used

- `arabic_sentiment_reviews.csv`
- Required columns:
  - `content`: raw Arabic review text
  - `label`: sentiment class

## Data Access in This Repo

Dataset is not committed to Git. Place manually at:

- `data/raw/arabic_sentiment_reviews.csv`

## Preprocessing Applied

1. Optional Arabic stopword removal (NLTK stopwords).
2. Keep Arabic Unicode range only.
3. Normalize Arabic characters:
   - `إأآا -> ا`
   - `ة -> ه`
   - `ى -> ي`
4. Remove diacritics.
5. Normalize whitespace.

## Splitting

- Train/test split: 70/30
- `random_state=42`
- Stratified by `label`

## Known Limitations

- Label quality depends on original source annotation quality.
- Text may include dialectal and code-switched Arabic.
- Distributional bias can exist across domains/platforms.
- Stopword and normalization choices may impact nuanced sentiment cues.

## Intended Use

- Research and prototyping for Arabic sentiment classification.
- Baseline development before task-specific domain adaptation.

## Out-of-Scope / Caution

- High-stakes moderation decisions without human review.
- Claims of demographic fairness without dedicated bias evaluation.

