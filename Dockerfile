FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && python -m nltk.downloader stopwords

COPY . /app

RUN mkdir -p /app/data/raw /app/data/processed /app/outputs

CMD ["python", "-m", "src.train.train_rnn", "--help"]

