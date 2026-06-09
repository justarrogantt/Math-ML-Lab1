import csv
import re
from collections import Counter
from pathlib import Path
from urllib.request import urlretrieve

from datasets import load_dataset

TRAIN_SIZE = 8000
TEST_SIZE = 2000
VOCAB_SIZE = 10000
MAX_TOKENS = 80

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
AG_NEWS_TRAIN_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
AG_NEWS_TEST_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return tokens


def build_vocab(texts, vocab_size=VOCAB_SIZE):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    special_tokens = [PAD_TOKEN, UNK_TOKEN]
    most_common = [word for word, _ in counter.most_common(max(vocab_size - len(special_tokens), 0))]

    idx_to_word = special_tokens + most_common
    word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}
    return word_to_idx, idx_to_word


def encode_text(text, word_to_idx, max_tokens=MAX_TOKENS):
    tokens = tokenize(text)[:max_tokens]
    if not tokens:
        tokens = [UNK_TOKEN]
    unk_idx = word_to_idx[UNK_TOKEN]
    return [word_to_idx.get(token, unk_idx) for token in tokens]


def _load_ag_news_with_datasets():
    dataset = load_dataset("ag_news")
    train_records = dataset["train"]
    test_records = dataset["test"]

    train_texts = list(train_records["text"])
    y_train = list(train_records["label"])
    test_texts = list(test_records["text"])
    y_test = list(test_records["label"])
    return train_texts, y_train, test_texts, y_test


def _read_csv_records(csv_path):
    texts = []
    labels = []
    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 3:
                continue
            label = int(row[0]) - 1
            title = row[1]
            description = row[2]
            texts.append(f"{title} {description}")
            labels.append(label)
    return texts, labels


def _load_ag_news_from_local_csv():
    base_dir = Path(__file__).resolve().parent
    train_candidates = [
        base_dir / "ag_news_train.csv",
        base_dir / "data" / "ag_news_train.csv",
        base_dir / "train.csv",
    ]
    test_candidates = [
        base_dir / "ag_news_test.csv",
        base_dir / "data" / "ag_news_test.csv",
        base_dir / "test.csv",
    ]

    train_path = next((path for path in train_candidates if path.exists()), None)
    test_path = next((path for path in test_candidates if path.exists()), None)

    if train_path is None or test_path is None:
        data_dir = base_dir / "data"
        data_dir.mkdir(exist_ok=True)
        downloaded_train_path = data_dir / "ag_news_train.csv"
        downloaded_test_path = data_dir / "ag_news_test.csv"

        print("Локальные CSV не найдены. Скачиваю резервные файлы AG News...")
        urlretrieve(AG_NEWS_TRAIN_URL, downloaded_train_path)
        urlretrieve(AG_NEWS_TEST_URL, downloaded_test_path)

        train_path = downloaded_train_path
        test_path = downloaded_test_path

    if train_path is None or test_path is None:
        raise FileNotFoundError(
            "Не удалось загрузить AG News через datasets и не найдены локальные CSV-файлы "
            "train/test рядом с проектом."
        )

    train_texts, y_train = _read_csv_records(train_path)
    test_texts, y_test = _read_csv_records(test_path)
    return train_texts, y_train, test_texts, y_test


def load_ag_news_data(
    train_size=TRAIN_SIZE,
    test_size=TEST_SIZE,
    vocab_size=VOCAB_SIZE,
    max_tokens=MAX_TOKENS,
):
    try:
        train_texts, y_train, test_texts, y_test = _load_ag_news_with_datasets()
        source = "datasets"
    except Exception as error:
        print(f"Предупреждение: не удалось загрузить AG News через datasets: {error}")
        print("Пробую резервный вариант через локальные CSV-файлы.")
        train_texts, y_train, test_texts, y_test = _load_ag_news_from_local_csv()
        source = "csv"

    train_texts = train_texts[:train_size]
    y_train = y_train[:train_size]
    test_texts = test_texts[:test_size]
    y_test = y_test[:test_size]

    word_to_idx, idx_to_word = build_vocab(train_texts, vocab_size=vocab_size)
    train_texts_encoded = [encode_text(text, word_to_idx, max_tokens=max_tokens) for text in train_texts]
    test_texts_encoded = [encode_text(text, word_to_idx, max_tokens=max_tokens) for text in test_texts]

    print(
        f"Источник данных: {source}. "
        f"Размер train: {len(train_texts_encoded)}, test: {len(test_texts_encoded)}, "
        f"словарь: {len(word_to_idx)}"
    )

    return train_texts_encoded, y_train, test_texts_encoded, y_test, word_to_idx, idx_to_word
