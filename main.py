from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import load_ag_news_data
from train import train

TRAIN_SIZE = 8000
TEST_SIZE = 2000
VOCAB_SIZE = 10000
MAX_TOKENS = 80
EMBEDDING_DIM = 50
NUM_CLASSES = 4
EPOCHS = 10
LEARNING_RATE = 0.05
SEED = 42

CLASS_NAMES = {
    0: "World / политика",
    1: "Sports / спорт",
    2: "Business / бизнес",
    3: "Sci/Tech / технологии",
}


def decode_indices(indices, idx_to_word):
    words = [idx_to_word[idx] for idx in indices if idx < len(idx_to_word)]
    return " ".join(words)


def save_plots(history, output_dir):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["test_loss"], marker="s", label="Test Loss")
    plt.title("Loss по эпохам")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "train_loss.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["test_loss"], marker="o", color="tab:red", label="Test Loss")
    plt.title("Test Loss по эпохам")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "test_loss.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], marker="o", label="Train Accuracy")
    plt.plot(epochs, history["test_accuracy"], marker="s", label="Test Accuracy")
    plt.title("Accuracy по эпохам")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy.png")
    plt.close()


def show_examples(model, test_texts, y_test, idx_to_word, num_examples=5):
    print("\nПримеры предсказаний:\n")
    for i in range(min(num_examples, len(test_texts))):
        predicted_class, probs = model.predict(test_texts[i])
        decoded_text = decode_indices(test_texts[i], idx_to_word)
        print(f"Пример {i + 1}")
        print(f"Текст: {decoded_text}")
        print(f"Истинный класс: {CLASS_NAMES[y_test[i]]}")
        print(f"Предсказанный класс: {CLASS_NAMES[predicted_class]}")
        print(f"Вероятности: {np.round(probs, 4)}")
        print("-" * 80)


def main():
    np.random.seed(SEED)
    output_dir = Path(__file__).resolve().parent

    print("Загрузка и подготовка данных AG News...")
    train_texts, y_train, test_texts, y_test, word_to_idx, idx_to_word = load_ag_news_data(
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        vocab_size=VOCAB_SIZE,
        max_tokens=MAX_TOKENS,
    )

    print("Запуск обучения модели...")
    model, history = train(
        train_texts=train_texts,
        y_train=y_train,
        test_texts=test_texts,
        y_test=y_test,
        vocab_size=len(word_to_idx),
        embedding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        seed=SEED,
    )

    final_test_accuracy = history["test_accuracy"][-1]
    print(f"\nИтоговая test accuracy: {final_test_accuracy:.4f}")

    save_plots(history, output_dir)
    print("Графики сохранены: train_loss.png, test_loss.png, accuracy.png")

    show_examples(model, test_texts, y_test, idx_to_word, num_examples=5)


if __name__ == "__main__":
    main()
