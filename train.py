import numpy as np

from model import EmbeddingNewsClassifier


def evaluate(model, texts, labels):
    total_loss = 0.0
    correct = 0

    for word_indices, y_true in zip(texts, labels):
        probs, loss, _ = model.forward(word_indices, y_true)
        total_loss += loss
        predicted_class = int(np.argmax(probs))
        if predicted_class == y_true:
            correct += 1

    mean_loss = total_loss / len(texts)
    accuracy = correct / len(texts)
    return mean_loss, accuracy


def train(
    train_texts,
    y_train,
    test_texts,
    y_test,
    vocab_size,
    embedding_dim=50,
    num_classes=4,
    epochs=10,
    learning_rate=0.05,
    seed=42,
):
    model = EmbeddingNewsClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    train_indices = np.arange(len(train_texts))

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    for epoch in range(epochs):
        rng.shuffle(train_indices)
        epoch_loss = 0.0
        epoch_correct = 0

        for idx in train_indices:
            word_indices = train_texts[idx]
            y_true = y_train[idx]

            probs, loss, cache = model.forward(word_indices, y_true)
            grads = model.backward(cache, y_true)
            model.update(grads, learning_rate)

            epoch_loss += loss
            predicted_class = int(np.argmax(probs))
            if predicted_class == y_true:
                epoch_correct += 1

        train_loss = epoch_loss / len(train_texts)
        train_accuracy = epoch_correct / len(train_texts)
        test_loss, test_accuracy = evaluate(model, test_texts, y_test)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | "
            f"test_loss={test_loss:.4f} | test_acc={test_accuracy:.4f}"
        )

    return model, history
