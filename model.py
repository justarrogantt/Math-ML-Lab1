import numpy as np


class EmbeddingNewsClassifier:
    def __init__(self, vocab_size, embedding_dim=50, num_classes=4, seed=42):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        rng = np.random.default_rng(seed)
        self.E = rng.normal(0.0, 0.1, size=(vocab_size, embedding_dim))
        self.W = rng.normal(0.0, 0.1, size=(num_classes, embedding_dim))
        self.b = np.zeros(num_classes, dtype=np.float64)

    @staticmethod
    def softmax(logits):
        shifted_logits = logits - np.max(logits)
        exp_logits = np.exp(shifted_logits)
        return exp_logits / np.sum(exp_logits)

    @staticmethod
    def cross_entropy(probs, y_true):
        eps = 1e-8
        return -np.log(probs[y_true] + eps)

    def forward(self, word_indices, y_true=None):
        embeddings = self.E[word_indices]
        text_vector = np.mean(embeddings, axis=0)
        logits = self.W @ text_vector + self.b
        probs = self.softmax(logits)

        cache = {
            "word_indices": word_indices,
            "embeddings": embeddings,
            "x": text_vector,
            "probs": probs,
        }

        if y_true is None:
            return probs, cache

        loss = self.cross_entropy(probs, y_true)
        return probs, loss, cache

    def backward(self, cache, y_true):
        word_indices = cache["word_indices"]
        x = cache["x"]
        probs = cache["probs"]
        text_length = len(word_indices)

        one_hot = np.zeros(self.num_classes, dtype=np.float64)
        one_hot[y_true] = 1.0

        delta = probs - one_hot
        dW = np.outer(delta, x)
        db = delta
        dx = self.W.T @ delta

        embedding_grads = {}
        grad_per_token = dx / text_length

        for word_idx in word_indices:
            if word_idx not in embedding_grads:
                embedding_grads[word_idx] = np.zeros_like(grad_per_token)
            embedding_grads[word_idx] += grad_per_token

        return {
            "dW": dW,
            "db": db,
            "dE": embedding_grads,
            "dx": dx,
            "delta": delta,
        }

    def update(self, grads, learning_rate):
        self.W -= learning_rate * grads["dW"]
        self.b -= learning_rate * grads["db"]
        for word_idx, grad in grads["dE"].items():
            self.E[word_idx] -= learning_rate * grad

    def predict(self, word_indices):
        probs, _ = self.forward(word_indices)
        predicted_class = int(np.argmax(probs))
        return predicted_class, probs
