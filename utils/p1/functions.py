"""Small helpers used by part1/project1.ipynb.

The notebook contains the data preparation, model choices and experiment loop.
This file only collects operations that it needs more than once.
"""

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader

from part1.clslib import Encoder, TokenIndicator
from part1.smlp import (
    SparseMLP, SparseDataset, sparse_collate,
    train_one_epoch, scipy_to_torch_sparse,
)


class TokenIndicators(TokenIndicator):
    """Reuse the tutorial encoder, fixing its reference to a global tokenizer."""

    def __len__(self):
        return self.tokenizer.get_vocab_size()


class BagOfWords(Encoder):
    """Complete the tutorial exercise: count each token instead of its presence."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def __call__(self, text):
        token_ids = self.tokenizer.encode(text).ids
        return np.bincount(token_ids, minlength=len(self)).astype(np.float32)


def encode_texts(texts, encoder):
    """Turn the tutorial's document vectors into one sparse feature matrix."""
    return csr_matrix([encoder(text) for text in texts], dtype=np.float32)


class TFIDF(BagOfWords):
    """Weight token counts using document frequencies learned from training text."""

    def fit(self, training_texts):
        counts = encode_texts(training_texts, BagOfWords(self.tokenizer))
        self.transformer = TfidfTransformer(norm="l2", smooth_idf=True)
        self.transformer.fit(counts)
        return self

    def __call__(self, text):
        counts = csr_matrix(super().__call__(text).reshape(1, -1))
        return self.transformer.transform(counts).toarray()[0]


class TwoHiddenMLP(SparseMLP):
    """Keep the course sparse first layer and add a second ReLU hidden layer."""

    def __init__(self, input_dim, hidden_dim=64, second_hidden_dim=32, dropout=0.2):
        super().__init__(input_dim, hidden_dim, num_classes=2, dropout=dropout)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, second_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden_dim, 2),
            nn.LogSoftmax(dim=-1),
        )


class LearningMLP(SparseMLP):
    """Tutorial MLP with editable hidden widths and activation for the learning notebook.

    (64,) means one hidden layer; (64, 32) means two. The course sparse
    first layer, forward method and probability convention stay the same.
    """

    def __init__(self, input_dim, hidden_layers=(64,), activation="relu", dropout=0.2):
        activations = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
        if not hidden_layers or any(not isinstance(w, int) or w < 1 for w in hidden_layers):
            raise ValueError("Use positive hidden widths, for example (64,) or (64, 32).")
        if activation not in activations:
            raise ValueError("activation must be relu, tanh or gelu")
        super().__init__(input_dim, hidden_layers[0], num_classes=2, dropout=dropout)
        self.activation = activations[activation]()
        if len(hidden_layers) == 1:
            return  # Preserve the tutorial's original output layer and initialization.
        layers = []
        for previous, width in zip(hidden_layers, hidden_layers[1:]):
            layers.extend([nn.Linear(previous, width), activations[activation](), nn.Dropout(dropout)])
        layers.extend([nn.Linear(hidden_layers[-1], 2), nn.LogSoftmax(dim=-1)])
        self.output = nn.Sequential(*layers)


def train_mlp(model, X_train, y_train, epochs=10, seed=42,
              batch_size=64, learning_rate=0.001, weight_decay=0.0001):
    """Train for a fixed number of epochs with the tutorial's training loop.

    The model is updated in place. Return training history for inspection.
    Validation data is deliberately absent: it cannot choose the best epoch.
    """
    loader = DataLoader(
        SparseDataset(X_train, y_train), batch_size=batch_size, shuffle=True,
        collate_fn=sparse_collate,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    loss = nn.NLLLoss()  # The course model returns log probabilities.
    history = []
    for epoch in range(epochs):
        history.append(train_one_epoch(model, loader, optimizer, loss, device="cpu"))
    return history


def predict_mlp(model, X, batch_size=64):
    """Return probabilities in column order [unharmful, harmful]."""
    model.eval()  # Disable dropout for prediction.
    batches = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            batch = scipy_to_torch_sparse(X[start:start + batch_size])
            batches.append(model(batch).exp().numpy())
    return np.concatenate(batches)


def classification_scores(gold, predictions):
    """Macro-F1 plus the two class-specific error rates, with harmful = 1."""
    tn, fp, fn, tp = confusion_matrix(gold, predictions, labels=[0, 1]).ravel()
    return {
        "macro_f1": float(f1_score(gold, predictions, labels=[0, 1], average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(gold, predictions)),
        "harmful_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "benign_fpr": float(fp / (tn + fp)) if tn + fp else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
