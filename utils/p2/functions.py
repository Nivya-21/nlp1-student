"""Small, reusable operations for ``part2/project2.ipynb``.

The notebook keeps the experimental story sequential.  This module contains
only operations that are reused or easier to test outside a notebook.
"""

from __future__ import annotations

import hashlib
import random
import re
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import nltk
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from part2.clslib import BasicTokenizer, Vocabulary
from part2 import clslib, experiment
from torch.utils.data import DataLoader
from functools import partial


LABELS = {"unharmful": 0, "harmful": 1}
EXCLUDED_SUBCATEGORY = "toxic_language_hate_speech"


class ProjectWordTokenizer(BasicTokenizer):
    """Course-compatible word tokenization without an implicit NLTK download.

    ``BasicTokenizer`` normally calls ``word_tokenize``, which also requires a
    downloaded sentence model.  Project prompts are already individual texts,
    so Treebank tokenization is sufficient and keeps the setup offline.
    """

    def tokens(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()
        if self.remove_punct:
            text = re.sub(r"[^\w\s]", " ", text)
        tokens = nltk.tokenize.TreebankWordTokenizer().tokenize(text)
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        return tokens


def clean_wildguard_prompts(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the Project 1 filtering and cleaning rules to WildGuardMix.

    All copies of a normalized prompt are removed when that prompt has
    contradictory labels.  Consistent duplicate prompts keep one copy.
    """
    required = {"prompt", "prompt_harm_label", "subcategory"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = frame.copy()
    audit = {"original_rows": int(len(data))}

    category = data["subcategory"].fillna("").astype(str).str.strip().str.casefold()
    excluded = category.eq(EXCLUDED_SUBCATEGORY)
    audit["excluded_category_rows"] = int(excluded.sum())
    data = data.loc[~excluded].copy()

    labels = data["prompt_harm_label"].fillna("").astype(str).str.strip().str.casefold()
    missing_label = labels.eq("")
    audit["missing_label_rows"] = int(missing_label.sum())
    unexpected = sorted(set(labels.loc[~missing_label]).difference(LABELS))
    if unexpected:
        raise ValueError(f"Unexpected labels: {unexpected}")
    data = data.loc[~missing_label].copy()
    data["label"] = labels.loc[~missing_label].map(LABELS).astype(int)
    audit["eligible_rows"] = int(len(data))

    data["prompt"] = data["prompt"].fillna("").astype(str).str.normalize("NFC").str.strip()
    empty = data["prompt"].eq("")
    audit["empty_prompt_rows"] = int(empty.sum())
    data = data.loc[~empty].copy()

    contradictory = data.groupby("prompt")["label"].transform("nunique").gt(1)
    audit["contradictory_label_rows"] = int(contradictory.sum())
    audit["contradictory_label_groups"] = int(data.loc[contradictory, "prompt"].nunique())
    data = data.loc[~contradictory].copy()

    audit["duplicate_rows"] = int(data.duplicated("prompt").sum())
    data = data.drop_duplicates("prompt").copy()
    data["id"] = data["prompt"].map(
        lambda text: "wg-" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    )
    data = data[["id", "prompt", "label"]].sort_values("id").reset_index(drop=True)
    audit["retained_rows"] = int(len(data))
    audit["retained_unharmful"] = int(data["label"].eq(0).sum())
    audit["retained_harmful"] = int(data["label"].eq(1).sum())
    audit["retained_harmful_prevalence"] = float(data["label"].mean())
    return data, audit


def balanced_sample(data: pd.DataFrame, examples_per_class: int, seed: int) -> pd.DataFrame:
    """Draw the same number of examples from labels 0 and 1."""
    rng = random.Random(seed)
    selected = []
    for label in (0, 1):
        candidates = data.index[data["label"].eq(label)].tolist()
        if len(candidates) < examples_per_class:
            raise ValueError(f"Label {label} has only {len(candidates)} examples")
        selected.extend(rng.sample(candidates, examples_per_class))
    rng.shuffle(selected)
    return data.loc[selected].reset_index(drop=True)


def stratified_folds(labels, n_splits: int, seed: int):
    """Return paired stratified folds that every system can reuse."""
    labels = np.asarray(labels)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(labels)), labels))


def embedding_coverage(texts, tokenizer, vocabulary) -> tuple[pd.Series, pd.DataFrame]:
    """Measure token/type coverage and list out-of-vocabulary tokens."""
    tokenized = [tokenizer.tokens(text) for text in texts]
    counts = Counter(token for row in tokenized for token in row)
    known = Counter({token: count for token, count in counts.items() if token in vocabulary})
    oov = Counter({token: count for token, count in counts.items() if token not in vocabulary})
    total_tokens = sum(counts.values())
    summary = pd.Series({
        "documents": len(tokenized),
        "tokens": total_tokens,
        "types": len(counts),
        "token_coverage": sum(known.values()) / total_tokens if total_tokens else 0.0,
        "type_coverage": len(known) / len(counts) if counts else 0.0,
    }, name="embedding coverage")
    oov_table = pd.DataFrame(oov.most_common(), columns=["token", "frequency"])
    return summary, oov_table


def coverage_by_label(data: pd.DataFrame, tokenizer, vocabulary) -> pd.DataFrame:
    """Report token coverage separately for unharmful and harmful prompts."""
    rows = []
    for label, group in data.groupby("label", sort=True):
        summary, _ = embedding_coverage(group["prompt"], tokenizer, vocabulary)
        rows.append({"label": int(label), **summary.to_dict()})
    return pd.DataFrame(rows)


@dataclass
class ToyEmbeddings:
    """Course-compatible embedding table for an offline smoke test."""

    vocab: Vocabulary
    vectors: np.ndarray
    num_dims: int
    unk_token: str = "<unk>"
    pad_token: str = "<pad>"

    def __contains__(self, word: str):
        return word.lower() in self.vocab

    def embed_word(self, word: str):
        return self.vectors[self.vocab.index(word.lower())]


def make_toy_embeddings(words, num_dims: int = 12, seed: int = 42) -> ToyEmbeddings:
    """Create deterministic vectors; they have no linguistic interpretation."""
    words = sorted({str(word).lower() for word in words})
    vocab = Vocabulary(["<pad>", "<unk>"], default="<unk>")
    for word in words:
        vocab.add(word)
    vocab.freeze()
    rng = np.random.default_rng(seed)
    vectors = rng.normal(0, 0.1, size=(len(vocab), num_dims)).astype(np.float32)
    vectors[vocab.index("<pad>")] = 0
    return ToyEmbeddings(vocab=vocab, vectors=vectors, num_dims=num_dims)


def classification_scores(gold, predictions) -> dict:
    """Project metrics with harmful as the positive class."""
    tn, fp, fn, tp = confusion_matrix(gold, predictions, labels=[0, 1]).ravel()
    return {
        "macro_f1": float(f1_score(gold, predictions, labels=[0, 1], average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(gold, predictions)),
        "harmful_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "benign_fpr": float(fp / (tn + fp)) if tn + fp else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def trainable_parameters(model: torch.nn.Module) -> int:
    """Count parameters updated by the optimizer."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def nearest_words(embeddings, word, k=5):
    """Cosine neighbours from the real table; missing words are not compared as UNK."""
    if word not in embeddings.vocab:
        return pd.DataFrame(columns=["query", "neighbour", "cosine"])
    query_id = embeddings.vocab.index(word)
    query = embeddings.vectors[query_id]
    scores = np.empty(len(embeddings.vocab), dtype=np.float32)
    for start in range(0, len(scores), 20000):
        block = embeddings.vectors[start:start + 20000]
        denominator = np.linalg.norm(block, axis=1) * np.linalg.norm(query)
        scores[start:start + len(block)] = (block @ query) / np.maximum(denominator, 1e-12)
    scores[[query_id, embeddings.vocab.index(embeddings.pad_token),
            embeddings.vocab.index(embeddings.unk_token)]] = -np.inf
    indices = np.argsort(scores)[-k:][::-1]
    return pd.DataFrame({"query": word, "neighbour": [embeddings.vocab.word(int(i)) for i in indices],
                         "cosine": scores[indices]})


def prepare_fold(training, validation, embeddings, tokenizer):
    """Build the original training-only, GloVe-restricted vocabulary.

    Only training words covered by GloVe receive individual rows. Every other
    token maps to the shared course UNK row; validation never expands the
    vocabulary.
    """
    if set(training["id"]) & set(validation["id"]):
        raise ValueError("Training and validation IDs overlap")
    training_tokens = [tokenizer.tokens(text) for text in training.prompt]
    validation_tokens = [tokenizer.tokens(text) for text in validation.prompt]
    words = sorted({token for row in training_tokens for token in row})
    vocabulary = Vocabulary([embeddings.pad_token, embeddings.unk_token], embeddings.unk_token)
    for word in words:
        if word in embeddings.vocab:
            vocabulary.add(word)
    vocabulary.freeze()
    vectors = np.stack([embeddings.vectors[embeddings.vocab.index(word)] for word in vocabulary])
    pad_index = vocabulary.index(embeddings.pad_token)
    unk_index = vocabulary.index(embeddings.unk_token)
    if pad_index != 0 or unk_index != 1:
        raise ValueError("Expected PAD=0 and UNK=1")
    def encode(rows):
        return [[vocabulary.index(token) for token in row] or [vocabulary.default_index] for row in rows]
    return {"vectors": vectors, "vocab": vocabulary,
            "pad_index": pad_index, "unk_index": unk_index,
            "train_ids": encode(training_tokens), "validation_ids": encode(validation_tokens),
            "train": training, "validation": validation}


def run_configuration(prepared, config, *, seed, epochs, batch_size, learning_rate,
                      weight_decay, device):
    """Fit one configuration for a fixed budget using the course modules.

    Validation is recorded every epoch, but does not pick a checkpoint. This
    returns the last epoch. Use an inner development split before adding early
    stopping to an outer-fold evaluation.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    matrix = torch.tensor(prepared["vectors"], dtype=torch.float32)
    if config.get("initialization", "pretrained") == "random":
        generator = torch.Generator().manual_seed(seed)
        scale = float(matrix[2:].std()) if len(matrix) > 2 else 0.1
        matrix = torch.randn(matrix.shape, generator=generator) * scale
        matrix[prepared["pad_index"]].zero_()
    initial_matrix = matrix.clone()
    # Matrix sampling uses a dedicated generator. Resetting here keeps the
    # classifier initialization matched between pretrained and random systems.
    torch.manual_seed(seed)
    freeze = config.get("frozen", True)
    if config["encoder"] == "lstm":
        encoder = clslib.LSTMEncoder(config.get("hidden_size", 64), matrix,
                                    padding_idx=prepared["pad_index"], freeze_embeddings=freeze)
    else:
        encoder = clslib.EmbedAndPoolEncoder(matrix, average_pool=config["encoder"] == "mean",
                                            padding_idx=prepared["pad_index"], freeze_embeddings=freeze)
    model = clslib.SoftmaxRegressor(config.get("classifier_width", 64),
                                   config.get("classifier_width", 64), 2, encoder,
                                   dropout=config.get("dropout", 0.2)).to(device)
    train_loader = DataLoader(
        experiment.LabelledCorpus(prepared["train_ids"], prepared["train"].label.tolist()),
        batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed),
        collate_fn=partial(experiment.collate_labelled_texts,
                           padding_value=prepared["pad_index"]))
    validation_loader = DataLoader(
        experiment.LabelledCorpus(prepared["validation_ids"], prepared["validation"].label.tolist()),
        batch_size=batch_size, shuffle=False,
        collate_fn=partial(experiment.collate_labelled_texts,
                           padding_value=prepared["pad_index"]))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.NLLLoss()
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        tick = time.perf_counter()
        training = experiment.training_one_epoch(model, train_loader, optimizer, criterion, device)
        validation = experiment.evaluate(model, validation_loader, criterion, device)
        metrics = classification_scores(np.concatenate(validation["labels"]),
                                        np.concatenate(validation["predictions"]))
        history.append({"system": config["name"], "seed": seed, "epoch": epoch,
                        "train_loss": training["loss"], "train_accuracy": training["accuracy"],
                        "validation_loss": validation["loss"], **metrics,
                        "epoch_seconds": time.perf_counter() - tick})
        print(f'{config["name"]} | seed {seed} | epoch {epoch}/{epochs} | '
              f'validation F1 {metrics["macro_f1"]:.4f}', flush=True)
    model.eval()
    probabilities = []
    with torch.no_grad():
        for inputs, _ in validation_loader:
            probabilities.append(model(inputs.to(device)).exp().cpu().numpy())
    probabilities = np.concatenate(probabilities)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    predictions = prepared["validation"][["id", "label"]].rename(columns={"label": "gold"}).copy()
    predictions["system"] = config["name"]
    predictions["seed"] = seed
    predictions["prediction"] = probabilities.argmax(axis=1)
    predictions["probability_harmful"] = probabilities[:, 1]
    predictions["confidence"] = probabilities.max(axis=1)
    final_matrix = model.encoder.embedding.weight.detach().cpu()
    embedding_changed = not torch.equal(initial_matrix, final_matrix)
    result = {**config, "seed": seed, "epochs": epochs, "n_train": len(prepared["train"]),
              "n_validation": len(predictions), "trainable_parameters": trainable_parameters(model),
              "embedding_changed": embedding_changed,
              "device": device, "fit_seconds": time.perf_counter() - started,
              **classification_scores(predictions.gold, predictions.prediction)}
    return result, pd.DataFrame(history), predictions


def input_features(text, tokenizer, vocabulary):
    """Pre-specified descriptive features; they do not enter the classifier."""
    tokens = tokenizer.tokens(text)
    return {"token_length": len(tokens),
            "oov_fraction": sum(token not in vocabulary for token in tokens) / max(1, len(tokens)),
            "has_negation": bool(re.search(r"\b(no|not|never)\b|n't\b", text.lower())),
            "has_security_term": any(token in {"authentication", "bypass", "fraud", "mfa", "phishing", "scam"}
                                     for token in tokens)}
