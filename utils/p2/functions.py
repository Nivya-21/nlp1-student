"""Small, reusable operations for ``part2/project2.ipynb``.

The notebook keeps the experimental story sequential.  This module contains
only operations that are reused or easier to test outside a notebook.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

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
        "unharmful_f1": float(f1_score(gold, predictions, pos_label=0, zero_division=0)),
        "harmful_f1": float(f1_score(gold, predictions, pos_label=1, zero_division=0)),
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


def prepare_fold(training, validation, embeddings, tokenizer, *,
                 vocab_mode="full_train", min_frequency=1):
    """Fit a deterministic model vocabulary from training text only.

    ``full_train`` is the primary inductive policy: every distinct training
    token receives a row. ``glove_restricted`` retains only training words found
    in GloVe. ``frequency_pruned`` retains words meeting ``min_frequency``.
    Validation tokens never expand or reorder the matrix, even if GloVe knows
    them; unseen validation tokens map to UNK.
    """
    if vocab_mode not in {"full_train", "glove_restricted", "frequency_pruned"}:
        raise ValueError(f"Unknown vocabulary mode: {vocab_mode}")
    if not isinstance(min_frequency, int) or min_frequency < 1:
        raise ValueError("min_frequency must be a positive integer")
    if set(training["id"]) & set(validation["id"]):
        raise ValueError("Training and validation IDs overlap")
    training_tokens = [tokenizer.tokens(text) for text in training.prompt]
    validation_tokens = [tokenizer.tokens(text) for text in validation.prompt]
    counts = Counter(token for row in training_tokens for token in row)

    reserved = {embeddings.pad_token, embeddings.unk_token}
    if vocab_mode == "full_train":
        retained_words = sorted(set(counts).difference(reserved))
    elif vocab_mode == "glove_restricted":
        retained_words = sorted(
            word for word in counts if word not in reserved and word in embeddings.vocab)
    else:
        retained_words = sorted(
            word for word, frequency in counts.items()
            if word not in reserved and frequency >= min_frequency)

    vocabulary = Vocabulary(
        [embeddings.pad_token, embeddings.unk_token], embeddings.unk_token)
    for word in retained_words:
        vocabulary.add(word)
    vocabulary.freeze()

    pretrained_vectors = np.zeros((len(vocabulary), embeddings.num_dims), dtype=np.float32)
    pretrained_mask = np.zeros(len(vocabulary), dtype=bool)
    for index, word in vocabulary.items():
        if index >= 2 and word in embeddings.vocab:
            pretrained_vectors[index] = embeddings.vectors[embeddings.vocab.index(word)]
            pretrained_mask[index] = True
    pad_index = vocabulary.index(embeddings.pad_token)
    unk_index = vocabulary.index(embeddings.unk_token)
    if pad_index != 0 or unk_index != 1:
        raise ValueError("Expected PAD=0 and UNK=1")
    if vocab_mode == "glove_restricted":
        pretrained_vectors[unk_index] = embeddings.vectors[
            embeddings.vocab.index(embeddings.unk_token)
        ]
    def encode(rows):
        return [[vocabulary.index(token) for token in row] or [vocabulary.default_index] for row in rows]
    if not hasattr(embeddings, "_random_initialization_moments"):
        observed_vectors = embeddings.vectors[2:]
        embeddings._random_initialization_moments = (
            observed_vectors.mean(axis=0).astype(np.float32),
            np.maximum(observed_vectors.std(axis=0), 1e-6).astype(np.float32),
        )
    random_mean, random_std = embeddings._random_initialization_moments
    return {"pretrained_vectors": pretrained_vectors,
            "embedding_dims": embeddings.num_dims,
            "vocab": vocabulary,
            "tokenizer_policy": {
                "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
                "lowercase": bool(tokenizer.lowercase),
                "remove_punct": bool(tokenizer.remove_punct),
                "remove_stopwords": bool(tokenizer.remove_stopwords),
                "language": tokenizer.language,
            },
            "pretrained_mask": pretrained_mask, "vocab_mode": vocab_mode,
            "min_frequency": min_frequency,
            "training_token_set": frozenset(counts),
            "training_types": len(set(counts).difference(reserved)),
            "dedicated_glove_rows": int(pretrained_mask.sum()),
            "dedicated_random_rows": int(len(vocabulary) - 2 - pretrained_mask.sum()),
            "pruned_training_types": int(
                len(set(counts).difference(reserved)) - len(retained_words)),
            "random_mean": random_mean, "random_std": random_std,
            "pad_index": pad_index, "unk_index": unk_index,
            "train_ids": encode(training_tokens), "validation_ids": encode(validation_tokens),
            "train": training, "validation": validation}


def initial_embedding_matrix(prepared, initialization, seed):
    """Create a scale-matched random base and optionally overlay GloVe rows."""
    if initialization not in {"pretrained", "random"}:
        raise ValueError(f"Unknown initialization: {initialization}")
    shape = (len(prepared["vocab"]), prepared["embedding_dims"])
    generator = torch.Generator().manual_seed(seed)
    mean = torch.tensor(prepared["random_mean"], dtype=torch.float32)
    std = torch.tensor(prepared["random_std"], dtype=torch.float32)
    random_matrix = torch.randn(shape, generator=generator) * std + mean
    random_matrix[prepared["pad_index"]].zero_()
    if initialization == "random":
        return random_matrix
    matrix = random_matrix
    mask = torch.from_numpy(prepared["pretrained_mask"])
    matrix[mask] = torch.from_numpy(prepared["pretrained_vectors"][prepared["pretrained_mask"]])
    if prepared["vocab_mode"] == "glove_restricted":
        matrix[prepared["unk_index"]] = torch.from_numpy(
            prepared["pretrained_vectors"][prepared["unk_index"]]
        )
    matrix[prepared["pad_index"]].zero_()
    return matrix


def run_configuration(prepared, config, *, seed, epochs, batch_size, learning_rate,
                      weight_decay, device):
    """Fit one configuration for a fixed budget using the course modules.

    Validation is recorded every epoch, but does not pick a checkpoint. This
    returns the last epoch. Use an inner development split before adding early
    stopping to an outer-fold evaluation.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    initialization = config.get("initialization", "pretrained")
    matrix = initial_embedding_matrix(prepared, initialization, seed)
    # Matrix construction uses a dedicated generator. Resetting here guarantees
    # matched downstream initialization for pretrained versus random systems.
    torch.manual_seed(seed)
    freeze = config.get("frozen", True)
    if config["encoder"] == "lstm":
        encoder = clslib.LSTMEncoder(config.get("hidden_size", 64), matrix,
                                    padding_idx=prepared["pad_index"], freeze_embeddings=freeze,
                                    word_dropout=config.get("word_dropout", 0.0))
    else:
        encoder = clslib.EmbedAndPoolEncoder(matrix, average_pool=config["encoder"] == "mean",
                                            padding_idx=prepared["pad_index"], freeze_embeddings=freeze)
    model = clslib.SoftmaxRegressor(config.get("classifier_width", 64),
                                   config.get("classifier_width", 64), 2, encoder,
                                   dropout=config.get("dropout", 0.2)).to(device)
    train_loader = DataLoader(
        experiment.LabelledCorpus(prepared["train_ids"], prepared["train"].label.tolist()),
        batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed),
        collate_fn=partial(experiment.collate_labelled_texts, padding_value=prepared["pad_index"]))
    validation_loader = DataLoader(
        experiment.LabelledCorpus(prepared["validation_ids"], prepared["validation"].label.tolist()),
        batch_size=batch_size, shuffle=False,
        collate_fn=partial(experiment.collate_labelled_texts, padding_value=prepared["pad_index"]))
    embedding_parameter = model.encoder.embedding.weight
    embedding_learning_rate = config.get("embedding_learning_rate", learning_rate)
    other_parameters = [parameter for parameter in model.parameters()
                        if parameter.requires_grad and parameter is not embedding_parameter]
    parameter_groups = [{"params": other_parameters, "lr": learning_rate,
                         "weight_decay": weight_decay}]
    if embedding_parameter.requires_grad:
        parameter_groups.append({"params": [embedding_parameter],
                                 "lr": embedding_learning_rate,
                                 "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(parameter_groups)
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
    pad_row = model.encoder.embedding.weight[prepared["pad_index"]].detach().cpu()
    if not torch.equal(pad_row, torch.zeros_like(pad_row)):
        raise AssertionError("PAD embedding changed during optimization")
    predictions = prepared["validation"][["id", "label"]].rename(columns={"label": "gold"}).copy()
    predictions["system"] = config["name"]
    predictions["seed"] = seed
    predictions["prediction"] = probabilities.argmax(axis=1)
    predictions["probability_harmful"] = probabilities[:, 1]
    predictions["confidence"] = probabilities.max(axis=1)
    result = {**config, "seed": seed, "epochs": epochs, "n_train": len(prepared["train"]),
              "n_validation": len(predictions), "trainable_parameters": trainable_parameters(model),
              "vocab_mode": prepared["vocab_mode"], "vocabulary_size": len(prepared["vocab"]),
              "glove_rows": prepared["dedicated_glove_rows"],
              "random_rows": prepared["dedicated_random_rows"],
              "pruned_training_types": prepared["pruned_training_types"],
              "min_frequency": prepared["min_frequency"],
              "embedding_learning_rate": embedding_learning_rate,
              "device": device, "fit_seconds": time.perf_counter() - started,
              **classification_scores(predictions.gold, predictions.prediction)}
    return result, pd.DataFrame(history), predictions


def input_features(text, tokenizer, training_token_set, model_vocabulary, glove_vocabulary):
    """Pre-specified descriptive features; they do not enter the classifier."""
    tokens = tokenizer.tokens(text)
    return {"token_length": len(tokens),
            "glove_oov_fraction": sum(token not in glove_vocabulary for token in tokens) / max(1, len(tokens)),
            "train_vocab_oov_fraction": sum(token not in training_token_set for token in tokens) / max(1, len(tokens)),
            "model_unk_fraction": sum(token not in model_vocabulary for token in tokens) / max(1, len(tokens)),
            "has_negation": bool(re.search(r"\b(no|not|never)\b|n't\b", text.lower())),
            "has_security_term": any(token in {"authentication", "bypass", "fraud", "mfa", "phishing", "scam"}
                                     for token in tokens)}


def experiment_signature(config, training_ids, validation_ids, *, seed, epochs):
    """Hash the resolved fit inputs so interrupted runs can resume safely."""
    payload = {
        "config": config,
        "training_ids": sorted(map(str, training_ids)),
        "validation_ids": sorted(map(str, validation_ids)),
        "seed": int(seed),
        "epochs": int(epochs),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_identity(path):
    """Return a stable content identity for an experimental input or implementation file."""
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def save_fit_artifacts(folder, signature, score, history, predictions):
    """Atomically save one completed fit; the completion marker is written last."""
    fit_folder = Path(folder) / "fits" / signature
    fit_folder.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "score.json": json.dumps(score, indent=2, default=str),
        "history.csv": history.to_csv(index=False),
        "predictions.csv": predictions.to_csv(index=False),
    }
    for name, content in artifacts.items():
        temporary = fit_folder / f"{name}.tmp"
        temporary.write_text(content)
        temporary.replace(fit_folder / name)
    marker = fit_folder / "complete.json.tmp"
    marker.write_text(json.dumps({"signature": signature, "status": "COMPLETE"}, indent=2))
    marker.replace(fit_folder / "complete.json")
    return fit_folder


def load_fit_artifacts(folder, signature):
    """Load a valid completed fit, or return ``None`` when it must be rerun."""
    fit_folder = Path(folder) / "fits" / signature
    marker = fit_folder / "complete.json"
    required = [fit_folder / "score.json", fit_folder / "history.csv",
                fit_folder / "predictions.csv"]
    if not marker.exists() or not all(path.exists() for path in required):
        return None
    status = json.loads(marker.read_text())
    if status != {"signature": signature, "status": "COMPLETE"}:
        return None
    return (json.loads(required[0].read_text()),
            pd.read_csv(required[1]), pd.read_csv(required[2]))
