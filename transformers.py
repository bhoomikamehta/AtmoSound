"""
AtmoSound — Inference-Only Transformers
=======================================
Lightweight versions of the preprocessing classes.
Only includes transform() (no fit), since weights are loaded from artifacts.
"""

import numpy as np
import re
from collections import Counter


class TfidfTransformer:
    """TF-IDF transform using pre-fitted vocabulary and IDF weights."""

    def __init__(self, vocabulary, idf, config):
        self.vocabulary_ = vocabulary
        self.idf_ = idf
        self.ngram_range = config["ngram_range"]
        self.sublinear_tf = config["sublinear_tf"]

    def transform(self, documents):
        n_docs = len(documents)
        n_features = len(self.vocabulary_)
        matrix = np.zeros((n_docs, n_features))

        for i, doc in enumerate(documents):
            ngrams = self._get_ngrams(self._tokenize(doc))
            if not ngrams:
                continue
            counts = Counter(ngrams)
            for term, count in counts.items():
                if term in self.vocabulary_:
                    idx = self.vocabulary_[term]
                    tf = count / len(ngrams)
                    if self.sublinear_tf:
                        tf = 1 + np.log(tf) if tf > 0 else 0
                    matrix[i, idx] = tf * self.idf_[idx]

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return matrix / norms

    def _tokenize(self, text):
        if not isinstance(text, str) or not text.strip():
            return []
        text = re.sub(r"[^a-z\s]", " ", text.lower())
        return [t for t in text.split() if len(t) > 1]

    def _get_ngrams(self, tokens):
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append(" ".join(tokens[i:i + n]))
        return ngrams


class SVDTransformer:
    """Project data using pre-fitted SVD components."""

    def __init__(self, components, mean):
        self.components_ = components
        self.mean_ = mean

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T


class MinMaxTransformer:
    """Scale data using pre-fitted min/range."""

    def __init__(self, min_vals, range_vals):
        self.min_ = min_vals
        self.range_ = range_vals

    def transform(self, X):
        return (X - self.min_) / self.range_


class NeuralNetwork:
    """Two-layer NN — inference only."""

    def __init__(self, weights, use_batch_norm=False):
        for attr, val in weights.items():
            setattr(self, attr, val)
        self.use_batch_norm = use_batch_norm

    def predict(self, X):
        z1 = X @ self.W1 + self.b1
        if self.use_batch_norm and hasattr(self, "running_mean1"):
            z1 = self.gamma1 * ((z1 - self.running_mean1) /
                                np.sqrt(self.running_var1 + 1e-8)) + self.beta1
        a1 = np.maximum(0, z1)

        z2 = a1 @ self.W2 + self.b2
        if self.use_batch_norm and hasattr(self, "running_mean2"):
            z2 = self.gamma2 * ((z2 - self.running_mean2) /
                                np.sqrt(self.running_var2 + 1e-8)) + self.beta2
        a2 = np.maximum(0, z2)

        z3 = a2 @ self.W3 + self.b3
        return 1.0 / (1.0 + np.exp(-np.clip(z3, -500, 500)))
