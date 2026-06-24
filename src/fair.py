"""Utilidades reutilizables para aprendizaje justo (FAIR Loss).

Extrae a módulo compartido la lógica que en `02_fair_loss.ipynb` estaba
definida inline para poder reutilizarla en otros notebooks, en especial en la
parte de incertidumbre.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

import keras
from keras import ops


EPS = 1e-8


def stack_target_sensitive(y, s):
    """Construye etiquetas `(N, 2) = [target, sensible]` en float32."""
    return np.column_stack([np.asarray(y), np.asarray(s)]).astype("float32")


def pearson_abs(p, s, eps=EPS):
    """|corr_Pearson(p, s)| diferenciable, implementada solo con `keras.ops`."""
    p = ops.cast(ops.reshape(p, (-1,)), "float32")
    s = ops.cast(ops.reshape(s, (-1,)), "float32")
    pm = p - ops.mean(p)
    sm = s - ops.mean(s)
    num = ops.sum(pm * sm)
    den = ops.sqrt(ops.sum(pm ** 2) * ops.sum(sm ** 2)) + eps
    return ops.abs(num / den)


def make_fair_loss(lambda_fair, base_error="bce", w_neg=1.0, w_pos=1.0, eps=EPS):
    """Factory de la FAIR loss: BCE/MSE ponderado + lambda * |corr(yhat, s)|."""
    lambda_fair = float(lambda_fair)
    if base_error not in ("bce", "mse"):
        raise ValueError("base_error debe ser 'bce' o 'mse'.")

    def fair_loss(y_true, y_pred):
        y = y_true[:, 0:1]
        s = y_true[:, 1:2]
        y_pred_clipped = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)

        if base_error == "bce":
            per_sample = keras.losses.binary_crossentropy(y, y_pred_clipped)
        else:
            per_sample = ops.mean(ops.square(y - y_pred_clipped), axis=-1)

        w = y[:, 0] * w_pos + (1.0 - y[:, 0]) * w_neg
        base = ops.sum(w * per_sample) / (ops.sum(w) + eps)
        corr = pearson_abs(y_pred, s, eps)
        return base + lambda_fair * corr

    fair_loss.__name__ = f"fair_loss_lambda_{lambda_fair:g}"
    return fair_loss


class FairCorr(keras.metrics.Metric):
    """|corr(yhat, s)| exacto por época mediante acumuladores."""

    def __init__(self, name="fair_corr", eps=EPS, **kwargs):
        super().__init__(name=name, **kwargs)
        self.eps = eps
        self.n = self.add_weight(name="n", initializer="zeros")
        self.sp = self.add_weight(name="sp", initializer="zeros")
        self.ss = self.add_weight(name="ss", initializer="zeros")
        self.spp = self.add_weight(name="spp", initializer="zeros")
        self.sss = self.add_weight(name="sss", initializer="zeros")
        self.sps = self.add_weight(name="sps", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        del sample_weight
        p = ops.reshape(ops.cast(y_pred, "float32"), (-1,))
        s = ops.reshape(ops.cast(y_true[:, 1:2], "float32"), (-1,))
        self.n.assign_add(ops.cast(ops.size(p), "float32"))
        self.sp.assign_add(ops.sum(p))
        self.ss.assign_add(ops.sum(s))
        self.spp.assign_add(ops.sum(p * p))
        self.sss.assign_add(ops.sum(s * s))
        self.sps.assign_add(ops.sum(p * s))

    def result(self):
        n = self.n
        cov = n * self.sps - self.sp * self.ss
        vp = n * self.spp - self.sp ** 2
        vs = n * self.sss - self.ss ** 2
        return ops.abs(cov / (ops.sqrt(vp * vs) + self.eps))

    def reset_state(self):
        for variable in self.variables:
            variable.assign(0.0)


class TargetAUC(keras.metrics.AUC):
    """AUC calculada sobre la columna target de `y_true` con forma `(N, 2)`."""

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true[:, 0:1], y_pred, sample_weight)


def demographic_parity_diff(y_prob, s, thr=0.5):
    """Diferencia absoluta de tasas positivas entre grupos sensibles."""
    y_hat = (np.asarray(y_prob).ravel() >= thr).astype(int)
    s = np.asarray(s).ravel()
    p_s0 = y_hat[s == 0].mean()
    p_s1 = y_hat[s == 1].mean()
    return abs(p_s0 - p_s1)


def evaluate_binary_predictions(y_true, y_prob, s=None, thr=0.5):
    """Evalúa un clasificador binario y, opcionalmente, sus métricas FAIR."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, (y_prob >= thr).astype(int)),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }
    if s is not None:
        s = np.asarray(s).ravel()
        metrics["fair_corr"] = abs(np.corrcoef(y_prob, s)[0, 1])
        metrics["dpd"] = demographic_parity_diff(y_prob, s, thr)
    return metrics
