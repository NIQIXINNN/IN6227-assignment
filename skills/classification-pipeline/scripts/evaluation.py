#!/usr/bin/env python
"""Metric definitions - one ruler for every model.

This module exists because metric definitions fail **silently**. `f1_score(average="macro")`
and `f1_score(average="micro")` differ by a few characters and produce different numbers
with no warning; `zero_division` defaults changed between sklearn versions; PR-AUC is
undefined for multiclass and raises rather than returning a sentinel. Two models scored by
two slightly different calls are not comparable, and nothing about the resulting table
says so.

So every metric is defined once, here, and `evaluate` uses these definitions for every
model. A null always travels with the note explaining it: "not defined for multiclass" and
"the metric raised" are different facts and only the note tells them apart.

The metric *choice* - which one the report leads with - is not here. That is a decision,
made in SKILL.md Step 7 from the class balance and the cost of each error type.

If you are about to add a default or a policy to this file, it belongs in SKILL.md instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data import EVIDENCE_SCHEMA
from paths import refuse_bad_output_dir


def _estimator_name(model) -> str:
    """The family an estimator belongs to, for the two-models check below.

    A `Pipeline` hides its estimator behind the `clf` step, so `type(model).__name__` would
    call every pipeline a "Pipeline" and the check would never fire. Fall back to the class
    name when the shape is anything unexpected.
    """
    steps = getattr(model, "named_steps", None)
    if steps and "clf" in steps:
        model = steps["clf"]
    return type(model).__name__


def _score(y_true, y_pred, metric: str, labels=None) -> float:
    """The metric definitions live here so both models are always compared on one ruler.

    `labels` fixes the set of classes the average runs over. Left as None, `f1_score`
    averages over whatever classes appear in `y_true` and `y_pred` **together** - so a fold
    that never validates a rare row does not score that class as zero, it *drops* it, and
    macro-F1 comes out **higher** than a fold that contains the class. On a six-row fixture
    where every row is predicted as the majority class, that is 0.333 with `labels=None`
    against 0.222 with the full label set: dropping the unscored class is worth more than
    the model's actual performance. Pass the pool's label set to turn that silent inflation
    into a visible `zero_division=0` penalty.

    Whether to take that penalty is a decision: it makes folds comparable but punishes a
    fold for a class the splitter could not have given it. `fold_consequences` is what tells
    you which folds those are.
    """
    if metric == "f1_macro":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0,
                              labels=labels))
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    raise ValueError(f"unsupported metric {metric!r}; use 'f1_macro' or 'accuracy'")


def evaluate(model, X, y, name: str = "", fig_dir=None, class_names=None) -> dict:
    """Score a FITTED model on data it has never been trained or selected on.

    Returns the metric set with fixed definitions - macro averaging, zero_division=0, and
    PR-AUC only for the binary case (it is not defined for multiclass, so it is reported
    as null rather than faked). Also writes the confusion matrix figure when fig_dir is
    given, since a report is expected to show one.

    A null `roc_auc` or `pr_auc` always comes with `roc_auc_note` / `pr_auc_note` saying
    why. Never report a null without its note: "undefined for multiclass" and "the metric
    raised" are different facts, and only the note tells them apart.

    Pass `class_names` (aligned to the model's `classes_`) whenever the label was
    label-encoded, or the per-class breakdown comes back keyed by "0"/"1" instead of the
    names the report needs to print.
    """
    y = np.asarray(y)
    pred = model.predict(X)
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", np.unique(y)))
    names = [str(c) for c in (class_names if class_names is not None else classes)]

    auc, auc_kind, auc_why = _safe_auc(y, proba, classes)
    pr_auc, pr_why = _safe_pr_auc(y, proba, classes)

    if fig_dir is not None:
        save_confusion_matrix(y, pred, name or "model", fig_dir, classes, names)

    return {
        "_evidence": {"type": "evaluation", "schema_version": EVIDENCE_SCHEMA,
                      "producer": "evaluate"},
        "model": name,
        "classes": names,
        "n_scored": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "roc_auc": auc,
        "roc_auc_averaging": auc_kind,
        # A null metric always travels with the reason it is null. "Not defined for
        # multiclass" is expected; anything else is a failure and must be read as one,
        # not pasted into the report as a blank cell.
        "roc_auc_note": auc_why,
        "pr_auc": pr_auc,
        "pr_auc_note": pr_why,
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "per_class": _per_class(y, pred, classes, names),
    }


def evaluate_pair(models: dict, X, y, fig_dir=None, class_names=None) -> dict:
    """Score **exactly two** models on one protocol; the dict you write to `metrics.json`.

    The assignment asks for two models compared on one shared protocol, and the report's
    comparison table has two columns. A run that ends with one model, or with two of the
    same family, still produces a table, a figure and a PDF that all *look* complete -
    the missing column reads as a model that happened to lose, not as a run that stopped
    early. So the count is checked here instead of being asked for in prose.

    `models` is `{name: fitted_model}`. Returns `{name: evaluate(...)}`, which is the
    shape `make_pdf.require_evidence` expects of `metrics.json`.
    """
    if len(models) != 2:
        raise ValueError(
            f"evaluate_pair needs exactly 2 models, got {len(models)}: {list(models)}\n"
            f"  the report compares two models side by side; with one, the second column "
            f"is missing and nothing about the table says so.\n"
            f"  choose a second family from reference.md section 2 and fit it in Step 6."
        )
    families = {name: _estimator_name(m) for name, m in models.items()}
    if len(set(families.values())) == 1:
        raise ValueError(
            f"the two models are the same family: {families}\n"
            f"  two of one family differ only in hyperparameters, so the comparison "
            f"cannot say anything about inductive bias.\n"
            f"  pick two families whose assumptions differ - see reference.md section 2."
        )
    return {name: evaluate(model, X, y, name=name, fig_dir=fig_dir, class_names=class_names)
            for name, model in models.items()}


def _per_class(y, pred, classes, names) -> dict:
    """Recall/precision per class, so the minority class cannot hide behind a macro mean."""
    y = np.asarray(y)
    pred = np.asarray(pred)
    out = {}
    for c, label in zip(classes, names):
        actual = int((y == c).sum())
        predicted = int((pred == c).sum())
        hit = int(((pred == c) & (y == c)).sum())
        out[label] = {
            "support": actual,
            "recall": round(hit / actual, 4) if actual else None,
            "precision": round(hit / predicted, 4) if predicted else None,
            "false_negatives": actual - hit,
        }
    return out


def _draw_confusion(ax, y_true, y_pred, title: str, labels) -> None:
    """Draw one confusion matrix onto an existing axis.

    Shared by the single and paired writers so both are the same picture at the same
    scale - two models' matrices that were rendered differently are not comparable at a
    glance, which is the only reason to put them side by side.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("predicted", fontsize=8)
    ax.set_ylabel("actual", fontsize=8)
    ax.set_title(title, fontsize=9)
    return im


def save_confusion_matrix(y_true, y_pred, name: str, fig_dir, classes=None,
                          labels=None) -> Path:
    """Write the confusion matrix for one model.

    `classes` are the values present in the data; `labels` are what to print (pass the
    readable names when the label column was encoded to integers).
    """
    fig_dir = Path(fig_dir)
    refuse_bad_output_dir(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = labels if labels is not None else [
        str(c) for c in (classes if classes is not None else np.unique(y_true))
    ]

    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    im = _draw_confusion(ax, y_true, y_pred, name, labels)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out = fig_dir / f"confusion_{name}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def save_confusion_pair(y_true, predictions: dict, fig_dir, labels=None,
                        name: str = "confusion_pair") -> Path:
    """Write several models' confusion matrices as one side-by-side figure.

    Two separate figures do not fit a two-page report - each is capped at about a fifth of
    a page - and side by side is how a marker compares them anyway. Every panel is drawn
    by `_draw_confusion`, so the only difference between panels is the model.
    """
    fig_dir = Path(fig_dir)
    refuse_bad_output_dir(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    if labels is None:
        labels = [str(c) for c in np.unique(y_true)]
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n + 0.6, 3.0), squeeze=False)
    for ax, (title, pred) in zip(axes[0], predictions.items()):
        im = _draw_confusion(ax, y_true, pred, title, labels)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out = fig_dir / f"{name}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _safe_auc(y_true, proba, classes) -> tuple[float | None, str | None, str | None]:
    """Return (value, what_was_computed, why_it_is_null).

    `why_it_is_null` is the part that matters: a null must never be ambiguous. Returning
    None without a reason is the same silent failure this module exists to prevent, and it
    is how a broken metric gets pasted into a report as a blank cell.
    """
    try:
        if len(classes) == 2:
            return float(roc_auc_score(y_true, proba[:, 1])), "binary", None
        return (float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro")),
                "multiclass-ovr-macro", None)
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def _safe_pr_auc(y_true, proba, classes) -> tuple[float | None, str | None]:
    """Return (value, why_it_is_null).

    `pos_label` must be passed explicitly. Without it sklearn assumes the positive class
    is the integer 1, which raises for a string-labelled target such as "no"/"yes" - and
    that exception used to be swallowed into the same None as "PR-AUC is undefined for
    multiclass", so a binary string-labelled dataset reported a blank PR-AUC with no
    explanation. The two cases are now distinct: multiclass is expected, a failure is not.
    """
    if len(classes) != 2:
        return None, "not defined for multiclass (expected, not a failure)"
    try:
        return (float(average_precision_score(y_true, proba[:, 1], pos_label=classes[1])),
                None)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
