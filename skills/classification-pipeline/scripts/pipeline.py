#!/usr/bin/env python
"""Composition and search: wiring a preprocessor to an estimator, and trying candidates.

Two mechanics live here, both of which fail silently when they are got wrong.

**Step names.** `build_pipeline` fixes the two step names as `pre` and `clf`. A grid key
like `clf__max_depth` addresses the estimator by name, so a pipeline whose second step is
called `model` breaks every key in the grid - and the error that comes back talks about
parameters not existing, not about the naming. Fixing the names in one place means the keys
in a grid always mean what they say.

**Scoring discipline.** `tune` fits only on training rows and scores only on held-back
rows, in both its modes. There is no path that scores on the rows it fitted on, so a
validation score can never quietly be a training score.

What `tune` deliberately does NOT do: choose a grid, choose a metric, choose whether the
final model is refitted on train only or on train + validation. All four are decisions. The
first two are arguments; the last is left to the caller, which is why `tune` returns
*unfitted* parameters.

If you are about to add a default or a policy to this file, it belongs in SKILL.md instead.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from sklearn.base import clone
from sklearn.pipeline import Pipeline

from evaluation import _score
from validation import fold_cap, make_splitter


def build_pipeline(preprocessor, estimator) -> Pipeline:
    """Wire a preprocessor and an estimator into one Pipeline.

    Preprocessing must be a step *inside* the pipeline, never applied to the frame before
    it. Fitted this way, the imputation medians and scaling means are learned from the
    training rows of each fit and nothing else; applied beforehand they are learned from
    every row that will later be validated against, and the leak raises nothing.
    """
    return Pipeline([("pre", preprocessor), ("clf", estimator)])


def tune(pipeline, grid: dict, X_fit, y_fit, *, X_score=None, y_score=None,
         folds: int | None = None, splitter=None, groups=None, labels=None,
         metric: str, seed: int = 42):
    """Try every combination in `grid`, score each, return (best_score, best_params, scores).

    Exactly one of the three scoring modes, and none of them is a default:

      * `X_score`/`y_score` given - fit on (X_fit, y_fit), score once on the holdout.
        One estimate, so it is noisier.
      * `folds=N` given - stratified N-fold CV over (X_fit, y_fit). The score is the mean
        of N estimates and every row is scored exactly once.
      * `splitter=` given - any object with `.split(X, y, groups)`; the folds come from it,
        which is how the entity-disjoint and time-ordered strategies plug in. Pass
        `groups=` alongside it when the splitter needs to know which entity each row
        belongs to. Build these with `validation.grouped_holdout`, `stratified_group_folds`
        or `temporal_folds`.

    Either way the model is only ever fitted on training rows and only ever scored on
    held-back rows. The returned params are unfitted - fitting the final model is a
    separate decision (train only, train + validation, or the whole pool), so it is left
    to the caller.

    `metric` is required, and has no default on purpose: it decides what the search is
    *for*. That is experimental configuration - fixed, reported, never tuned - and it is
    named by the caller because which metric suits this dataset is a decision, not a
    property of the search. `_score` implements `f1_macro` and `accuracy`; on an
    imbalanced dataset accuracy is dominated by the majority class and will happily call
    a model good when it catches none of the minority, but that is a reason to choose
    macro-F1 deliberately rather than a reason to have it chosen for you.

    `labels` fixes the class set every fold is averaged over; see `evaluation._score` for
    why leaving it None lets a fold that never saw the rarest class score higher than one
    that did.
    """
    if (X_score is None) != (y_score is None):
        raise ValueError("pass both X_score and y_score, or neither")
    if labels is not None:
        # `labels` must cover every class in y_fit. It may name MORE than y_fit holds -
        # a rolling-origin split scores a pool whose training prefix can be missing a
        # class, and there the pool's full label set is the right thing to pass.
        #
        # What must not happen is the reverse: a class present in y_fit but absent from
        # `labels` is dropped from the macro average rather than scored zero, so the
        # result comes back HIGHER. That is the silent inflation this argument exists to
        # prevent, and it is also the signature of passing display names for an encoded y
        # - "no"/"yes" against a y of 0/1 omits both classes.
        present = set(np.unique(np.asarray(y_fit)).tolist())
        omitted = sorted(present - set(labels), key=str)
        if omitted:
            raise ValueError(
                f"labels={list(labels)} omits {omitted}, which do appear in y_fit "
                f"(present: {sorted(present, key=str)})\n"
                f"  a class left out of `labels` is dropped from the macro average rather "
                f"than scored zero, so the score comes back higher than one that includes "
                f"it - and nothing raises.\n"
                f"  if the label was encoded to integers, pass the encoded values here and "
                f"keep the readable names for `class_names` on evaluate and for the "
                f"confusion matrices."
            )
    chosen = [X_score is not None, folds is not None, splitter is not None]
    if sum(chosen) != 1:
        raise ValueError(
            "pass exactly one of (X_score, y_score), folds=N, or splitter=...; got "
            f"{sum(chosen)}"
        )
    if splitter is not None and groups is None and getattr(splitter,
                                                           "enforces_disjoint_groups",
                                                           False):
        # Caught here rather than inside the splitter so the message can name the argument.
        raise ValueError(
            f"{splitter!r} needs groups= - the entity each row belongs to. Without it the "
            f"'group' split is a random split with extra steps."
        )

    keys = list(grid)
    if folds is not None:
        if folds < 2:
            raise ValueError("folds must be >= 2")
        cap = fold_cap(y_fit)
        if folds > cap:
            raise ValueError(
                f"folds={folds} but the rarest class has only {cap} row(s); "
                f"cap it at {cap}"
            )
        splitter = make_splitter(folds, seed)

    best_score, best_params = -1.0, None
    scores: dict[str, float] = {}
    for combo in product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        est = clone(pipeline).set_params(**params)
        if splitter is None:
            est.fit(X_fit, y_fit)
            got = _score(y_score, est.predict(X_score), metric, labels=labels)
        else:
            fold_scores = []
            split_args = (X_fit, y_fit) if groups is None else (X_fit, y_fit, groups)
            for tr_i, va_i in splitter.split(*split_args):
                fold = clone(est).fit(X_fit.iloc[tr_i], y_fit[tr_i])
                fold_scores.append(_score(y_fit[va_i], fold.predict(X_fit.iloc[va_i]),
                                          metric, labels=labels))
            got = float(np.mean(fold_scores))

        scores["|".join(f"{k}={params[k]!r}" for k in keys)] = round(float(got), 6)
        if got > best_score:  # strict: on a tie the earlier combination wins
            best_score, best_params = float(got), params

    return best_score, best_params, scores
