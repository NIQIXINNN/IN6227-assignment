#!/usr/bin/env python
"""Fit every family that can run, on one protocol, so the pair is chosen from numbers.

**Why this file exists.** Step 4 used to ask for candidates to be eliminated *in prose*,
each rejection naming a number from `model_selection.json`. That rule checks the *form* of
a reason - did it cite a number? - and not its *substance* - does the number support the
conclusion? The gap is what it costs. Across five runs on four datasets the same pair came
back every time, `logistic_regression` + `random_forest`, with rejection reasons that
measurement contradicts. Two of the measured cases:

  * spambase - `svm` rejected as having "the same boundary logistic regression already
    has". An RBF kernel is not linear, and it scored macro-F1 0.9313 against 0.9189.
  * ionosphere - `svm` rejected because the kernel search "adds variance for no clear
    gain". It came first, 0.9434 against 0.9287, at the same fold spread.

A model asked to *argue* about a candidate can always find a number to argue with. So this
module removes the argument: every family that can run is fitted on the same folds and
scored on the same metric, and the shortlist is arithmetic on those numbers.

**What it does NOT do.** It does not choose the pair. It reports a shortlist. Two families
still have to be taken from that shortlist, and they still have to differ in inductive bias
- which is a judgement, and stays in SKILL.md Step 4. A shortlist is not a decision.

**What it does not look at.** The test rows. `screen` forms its split before fitting
anything and screens on the pool only, because ranking families on rows the final model
will be scored against is the same leak the rest of this skill exists to prevent - and it
would raise nothing, it would just make the report's held-out number optimistic.
"""

from __future__ import annotations

import numpy as np

from evaluation import _score
from models import MODEL_REGISTRY, SCALING_REQUIRED, make_model
from pipeline import build_pipeline
from preprocessing import make_preprocessor

# ------------------------------------------------------------------ hard cost bounds
#
# These are the ONLY grounds on which a family may leave the screen without being fitted.
# They are about whether a fit can finish, not about whether it would win - "SVM is
# probably noisy at this size" is exactly the guess this module exists to replace with a
# number, and it is the guess that was wrong twice.
#
# Both bounds name a number, and both are properties of the estimator's cost, not of any
# particular dataset's difficulty.

SVM_MAX_ROWS = 15000        # an RBF kernel costs more than linearly in rows
KNN_MAX_CELLS = 6_000_000   # a neighbour search costs more as columns multiply rows

HARD_COST = {
    "svm": {"rows_over": SVM_MAX_ROWS,
            "why": f"an RBF kernel's fit cost grows faster than linearly in rows and "
                   f"{SVM_MAX_ROWS:,} is where one fit stops fitting in the screen budget"},
    "knn": {"cells_over": KNN_MAX_CELLS,
            "why": f"a neighbour search costs more as rows multiply columns, and "
                   f"{KNN_MAX_CELLS:,} cells is where one fit stops fitting the budget"},
}

# ------------------------------------------------------------------ protocol defaults
#
# A screen has to pick a protocol before any plan exists, so these are the values it uses
# when nothing was passed. They are defaults for a *screen*, not recommendations for the
# run: the plan chooses its own in Step 6, and `run` obeys the plan. Every value used is
# recorded in the evidence so the difference is visible rather than assumed away.

SCREEN_FOLDS = 3
SCREEN_TEST_SIZE = 0.2
SCREEN_SAMPLE = 15000

# Entity detection. A column is an entity key when its rows repeat AND rows sharing a value
# resemble each other AND the column is not one of the model's features. All three numbers
# come from `structure.json`, which Step 2 already wrote; this reads that search rather than
# repeating it.
#
#   * `rows_per_value_mean >= 2` excludes a column that is unique per row.
#   * the lift excludes an ordinary strong predictor, which also makes rows that share a
#     value resemble each other. A bare 2-valued category scores a lift near 2 and is out.
#   * `not is_in_feature_set` is the one that is easy to leave out and expensive to omit.
#     A column the model is *given as a feature* is not an identifier: rows share a value
#     because the value predicts the label, which is a consequence of the column being
#     useful rather than evidence that the rows belong to one entity. An entity key is the
#     opposite - it carries no predictive content of its own, which is exactly why
#     `classify_columns` holds it out of the feature set.
#
# Leaving the feature test out grouped spambase by `capital_run_length_total` (919 distinct
# values, lift 46.6), cutting folds on a predictor instead of on entities. mice_protein's
# `mouse_id` passes all three: 15 rows per value, lift 76.6, and not a feature.
GROUP_MIN_ROWS_PER_VALUE = 2
GROUP_MIN_LIFT = 5.0

BOTH_METRICS = ("f1_macro", "accuracy")


def choose_protocol(structure: dict, *, group_column: str | None = None,
                    time_column: str | None = None) -> dict:
    """The split the screen runs under, and the evidence that chose it.

    `structure.json` already searched for entity and time columns (Step 2 wrote it), so the
    screen reads that search rather than repeating it. An explicit argument wins over the
    inference - the caller may know something the search could not see.
    """
    if time_column:
        return {"unit": "time", "strategy": "temporal", "group_column": None,
                "time_column": time_column,
                "why": f"`{time_column}` was passed in, so rows are ordered by it"}
    if group_column:
        return {"unit": "entity", "strategy": "stratified_group_kfold",
                "group_column": group_column, "time_column": None,
                "why": f"`{group_column}` was passed in, so folds are entity-disjoint"}

    for t in structure.get("time_candidates") or []:
        if t.get("is_non_decreasing"):
            return {"unit": "time", "strategy": "temporal", "group_column": None,
                    "time_column": t["column"],
                    "why": (f"structure.json found `{t['column']}` already in non-decreasing "
                            f"time order, so folds are rolling-origin rather than random")}

    candidates = list(structure.get("group_candidates") or [])
    for g in candidates:
        rows = g.get("rows_per_value_mean") or 0
        lift = g.get("lift_over_chance")
        if (rows >= GROUP_MIN_ROWS_PER_VALUE and lift is not None and lift >= GROUP_MIN_LIFT
                and not g.get("is_in_feature_set")):
            return {"unit": "entity", "strategy": "stratified_group_kfold",
                    "group_column": g["column"], "time_column": None,
                    "why": (f"structure.json reports `{g['column']}` with "
                            f"{rows:g} rows per value, a lift of {lift:.1f} over chance, "
                            f"and no place in the feature set - the shape of an entity key, "
                            f"so folds are entity-disjoint")}

    # Say what was considered and rejected, not just that nothing was found. When a
    # candidate is close, the reader can override with `--group-column` - which is only
    # possible if they can see that a candidate existed at all.
    near = [g for g in candidates
            if (g.get("rows_per_value_mean") or 0) >= GROUP_MIN_ROWS_PER_VALUE
            and (g.get("lift_over_chance") or 0) >= GROUP_MIN_LIFT]
    if near:
        best = max(near, key=lambda g: g.get("lift_over_chance") or 0)
        detail = (f"the closest was `{best['column']}` "
                  f"({best.get('rows_per_value_mean'):g} rows per value, lift "
                  f"{best.get('lift_over_chance'):.1f}), rejected because it is one of the "
                  f"model's features - a column that predicts the label makes rows sharing "
                  f"a value resemble each other without them belonging together")
    else:
        detail = "no column repeats with a lift over chance"
    return {"unit": "row", "strategy": "stratified_kfold", "group_column": None,
            "time_column": None,
            "why": (f"structure.json reports {len(candidates)} group candidate(s) and "
                    f"{detail}, so rows are treated as independent")}


def _excluded(family: str, n_rows: int, one_hot_width: int) -> str | None:
    """The reason this family cannot be fitted here, or None if it can."""
    rule = HARD_COST.get(family)
    if not rule:
        return None
    if "rows_over" in rule and n_rows > rule["rows_over"]:
        return (f"not screened: {n_rows:,} rows is over the {rule['rows_over']:,} row bound "
                f"for this family - {rule['why']}")
    if "cells_over" in rule and n_rows * one_hot_width > rule["cells_over"]:
        return (f"not screened: {n_rows:,} rows x {one_hot_width:,} columns is over the "
                f"{rule['cells_over']:,} cell bound for this family - {rule['why']}")
    return None


def _subsample(groups, y, cap: int, seed: int):
    """Indices for a stratified subsample that keeps every entity whole.

    Whole entities, not whole rows: subsampling rows out of an entity would leave the
    remaining rows of that entity more alike than a real fold ever sees, which flatters
    every family equally and so changes nothing about the ranking - but it would make the
    recorded protocol a lie.
    """
    n = len(y)
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    if groups is None:
        keep = rng.permutation(n)[:cap]
    else:
        uniq = np.unique(groups)
        chosen = set(rng.permutation(uniq)[: max(1, int(len(uniq) * cap / n))].tolist())
        keep = np.array([i for i, g in enumerate(groups) if g in chosen])
    # Keep class coverage: a subsample missing a class scores higher on macro-F1, not
    # lower, so this is the direction that would flatter the screen.
    for c in np.unique(y):
        if not (y[keep] == c).any():
            first = int(np.flatnonzero(y == c)[0])
            keep = np.append(keep, first)
    return np.sort(keep)


def run_screen(X_pool, y_pool, *, numeric, categorical, splitter, groups=None,
               one_hot_width: int | None = None, protocol: dict | None = None,
               folds: int = SCREEN_FOLDS, seed: int = 42, labels=None,
               metrics=BOTH_METRICS, sample: int = SCREEN_SAMPLE, on_step=None) -> dict:
    """Fit every family that can run, on `splitter`, and report the spread.

    Returns evidence, not a verdict. `shortlist` is arithmetic: every family whose mean is
    within one standard deviation of the best mean. That rule is deliberately generous -
    it is there to exclude what is clearly behind, not to crown a winner, because a
    difference smaller than the fold-to-fold spread is not a difference.
    """
    say = on_step or (lambda _: None)
    n_pool_all = len(y_pool)
    width = int(one_hot_width if one_hot_width is not None else len(numeric) + len(categorical))

    keep = _subsample(groups, y_pool, sample, seed)
    # Renumbered, because a splitter hands back *positions* and `X.iloc[position]` has to
    # mean the same row as `y[position]`. Without the reset the two agree only while the
    # pool happens to carry a RangeIndex, which is exactly the kind of agreement that stops
    # holding after someone upstream changes how the frame is built.
    X = X_pool.iloc[keep].reset_index(drop=True)
    y = y_pool[keep]
    g = groups[keep] if groups is not None else None
    n_rows = len(y)

    rows, excluded, errored = [], [], []
    for family in sorted(SCALING_REQUIRED):
        why = _excluded(family, n_rows, width)
        if why:
            excluded.append({"family": family, "reason": why})
            say(f"skipping {family} - {why.split(': ', 1)[1]}")
            continue
        say(f"screening {family}")
        per_fold = {m: [] for m in metrics}
        split_args = (X, y) if g is None else (X, y, g)
        # Building the pipeline is inside the `try` too: a family whose *construction*
        # fails has not been measured either, and letting that raise would take the whole
        # screen down for the same non-reason.
        try:
            pipe = build_pipeline(
                make_preprocessor(numeric, categorical, SCALING_REQUIRED[family]),
                _seeded(family, seed))
            for tr_i, va_i in splitter.split(*split_args):
                fitted = pipe.fit(X.iloc[tr_i], y[tr_i])
                pred = fitted.predict(X.iloc[va_i])
                for m in metrics:
                    per_fold[m].append(_score(y[va_i], pred, m, labels=labels))
        except Exception as exc:                                    # noqa: BLE001
            # A family that cannot be fitted *here* is not a family that lost. It is
            # recorded as exactly what happened, in its own list, with the exception text -
            # never folded into the performance numbers, because "did not run" and "ran and
            # came last" are different facts and only one of them is about the dataset.
            #
            # This exists because it already happened: a threadpoolctl older than 3.0 could
            # not read a newer MKL's version, so sklearn's brute-force neighbour search
            # raised inside `threadpool_limits` and KNN died on any dataset wide enough for
            # `algorithm="auto"` to pick brute. Left uncaught, one environment quirk took
            # the whole screen - six families' worth of work - down with it.
            errored.append({"family": family, "error": f"{type(exc).__name__}: {exc}",
                            "note": "not scored - the fit raised, so this family was "
                                    "neither measured nor excluded"})
            say(f"  {family}: NOT SCORED - {type(exc).__name__}: {exc}")
            continue
        entry = {"family": family,
                 "scaling_used": SCALING_REQUIRED[family],
                 "n_folds": len(per_fold[metrics[0]]),
                 "per_fold": {m: [round(float(v), 6) for v in per_fold[m]] for m in metrics},
                 "mean": {m: round(float(np.mean(per_fold[m])), 6) for m in metrics},
                 "std": {m: round(float(np.std(per_fold[m])), 6) for m in metrics}}
        rows.append(entry)
        say(f"  {family}: " + "  ".join(
            f"{m} {entry['mean'][m]:.4f} (±{entry['std'][m]:.4f})" for m in metrics))

    # The shortlist is computed on every metric, so whichever one the plan picks in Step 7
    # already has a shortlist behind it and no metric has to be chosen here.
    shortlist = {}
    for m in metrics:
        scored = [r for r in rows if r["n_folds"]]
        if not scored:
            shortlist[m] = []
            continue
        best = max(scored, key=lambda r: r["mean"][m])
        floor = best["mean"][m] - best["std"][m]
        shortlist[m] = sorted(
            [r["family"] for r in scored if r["mean"][m] >= floor],
            key=lambda f: -next(r["mean"][m] for r in scored if r["family"] == f))

    # Step 4 has to take two families that differ in inductive bias, and a shortlist is not
    # obliged to contain such a pair - spambase's is random_forest + gradient_boosting, two
    # tree ensembles. Naming the legal pairs here makes that checkable rather than argued,
    # and makes the exception visible: when the list is empty, the pair has to reach below
    # the floor for its second member, and the plan has to say so.
    pairs = {m: [[a, b] for i, a in enumerate(shortlist[m]) for b in shortlist[m][i + 1:]
                 if _bias(a) != _bias(b)] for m in metrics}

    return {
        "protocol": dict(protocol or {}, folds=folds, seed=seed, metrics=list(metrics)),
        "n_pool": int(n_pool_all),
        "n_screened": int(n_rows),
        "subsampled": bool(n_rows < n_pool_all),
        "one_hot_width": width,
        "families": rows,
        "excluded_without_fitting": excluded,
        "errored": errored,
        "shortlist": shortlist,
        "bias": {r["family"]: _bias(r["family"]) for r in rows},
        "pairs_with_different_bias": pairs,
        "shortlist_rule": ("every family whose mean is within one standard deviation of the "
                           "best mean on that metric - wider than it looks, and deliberately "
                           "so: it is here to exclude what is clearly behind, not to crown "
                           "a winner"),
        "how_to_read_this": (
            "These are screening numbers, not the report's numbers. Each family is fitted "
            "with its own defaults - no grid - so a family that is merely badly tuned here "
            "will look worse than it is. That is why the shortlist is one standard "
            "deviation wide and why the pair is taken from it rather than from the top of "
            "it. A family under the floor is behind on its own defaults and at its own "
            "spread; a family in the shortlist is one this dataset cannot rule out."),
        "note": ("evidence, not a verdict - two families still have to be taken from "
                 "`shortlist`, differing in inductive bias, and the reason recorded. "
                 "`pairs_with_different_bias` lists the shortlist combinations that satisfy "
                 "that; an empty list means no pair inside the shortlist does, and the "
                 "second family has to come from below the floor. Choosing the top two "
                 "scores is not the same decision and is not what Step 4 asks for."),
    }


def _bias(family: str) -> str:
    """The coarse name of the boundary this family can reach - `MODEL_REGISTRY`'s own.

    Two families differ in inductive bias when these differ. Read from the registry rather
    than restated here, so a family added later is covered without an edit, and so the
    report's claim about bias and the screen's check on it cannot drift apart.
    """
    return MODEL_REGISTRY[family]["inductive_bias"]


def _seeded(family: str, seed: int):
    """The estimator with the seed applied wherever it has a place to put one.

    Mirrors `driver._seeded`, and the duplication is deliberate: `driver` imports from
    `scripts/`, so `scripts/screen.py` importing `driver` back would be a cycle. Both are
    three lines and both would have to change together if the registry ever gained a
    family that wanted the seed somewhere else.
    """
    model = make_model(family)
    if "random_state" in model.get_params():
        model.set_params(random_state=seed)
    return model
