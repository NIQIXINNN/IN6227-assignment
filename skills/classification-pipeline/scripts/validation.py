#!/usr/bin/env python
"""Splitting, and the arithmetic that says what a split would contain.

Splitting is where the most expensive silent failures live, because a bad split produces a
*plausible* score rather than an error:

  * an unstratified split quietly starves the rarest class, and on an imbalanced dataset
    that is exactly the class the report is about;
  * a fold holding zero rows of a class scores it as absent, and macro-F1 averaged over
    the visible classes goes **up**, not down - the missing class stops being counted
    rather than counting as zero.

So `stratified_split` has no option to turn stratification off, and `validation_report`
returns arithmetic rather than a recommendation. Which strategy to use is a decision made
per dataset in SKILL.md Step 6, from these numbers; a threshold baked in here would be a
decision that fits one CSV and is wrong for the next.

`fold_cap` is the one enforced limit, because it is arithmetic rather than judgement: a
fold cannot hold fewer than one row of a class.

If you are about to add a default or a policy to this file, it belongs in SKILL.md instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)


def class_counts(y) -> np.ndarray:
    """Rows per class, indexed by the encoded label. Plain arithmetic.

    The negative check is here because of what `np.bincount` says without it: `'list'
    argument must have no negative elements`, which names neither the column, the file, nor
    the thing that actually went wrong. A negative code means something upstream encoded a
    missing label - `driver._encode` refuses to do that, so reaching this line with one
    means a new producer of `y` bypassed it, and the sentence here is what says so.
    """
    y = np.asarray(y)
    if y.size and (y < 0).any():
        raise ValueError(
            f"{int((y < 0).sum())} label code(s) are negative, so they belong to no class. "
            f"A negative code is what a missing label becomes when it is cast to an integer "
            f"instead of being dropped first."
        )
    return np.bincount(y)


def stratified_split(X, y, test_size: float, seed: int):
    """Split so every class appears in both halves in proportion. Always stratified.

    There is deliberately no option to turn stratification off. Unstratified splitting
    does not raise - it just quietly under-represents the rarest class, and on an
    imbalanced dataset that is exactly the class the report is about.
    """
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def fold_cap(y_pool) -> int:
    """The most folds this pool can support: a fold needs >= 1 row of every class."""
    return int(class_counts(y_pool).min())


def make_splitter(folds: int, seed: int) -> StratifiedKFold:
    """The k-fold splitter `tune` uses, with shuffle fixed on.

    `shuffle=False` is not offered: on a frame ordered by class it hands each fold a
    different class mix, and sklearn warns about the *row* counts, never about the class
    imbalance that actually breaks the score.
    """
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def _validate_groups(X, groups):
    if groups is None:
        raise ValueError(
            "this splitter needs groups - the column (or array) saying which entity each "
            "row belongs to. Pass groups= to tune(), or use the plain stratified splitter "
            "if the rows really are independent (SKILL.md Step 3)."
        )
    groups = np.asarray(groups)
    if len(groups) != len(X):
        raise ValueError(
            f"groups has {len(groups)} entries but X has {len(X)} rows. A group array that "
            f"is misaligned with X does not raise anywhere downstream - it just splits on "
            f"the wrong entities and reports a perfectly plausible score."
        )
    return groups


class _GroupedHoldout:
    """One holdout in which whole entities go to one side or the other.

    Yields **positional** indices, like every splitter here, so `X.iloc[idx]` is always
    correct. Indexing by label (`groups[X_train.index]`) silently misaligns the moment
    anything upstream has dropped or reordered a row, and the result is still a number.
    """

    enforces_disjoint_groups = True

    def __init__(self, test_size: float, seed: int):
        self.test_size = test_size
        self.seed = seed

    def split(self, X, y=None, groups=None):
        groups = _validate_groups(X, groups)
        return GroupShuffleSplit(n_splits=1, test_size=self.test_size,
                                 random_state=self.seed).split(X, y, groups)

    def __repr__(self):
        return f"grouped_holdout(test_size={self.test_size}, seed={self.seed})"


class _StratifiedGroupFolds:
    """k folds that keep every entity whole and every class's proportion.

    There is deliberately no `shuffle=False` path. On a frame ordered by class or by
    entity, an unshuffled split hands each fold a different class mix - and scikit-learn
    warns about the *row* counts, never about the imbalance that actually breaks the score.
    """

    enforces_disjoint_groups = True

    def __init__(self, n_splits: int, seed: int):
        self.n_splits = n_splits
        self.seed = seed

    def split(self, X, y=None, groups=None):
        groups = _validate_groups(X, groups)
        if y is None:
            raise ValueError("stratified group folds need y as well as groups")
        n_groups = int(len(np.unique(groups)))
        if self.n_splits > n_groups:
            raise ValueError(
                f"n_splits={self.n_splits} but there are only {n_groups} distinct entities; "
                f"a fold cannot be built without reusing an entity"
            )
        return StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True,
                                    random_state=self.seed).split(X, y, groups)

    def __repr__(self):
        return f"stratified_group_folds(n_splits={self.n_splits}, seed={self.seed})"


class _TemporalFolds:
    """Rolling-origin folds over rows in time order, with a mandatory gap.

    The failure this guards against: `TimeSeriesSplit` splits by **row position**. Given a
    frame that is not in time order it trains on rows that come later in time than the rows
    it validates on, and reports an excellent score. Nothing errors, because from the
    splitter's point of view it did exactly what it was asked.

    So `order` is required and checked, and `gap` has no default - the number of rows at
    the boundary to drop is a property of the data (how far ahead the label looks), not
    something this module can pick.
    """

    enforces_disjoint_groups = False

    def __init__(self, n_splits: int, gap: int, order):
        if gap is None:
            raise ValueError(
                "gap is required. It is the number of rows at each fold boundary to leave "
                "out of training, so that a label which looks ahead cannot be learned from "
                "its own future. There is no safe default - it depends on the data."
            )
        self.n_splits = n_splits
        self.gap = int(gap)
        self.order = order

    def _resolved_order(self, X):
        if isinstance(self.order, str):
            if self.order not in getattr(X, "columns", []):
                raise ValueError(
                    f"order={self.order!r} is not a column of X; pass the column name or "
                    f"the array itself"
                )
            raw = X[self.order]
        else:
            raw = self.order
        raw = pd.Series(np.asarray(raw)).reset_index(drop=True)
        if len(raw) != len(X):
            raise ValueError(f"order has {len(raw)} entries but X has {len(X)} rows")
        return raw

    def split(self, X, y=None, groups=None):
        raw = self._resolved_order(X)
        steps = raw.diff().dropna()
        if len(steps) and not bool((steps >= 0).all()):
            back = int((steps < 0).sum())
            raise ValueError(
                f"order={self.order!r} is not non-decreasing: {back} of {len(steps)} steps "
                f"go backwards. TimeSeriesSplit splits by row position, so on an unsorted "
                f"frame it trains on the future and validates on the past - and reports a "
                f"good score. Sort the frame by this column first."
            )
        n = len(X)
        if self.n_splits + 1 + self.gap > n:
            raise ValueError(
                f"n_splits={self.n_splits} with gap={self.gap} needs more than {n} rows"
            )
        return TimeSeriesSplit(n_splits=self.n_splits, gap=self.gap).split(X)

    def __repr__(self):
        return f"temporal_folds(n_splits={self.n_splits}, gap={self.gap}, order={self.order!r})"


def grouped_holdout(test_size: float, seed: int) -> _GroupedHoldout:
    """Hold out whole entities, not rows. Use when several rows share an entity."""
    return _GroupedHoldout(test_size, seed)


def stratified_group_folds(n_splits: int, seed: int) -> _StratifiedGroupFolds:
    """k folds, entity-disjoint and class-proportional. Use over k-fold whenever rows repeat."""
    return _StratifiedGroupFolds(n_splits, seed)


def temporal_folds(n_splits: int, gap: int, order) -> _TemporalFolds:
    """Folds that always train on the past and validate on the future.

    `gap` is required and has no default: it is the number of boundary rows to drop, which
    depends on how far ahead the label looks. `order` is the column (or array) holding the
    time - passing row positions is only correct if the frame is already sorted by it, and
    the check will tell you if it is not.
    """
    return _TemporalFolds(n_splits, gap, order)


def fold_consequences(splitter, X, y, groups=None, classes=None) -> dict:
    """Realize the folds and report what each one actually contains. Fits nothing.

    This is where a split that looks fine on paper stops looking fine. The two things it
    reports are the two that inflate a score without raising:

      * **a class absent from a fold.** `evaluate` averages over the classes it can see, so
        a fold that never validates a rare row scores that class as absent - and macro-F1
        goes *up*, not down. Nothing warns.
      * **entities on both sides.** The model recognises the entity instead of learning the
        task.

    `groups_shared_between_train_and_val` is expected to be 0 for a group splitter and is
    reported for any splitter, including the plain one - seeing a non-zero number on a
    random split is the evidence that the rows are not independent.
    """
    y = np.asarray(y)
    classes = list(classes) if classes is not None else sorted(set(y.tolist()))
    groups_arr = np.asarray(groups) if groups is not None else None
    n = len(X)

    folds, absent_from_val, absent_from_train, shared = [], [], [], []
    for i, (tr, va) in enumerate(splitter.split(X, y, groups_arr)):
        tr, va = np.asarray(tr), np.asarray(va)
        tr_classes, va_classes = set(y[tr].tolist()), set(y[va].tolist())
        miss_va = [str(c) for c in classes if c not in va_classes]
        miss_tr = [str(c) for c in classes if c not in tr_classes]
        n_shared = None
        if groups_arr is not None:
            n_shared = int(len(set(groups_arr[tr].tolist()) & set(groups_arr[va].tolist())))
            shared.append(n_shared)
        absent_from_val.append(miss_va)
        absent_from_train.append(miss_tr)
        folds.append({
            "fold": i,
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "train_first_row": int(tr.min()) if len(tr) else None,
            "val_first_row": int(va.min()) if len(va) else None,
            "classes_absent_from_train": miss_tr,
            "classes_absent_from_val": miss_va,
            "groups_shared_between_train_and_val": n_shared,
            "rarest_class_rows_in_val": int(min((y[va] == c).sum() for c in classes)),
        })

    out = {
        "splitter": repr(splitter),
        "n_rows": n,
        "n_folds": len(folds),
        "n_classes": len(classes),
        "classes": [str(c) for c in classes],
        "folds": folds,
        "any_class_absent_from_val": any(absent_from_val),
        "any_class_absent_from_train": any(absent_from_train),
        "max_groups_shared_between_train_and_val": max(shared) if shared else None,
        "note": ("a class absent from a fold is not scored as zero - macro averaging runs "
                 "over the classes it can see, so the fold scores HIGHER than one that "
                 "contains it. Read any_class_absent_from_val as 'this fold's score is not "
                 "comparable to its siblings'"),
    }

    if groups is not None and getattr(splitter, "enforces_disjoint_groups", False):
        if out["max_groups_shared_between_train_and_val"] != 0:
            raise ValueError(
                f"{splitter!r} shares entities between train and validation "
                f"(max {out['max_groups_shared_between_train_and_val']}). It is a group "
                f"splitter, so this is a defect, not a finding."
            )
    return out


def validation_report(y_pool, val_fraction: float, folds: int, classes,
                      groups=None, order=None) -> dict:
    """What each selection strategy would actually contain on this dataset.

    Returns consequences, never a recommendation. Choosing between them is the agent's
    call (SKILL.md Step 6), and the numbers here are what that choice has to be justified
    from. Note especially `one_row_share_of_rarest_recall`: how far a single flipped row
    moves the rarest class's recall, which is what "is this estimate stable enough to
    select on" actually means.
    """
    counts = class_counts(y_pool)
    n = int(counts.sum())
    labels = [str(c) for c in classes]
    rarest_i = int(np.argmin(counts))
    rarest = int(counts[rarest_i])

    def share(per_estimate: float) -> float | None:
        return round(1.0 / per_estimate, 4) if per_estimate > 0 else None

    holdout_est = rarest * val_fraction
    fold_est = rarest / folds

    out = {
        "selection_metric": "macro-F1",
        "why_the_metric_decides_this": (
            "selection is on macro-F1, where every class contributes 1/n_classes of the "
            "score however small it is, so the smallest class - not the row count - sets "
            "the noise floor of the estimate being selected on"
        ),
        "pool_rows": n,
        "n_classes": len(counts),
        "class_counts": {lab: int(c) for lab, c in zip(labels, counts)},
        "rarest_class": labels[rarest_i],
        "rarest_count_in_pool": rarest,
        "holdout": {
            "val_fraction_of_pool": val_fraction,
            "val_rows": int(round(n * val_fraction)),
            "rarest_rows_per_estimate": round(holdout_est, 1),
            "rows_per_class_per_estimate": {lab: round(c * val_fraction, 1)
                                            for lab, c in zip(labels, counts)},
            "one_row_share_of_rarest_recall": share(holdout_est),
            "estimates": 1,
            "final_fit_rows": int(round(n * (1 - val_fraction))),
            "cost": "1x fitting; the validation rows are never trained on",
            "val_size_sensitivity": [
                {"val_fraction_of_pool": vs,
                 "rarest_rows_per_estimate": round(rarest * vs, 1),
                 "one_row_share_of_rarest_recall": share(rarest * vs)}
                for vs in (0.1, 0.2, 0.3, 0.4)
            ],
        },
        "kfold": {
            "folds": folds,
            "val_rows_per_fold": int(round(n / folds)),
            "rarest_rows_per_estimate": round(fold_est, 1),
            "rows_per_class_per_estimate": {lab: round(c / folds, 1)
                                            for lab, c in zip(labels, counts)},
            "one_row_share_of_rarest_recall": share(fold_est),
            "estimates": folds,
            "final_fit_rows": n,
            "cost": (f"{folds}x fitting; every row is validated against exactly once, so the "
                     f"score is the mean of {folds} estimates rather than a single one, and "
                     f"the final model still trains on all {n:,} rows"),
        },
        "note": ("these are consequences, not a recommendation - the choice is made in "
                 "SKILL.md Step 6 and must be justified in the report from these numbers"),
    }

    if groups is not None:
        out["groups"] = _group_consequences(y_pool, classes, groups, val_fraction, folds)
    if order is not None:
        out["order"] = _order_consequences(y_pool, classes, order, folds)

    return out


def _group_consequences(y_pool, classes, groups, val_fraction: float, folds: int) -> dict:
    """The same arithmetic as above, with the entity as the unit instead of the row.

    The quantum changes and it changes in the direction that hurts. With rows as the unit,
    one flipped row moves the rarest class's recall by 1/n. With entities as the unit, one
    *entity* crossing the split boundary moves every rare row that entity owns - so a
    single entity can move the estimate several times as far, and no amount of extra rows
    fixes it. That is why `rarest_entities_in_pool`, not `rarest_rows_in_pool`, is the
    number to decide on here.
    """
    y = np.asarray(y_pool)
    counts = class_counts(y)
    labels = [str(c) for c in classes]
    rarest_i = int(np.argmin(counts))
    rarest_rows = int(counts[rarest_i])
    rarest_label = labels[rarest_i]

    g = np.asarray(groups)
    _, inv = np.unique(g, return_inverse=True)
    n_groups = int(inv.max()) + 1
    rows_per_group = np.bincount(inv)
    rare_per_group = np.bincount(inv, weights=(y == rarest_i).astype(float))
    groups_with_rare = int((rare_per_group > 0).sum())
    most_rare_in_one = int(rare_per_group.max()) if len(rare_per_group) else 0

    def share(entities: float):
        return round(1.0 / entities, 4) if entities > 0 else None

    one_entity_moves = (round(most_rare_in_one / rarest_rows, 4) if rarest_rows else None)

    return {
        "unit_of_splitting": "entity",
        "n_groups": n_groups,
        "rows_per_group_mean": round(len(y) / n_groups, 2),
        "rows_per_group_median": float(np.median(rows_per_group)),
        "rows_per_group_max": int(rows_per_group.max()),
        "rarest_class": rarest_label,
        "rarest_rows_in_pool": rarest_rows,
        "rarest_entities_in_pool": groups_with_rare,
        "rare_rows_in_the_largest_single_entity": most_rare_in_one,
        "holdout": {
            "entities_per_estimate": round(groups_with_rare * val_fraction, 1),
            "one_entity_share_of_rarest_recall": share(groups_with_rare * val_fraction),
        },
        "kfold": {
            "folds": folds,
            "entities_per_estimate": round(groups_with_rare / folds, 1),
            "one_entity_share_of_rarest_recall": share(groups_with_rare / folds),
        },
        "why_this_is_worse_than_the_row_arithmetic": (
            f"with rows as the unit one flipped row moves {rarest_label!r} recall by "
            f"{round(1.0 / rarest_rows, 4) if rarest_rows else None}; with entities as the "
            f"unit one entity moves it by up to {one_entity_moves}, because every rare row "
            f"that entity owns travels with it. The estimate is coarser than the row count "
            f"suggests, and the count to decide on is the {groups_with_rare} entities, not "
            f"the {rarest_rows} rows"
        ),
        "note": ("consequences, not a recommendation - entity structure is decided in "
                 "SKILL.md Step 3 and the strategy in Step 6"),
    }


def _order_consequences(y_pool, classes, order, folds: int) -> dict:
    """What a rolling-origin split would look like, given the time order of the rows.

    Unlike the two above, this cannot be folded into a holdout/k-fold comparison: time
    folds are not exchangeable. Each one trains on a longer prefix than the last, so the
    fold scores come from models of different sizes and averaging them hides that.
    """
    y = np.asarray(y_pool)
    if isinstance(order, str):
        raise ValueError(
            f"validation_report has no frame to look {order!r} up in - pass the order as "
            f"an array aligned to y_pool (X[order].to_numpy()), or pass order= to the "
            f"splitter instead"
        )
    s = pd.Series(np.asarray(order)).reset_index(drop=True)
    if len(s) != len(y):
        raise ValueError(f"order has {len(s)} entries but y has {len(y)}")
    steps = s.diff().dropna()
    non_decreasing_share = round(float((steps >= 0).mean()), 6) if len(steps) else None
    counts = class_counts(y)
    rarest_i = int(np.argmin(counts))
    n = len(y)

    return {
        "unit_of_splitting": "time window",
        "n_rows": n,
        "n_distinct_times": int(s.nunique(dropna=True)),
        "first": str(s.iloc[0]) if n else None,
        "last": str(s.iloc[-1]) if n else None,
        "is_non_decreasing": bool((steps >= 0).all()) if len(steps) else None,
        "non_decreasing_share": non_decreasing_share,
        "folds": folds,
        "rows_per_fold_approx": int(round(n / (folds + 1))),
        "rarest_class": [str(c) for c in classes][rarest_i],
        "note": ("a rolling-origin split gives folds that are NOT exchangeable - each trains "
                 "on a longer prefix than the last, so the fold scores come from models of "
                 "different sizes. Report the spread across folds, not only the mean, and "
                 "remember the last fold is the only one that resembles deployment"),
        "warning": (None if (len(steps) == 0 or bool((steps >= 0).all())) else
                    "the rows are not in time order; temporal_folds will refuse to split "
                    "them until the frame is sorted by this column"),
    }


def format_validation_report(d: dict) -> str:
    """Render `validation_report` for a terminal."""
    h, k = d["holdout"], d["kfold"]
    pct = lambda v: "n/a" if v is None else f"{v:.2%}"
    lines = [
        "validation decision inputs - these are computed, not chosen; you choose (SKILL.md Step 6)",
        f"  selection metric : {d['selection_metric']} - {d['why_the_metric_decides_this']}",
        f"  pool             : {d['pool_rows']:,} rows, {d['n_classes']} classes",
        f"  rarest class     : {d['rarest_class']!r} with {d['rarest_count_in_pool']:,} rows",
        "",
        "  option         estimates   val rows   rarest rows/estimate   1 row =    final fit",
        f"  holdout {h['val_fraction_of_pool']:<5.0%}         {h['estimates']:>3}   "
        f"{h['val_rows']:>9,}   {h['rarest_rows_per_estimate']:>18}   "
        f"{pct(h['one_row_share_of_rarest_recall']):>7}   {h['final_fit_rows']:>9,}",
        f"  kfold k={k['folds']:<2}          {k['estimates']:>3}   {k['val_rows_per_fold']:>9,}   "
        f"{k['rarest_rows_per_estimate']:>18}   {pct(k['one_row_share_of_rarest_recall']):>7}   "
        f"{k['final_fit_rows']:>9,}",
        "",
        "  val_size sensitivity (holdout): " + " | ".join(
            f"{s['val_fraction_of_pool']:.0%} -> {s['rarest_rows_per_estimate']:g} rare rows"
            for s in h["val_size_sensitivity"]
        ),
        "",
        "  No recommendation is made here. Decide in SKILL.md Step 6 and justify it in the",
        "  report using the numbers above.",
    ]

    g = d.get("groups")
    if g:
        lines += [
            "",
            "  ENTITY STRUCTURE - the unit of splitting is the entity, not the row",
            f"  {g['n_groups']:,} entities; rows per entity mean {g['rows_per_group_mean']:g}, "
            f"median {g['rows_per_group_median']:g}, max {g['rows_per_group_max']:,}",
            f"  rarest class {g['rarest_class']!r}: {g['rarest_rows_in_pool']:,} rows spread "
            f"over {g['rarest_entities_in_pool']:,} entities",
            f"  one entity carries up to "
            f"{g['rare_rows_in_the_largest_single_entity']:,} of those rows, so one entity "
            f"crossing the boundary moves that class's recall by up to "
            f"{pct(g['holdout']['one_entity_share_of_rarest_recall'])}",
            f"  {g['why_this_is_worse_than_the_row_arithmetic']}",
        ]

    o = d.get("order")
    if o:
        lines += [
            "",
            "  TIME STRUCTURE - the unit of splitting is a time window",
            f"  {o['n_distinct_times']:,} distinct times from {o['first']} to {o['last']}",
            f"  non-decreasing: {o['is_non_decreasing']} "
            f"(share of steps that do not go backwards: {o['non_decreasing_share']})",
            f"  {o['folds']} folds, ~{o['rows_per_fold_approx']:,} validation rows each",
            f"  {o['note']}",
        ]
        if o.get("warning"):
            lines.append(f"  WARNING: {o['warning']}")

    return "\n".join(lines)
