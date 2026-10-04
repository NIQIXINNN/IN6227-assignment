#!/usr/bin/env python
"""Reading, profiling and evidence-gathering about a tabular frame.

Four of the five functions here return **evidence, not verdicts**. `label_candidates`,
`rarest_class_evidence`, `structure_evidence` and `model_selection_evidence` each enumerate
what they found together with the numbers behind it, and deliberately decline to pick.
Choosing the label, deciding whether a thin class is real, deciding whether rows are
independent, and deciding which model families suit the data are judgements that belong to
the agent and to the report - and a wrong one of any of them invalidates everything
downstream without raising anything.

The exception is `load_table`, which is a plain capability.

If you are about to add a default, a policy or a threshold to this file, it belongs in
SKILL.md instead.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from paths import refuse_bad_output_dir
from preprocessing import make_preprocessor

EVIDENCE_SCHEMA = 1   # bump when a key is renamed or removed; see make_pdf.require_evidence


def load_table(path) -> pd.DataFrame:
    """Read a CSV / TSV / XLSX into a frame with stripped column names."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif suffix in (".tsv", ".tab"):
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ------------------------------------------------------------------ label choice


def label_candidates(df: pd.DataFrame, max_classes: int = 50) -> list[dict]:
    """Every column that could plausibly be the target, with why it might be.

    Returns all of them, not a pick: choosing the label is the agent's job. `name_hint`
    and `looks_like_id` are evidence, not a verdict - read them and decide. If two
    candidates are close, ask the user rather than guessing.
    """
    hints = ("label", "class", "target", "outcome", "response", "y", "result")
    out = []
    for col in df.columns:
        nunique = int(df[col].nunique(dropna=True))
        is_numeric = bool(pd.api.types.is_numeric_dtype(df[col]))
        unique_per_row = nunique / max(len(df), 1)
        hint = str(col).strip().lower() in hints
        discrete = 2 <= nunique <= max_classes
        out.append({
            "column": str(col),
            "dtype": str(df[col].dtype),
            "n_unique": nunique,
            "n_missing": int(df[col].isna().sum()),
            "unique_per_row": round(unique_per_row, 4),
            "name_matches_target_convention": hint,
            "plausible_class": bool(discrete and not (is_numeric and unique_per_row > 0.2)),
            "looks_like_id": bool(nunique == len(df) and not pd.api.types.is_float_dtype(df[col])),
        })
    return out


def classify_columns(df: pd.DataFrame, label: str) -> tuple[list[str], list[str], list[str]]:
    """Split the remaining columns into (numeric, categorical, dropped_as_id_like)."""
    numeric, categorical, dropped = [], [], []
    for col in df.columns:
        if col == label:
            continue
        nunique = df[col].nunique(dropna=True)
        if nunique == len(df) and not pd.api.types.is_float_dtype(df[col]):
            dropped.append(str(col))
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(str(col))
        else:
            categorical.append(str(col))
    return numeric, categorical, dropped


# ------------------------------------------------------------------------- eda


def profile(df: pd.DataFrame, label: str, numeric: list[str], categorical: list[str],
            dropped: list[str], label_reason: str = "") -> dict:
    """The EDA facts a report needs. Facts only - no interpretation."""
    counts = df[label].value_counts(dropna=True)
    imbalance = float(counts.max() / counts.min()) if len(counts) > 1 and counts.min() else None
    return {
        "_evidence": {"type": "eda", "schema_version": EVIDENCE_SCHEMA, "producer": "profile"},
        "n_rows": int(len(df)),
        "n_features": int(len(df.columns) - 1),
        "label_column": str(label),
        "label_reason": label_reason,
        "n_classes": int(len(counts)),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "imbalance_ratio": imbalance,
        "numeric_features": list(numeric),
        "categorical_features": list(categorical),
        "dropped_id_like": list(dropped),
        "missing_per_column": {str(c): int(n) for c, n in df.isna().sum().items() if n},
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_summary": {
            c: {
                "mean": float(df[c].mean()),
                "std": float(df[c].std()) if len(df) > 1 else None,
                "min": float(df[c].min()),
                "median": float(df[c].median()),
                "max": float(df[c].max()),
                "skew": float(df[c].skew()) if len(df) > 2 else None,
            }
            for c in numeric
        },
    }


# Reporting threshold, not a decision: at or above this many levels a column adds that many
# columns under one-hot, which is the number that actually constrains a linear model or a
# kernel. Which columns to list is arithmetic; whether that rules a family out is not.
HIGH_CARDINALITY_LEVELS = 20


def model_selection_evidence(df: pd.DataFrame, label: str, numeric: list[str],
                             categorical: list[str]) -> dict:
    """The dataset-shape facts a model-family choice has to be argued from.

    Step 4 asks which families suit *this* dataset, and that question is only answerable
    from numbers. Without them the argument collapses into a claim about the algorithms -
    "a tree needs no scaling, logistic regression does" - which is equally true of every
    dataset in existence and therefore justifies nothing about this one. That is the exact
    substitution this function exists to make visible: the reason has to cite a number
    below, or it is not a reason for this dataset.

    Facts here, no ranking. Nothing in this dict says a family is suitable.

      * **size and dimensionality**, and their ratio - distance-based methods degrade as
        the column count grows relative to the row count, and no accuracy figure warns you.
      * **`one_hot_width`** - how wide the frame becomes when categoricals are expanded.
        This, not the raw column count, is what a linear model or a kernel actually faces.
      * **categorical cardinality** - one column with 5,000 levels is a different problem
        from 50 columns with 2, even at the same total width.
      * **class balance** - it decides whether a family that optimises plain accuracy is
        admissible at all.
      * **constant / near-constant columns and missing share** - how much of the frame
        carries nothing.
    """
    feats = list(numeric) + list(categorical)
    counts = df[label].value_counts(dropna=True)
    card = {str(c): int(df[c].nunique(dropna=True)) for c in categorical}
    n_rows, n_feat = int(len(df)), len(feats)

    constant, near_constant = [], []
    for c in feats:
        top = df[c].value_counts(dropna=True)
        if not len(top):
            continue
        share = float(top.iloc[0]) / max(1, int(top.sum()))
        if len(top) <= 1 or share >= 0.999:
            constant.append(str(c))
        elif share >= 0.99:
            near_constant.append(str(c))

    return {
        "n_rows": n_rows,
        "n_features": n_feat,
        "n_numeric": len(numeric),
        "n_categorical": len(categorical),
        "rows_per_feature": round(n_rows / n_feat, 1) if n_feat else None,
        "n_classes": int(len(counts)),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "imbalance_ratio": (round(float(counts.max() / counts.min()), 4)
                            if len(counts) > 1 and counts.min() else None),
        "rarest_class": str(counts.idxmin()) if len(counts) else None,
        "rarest_rows": int(counts.min()) if len(counts) else 0,
        "one_hot_width": n_feat - len(categorical) + sum(card.values()),
        "categorical_cardinality": card,
        "max_cardinality": max(card.values()) if card else 0,
        "median_cardinality": float(np.median(list(card.values()))) if card else 0.0,
        "high_cardinality_columns": sorted(c for c, n in card.items()
                                           if n >= HIGH_CARDINALITY_LEVELS),
        "constant_columns": constant,
        "near_constant_columns": near_constant,
        "missing_share": (round(float(df[feats].isna().to_numpy().mean()), 6)
                          if n_feat else 0.0),
        "duplicate_row_share": round(float(df.duplicated().mean()), 6) if n_rows else 0.0,
        "note": ("facts about this dataset's shape, not a ranking. Read them against the "
                 "capability table from describe_models(), and eliminate a family only for a "
                 "reason that names one of these numbers (SKILL.md Step 4)"),
    }


# Label values that usually name a bucket rather than a category - a "class" called `other`
# or `unknown` is often whatever did not fit. Not always: "other" can be a real category the
# task has. The flag reports the name and leaves the reading to the report, which is why
# `looks_like_placeholder_name` is a boolean about a STRING, not a judgement about a class.
PLACEHOLDER_LABELS = {
    "other", "others", "unknown", "unk", "n/a", "na", "nan", "none", "null", "nil",
    "?", "-", "--", "", "misc", "miscellaneous", "unspecified", "not specified",
    "no label", "missing", "error", "invalid", "test", "tmp", "temp", "todo", "void",
}


def rarest_class_evidence(df: pd.DataFrame, label: str, numeric: list[str],
                          categorical: list[str], scan_limit: int = 200_000) -> dict:
    """Evidence about the rarest class - four separate questions, no verdict.

    When the rarest class is thin, the first question is NOT "which validation strategy".
    It is what the evidence about this class actually says, and that is four questions
    rather than one: is the name meaningful, are the rows duplicated, do identical rows
    carry different labels, and do the rows separate locally. A class can be entirely real
    and still fail the separation question, so more than one reading is possible at once
    and the responses differ - one is a hard task to model and report honestly, another is
    a cleaning decision, and keeping a duplicated or catch-all class damages every number
    in the report.

    This returns the evidence, not a verdict:

      * `looks_like_placeholder_name` - the class is literally called `other`/`unknown`/...
        A name like that MAY be a bucket for whatever did not fit, or a legitimate "other"
        the task really has; the name alone does not settle it.
      * `duplicate_rows_within_class` / `effective_unique_rows` - 12 rows may be 3 records
        repeated four times, so only the unique count is a defensible sample size.
      * `contradicted_rows` - rows in this class whose FEATURE VECTOR is identical to a row
        labelled something else. Identical inputs under two labels may be label noise, a
        duplicate conflict, genuine ambiguity, or a discriminative column that is missing;
        it is the strongest single signal here, and which of those it is remains a
        judgement.
      * `neighbour_same_class_share` against `chance_baseline` - for each rare row, is its
        nearest neighbour in feature space also rare? A share near the chance baseline says
        THESE COLUMNS give the class little local structure; a real class can be spread
        through the feature space just as easily. A share far above says only that it is
        separable here - which is a fact about the columns, not about how meaningful the
        class is.

    `scan_limit` bounds the number of ROWS the neighbour search runs over, not the size of
    the rare class. Scoping it to the rare class would have been backwards: the scan is
    most useful for a class in the hundreds or thousands, which is exactly what a
    rare-class-sized limit skips. The search itself is tree-backed and near-linear, so the
    limit only bites at extreme row counts, and when it does the subsampling is stated in
    `neighbour_search_note` rather than silently returning nothing.
    """
    counts = df[label].value_counts(dropna=True)
    rare_label = str(counts.idxmin())
    n_rare = int(counts.min())
    n = int(len(df))

    y_str = df[label].astype(str).to_numpy()
    is_rare = y_str == rare_label

    out: dict = {
        "rarest_class": rare_label,
        "n_rows": n_rare,
        "share_of_rows": round(n_rare / n, 6),
        "looks_like_placeholder_name": rare_label.strip().lower() in PLACEHOLDER_LABELS,
        "note": "evidence, not a verdict - deciding whether to model, merge or drop this "
                "class is a judgement and belongs in the report",
    }

    if not numeric and not categorical:
        # Without feature columns every row has an identical (empty) key, so the
        # duplicate and contradiction counts below would silently come back as "all rows"
        # - a confident, completely wrong answer. Refuse instead.
        raise ValueError(
            "rarest_class_evidence needs the feature columns: pass the numeric and "
            "categorical lists from classify_columns(). With no features every row looks "
            "like a duplicate of every other."
        )

    feats = df[numeric + categorical]
    key = feats.astype(str).agg("|".join, axis=1)

    rare_key = key[is_rare]
    out["duplicate_rows_within_class"] = int(rare_key.duplicated().sum())
    out["effective_unique_rows"] = int(rare_key.nunique())

    # Identical features, different labels: a contradiction, i.e. label noise.
    other_keys = set(key[~is_rare])
    out["contradicted_rows"] = int(rare_key.isin(other_keys).sum())

    # These three are always present (null when the scan is skipped) so that callers can
    # read a fixed set of keys without checking whether the search ran.
    out["neighbour_same_class_share"] = None
    out["chance_baseline"] = None
    out["lift_over_chance"] = None
    out["neighbour_search_note"] = None

    if n_rare < 2:
        out["skipped_neighbour_search"] = (
            f"class has {n_rare} row(s); nothing to compare against"
        )
        return out

    # Subsample the FRAME, never the rare class on its own: the point of the scan is to ask
    # where each rare row sits, so all of them should be queried. Dropping majority rows
    # only makes the reference set sparser, which is why the note records the class balance
    # of what was actually searched.
    rare_idx = np.flatnonzero(is_rare)
    if n > scan_limit:
        rng = np.random.default_rng(0)
        other = np.flatnonzero(~is_rare)
        room = max(scan_limit - len(rare_idx), 1)
        if room < len(other):
            keep = np.concatenate([rare_idx, rng.choice(other, room, replace=False)])
            out["neighbour_search_note"] = (
                f"frame has {n:,} rows, above the {scan_limit:,}-row scan limit; all "
                f"{len(rare_idx):,} rare rows were kept and the majority was subsampled to "
                f"{room:,}, so the chance baseline describes the sample, not the frame"
            )
        else:
            keep = np.arange(n)
    else:
        keep = np.arange(n)

    Z = np.asarray(
        make_preprocessor(numeric, categorical, scale=True).fit_transform(feats.iloc[keep]),
        dtype=float,
    )
    keep_is_rare = is_rare[keep]
    pos_rare = np.flatnonzero(keep_is_rare)

    # Tree-backed exact search instead of a Python loop over distance vectors: the loop was
    # O(n_rare x n), which is what forced the old (and mis-scoped) cost guard. scipy rather
    # than sklearn.neighbors because the sklearn path routes through threadpoolctl, which
    # older versions raise on when they cannot parse the MKL version - an environment
    # failure inside a primitive whose whole job is to produce a number.
    _, ind = cKDTree(Z).query(Z[pos_rare], k=2)
    share = float(keep_is_rare[ind[:, 1]].mean())    # column 0 is the row itself
    baseline = (int(keep_is_rare.sum()) - 1) / (len(keep) - 1)

    out["neighbour_same_class_share"] = round(share, 4)
    out["chance_baseline"] = round(baseline, 6)
    out["lift_over_chance"] = round(share / baseline, 2) if baseline > 0 else None
    out["skipped_neighbour_search"] = None
    return out


# Column names that suggest a row is one of several belonging to the same entity. Matched
# on whole name tokens, so `patient_id` and `user` match while `grid` and `width` do not.
ENTITY_NAME_HINTS = (
    "id", "key", "group", "patient", "subject", "user", "customer", "client", "session",
    "account", "device", "household", "family", "store", "shop", "person", "student",
    "employee", "member", "order", "invoice", "ticket", "case", "sample", "specimen",
    "batch", "site", "visit", "clip", "match", "team", "school", "firm", "company",
    "hospital", "ward", "sensor", "machine", "asset", "vehicle",
)

# Column names that suggest a date or time.
TIME_NAME_HINTS = (
    "date", "time", "timestamp", "datetime", "day", "week", "month", "quarter", "year",
    "period", "epoch", "created", "updated", "start", "end", "begin", "closed", "opened",
)


def _name_matches(column, hints) -> bool:
    """True when any whole token of `column` is one of `hints`.

    Token equality rather than substring, because substring matching on a short hint like
    "id" fires on `grid`, `width` and `valid` - and a false positive here would tell the
    agent to hold a real feature out of the model.
    """
    parts = re.split(r"[^a-z0-9]+", str(column).strip().lower())
    return any(part in hints for part in parts)


def _try_datetime(s: pd.Series):
    """Return (parsed_series_or_None, parse_rate). Never guesses for numeric columns.

    A numeric column is deliberately not parsed: the integer 1975 is a year, a count, a
    price or a score, and nothing in the column itself says which. Turning it into a date
    is an inference, not a reading, so it is left to the caller - `structure_evidence`
    still reports the column when its *name* suggests a time, and says the unit needs
    confirming.
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s, 1.0
    if pd.api.types.is_numeric_dtype(s):
        return None, 0.0
    try:
        # pandas warns per column that it had to guess the date format. That guess is
        # exactly what this function is here to make, and the resulting parse rate is
        # reported - so the warning is noise, and on a 16-column frame it is 16 lines of it.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s, errors="coerce")
    except Exception:
        return None, 0.0
    rate = float(parsed.notna().mean()) if len(s) else 0.0
    return (parsed, rate) if rate >= 0.9 else (None, rate)


def structure_evidence(df: pd.DataFrame, label: str, numeric: list[str],
                       categorical: list[str], dropped: list[str] | None = None,
                       scan_limit: int = 5_000) -> dict:
    """Evidence about what a ROW is - an independent observation, or one of several per entity.

    Every splitter in `validation` assumes rows are independent. When they are not, the
    split that looks correct puts the same entity in train and in validation, the model
    recognises the entity rather than learning the task, and the score comes back high. It
    does not raise. It does not warn. That is the failure this function exists to make
    visible *before* a split is chosen.

    Two kinds of structure are reported, neither as a verdict:

    **Entity keys.** A column where a value covers several rows may mean each row is a
    repeat observation of one entity (a patient, a user, a device). The discriminating
    test is `same_value_share_nearest_neighbour`: for each row, is its nearest neighbour in
    the feature space a row carrying the same value? The candidate column is **held out of
    the feature space** for that test. It has to be - left in, a candidate makes its own
    rows distance-zero from each other and every single column looks like a group key.
    Read the result against `chance_baseline_share`, which is the probability that two
    randomly chosen rows share the value.

    The caveat, stated here rather than buried: **a high lift on its own does not prove the
    column is an entity key.** It proves rows sharing the value resemble each other in the
    other columns - which is true both of a genuine entity key and of a strong ordinary
    predictor like `region`. What separates them is `rows_per_value_*` (an entity key
    repeats a handful of times; a feature repeats thousands) together with the name. A
    column that is a plausible predictor belongs in `numeric`/`categorical`; a column that
    identifies an entity must stay out of both and be passed as `groups` to the splitter.

    **Time.** A column that parses as a date, whether the frame is already sorted by it,
    and how many distinct values it holds. `temporal_folds` needs the order column
    explicitly and refuses an unsorted one, so what matters here is whether an order
    exists at all.

    `dropped` columns are reviewed separately: `classify_columns` drops a column with one
    distinct value per row as id-like, which is right for modelling and wrong for
    forgetting - a `patient_id` that happens to be unique in *this* file is still the key
    that will repeat in the next one.

    `scan_limit` bounds the rows the neighbour search runs over. It is far smaller than the
    limit in `rarest_class_evidence`, and for a different reason: that scan queries only the
    rare rows against the tree, while this one queries **every** row, once per candidate
    column. In the ~40 dimensions a one-hot frame produces, a KD-tree degrades towards brute
    force, so the cost grows with the square of the rows times the number of columns - 31k
    rows across 15 candidates took over four minutes at full size. The statistic is a share,
    which settles quickly, so a 5,000-row sample gives the same reading in seconds. When the
    limit bites, `search_note` says the numbers describe the sample.
    """
    dropped = list(dropped or [])
    n = int(len(df))
    feature_cols = list(numeric) + list(categorical)

    group_candidates: list[dict] = []
    time_candidates: list[dict] = []
    dropped_reviewed: list[dict] = []
    not_examined: list[dict] = []

    for col in df.columns:
        if col == label:
            continue
        col = str(col)
        s = df[col]
        nunique = int(s.nunique(dropna=True))

        # ---------------------------------------------------------------- time
        parsed, parse_rate = _try_datetime(s)
        time_hint = _name_matches(col, TIME_NAME_HINTS)
        if parsed is not None or (time_hint and nunique > 1):
            entry = {
                "column": col,
                "n_unique": nunique,
                "name_matches_time_convention": time_hint,
                "parsed_as_datetime": parsed is not None,
                "parse_rate": round(parse_rate, 4),
                "first": str(parsed.min()) if parsed is not None else None,
                "last": str(parsed.max()) if parsed is not None else None,
                "n_missing_after_parse": int(parsed.isna().sum()) if parsed is not None else None,
                "is_non_decreasing": (bool((parsed.diff().dropna() >= pd.Timedelta(0)).all())
                                      if parsed is not None else None),
                "non_decreasing_share": (round(float((parsed.diff().dropna()
                                                      >= pd.Timedelta(0)).mean()), 6)
                                         if parsed is not None else None),
            }
            if parsed is None:
                entry["note"] = (
                    "value is numeric and its name suggests a time, but a bare number is "
                    "not parsed as a date here - 1975 could be a year, a count or a price. "
                    "Confirm the unit before treating it as an order."
                )
            elif not entry["is_non_decreasing"]:
                entry["note"] = (
                    "rows are NOT in time order, so temporal_folds would train on the "
                    "future if given the row positions as they stand. Sort by this column "
                    "first, or pass order= to select it explicitly."
                )
            else:
                entry["note"] = "rows are already in non-decreasing time order"
            time_candidates.append(entry)

        # --------------------------------------------------------------- group
        if 2 <= nunique < n:
            entry = {
                "column": col,
                "n_unique": nunique,
                "rows_per_value_mean": round(n / nunique, 2),
                "rows_per_value_median": float(s.value_counts().median()),
                "rows_per_value_max": int(s.value_counts().max()),
                "name_matches_entity_convention": _name_matches(col, ENTITY_NAME_HINTS),
                "is_in_feature_set": col in feature_cols,
                "example_values": [str(v) for v in s.dropna().unique()[:3]],
                "same_value_share_nearest_neighbour": None,
                "chance_baseline_share": None,
                "lift_over_chance": None,
                "lift_withheld_because": None,
                "rows_searched": None,
                "rows_whose_neighbour_shares_the_value": None,
                "search_note": None,
            }
            if col in dropped:
                entry["is_in_feature_set"] = False
                entry["note"] = (
                    "id-like and already excluded from the feature set - but a key that is "
                    "unique in this file can repeat in another, so check whether its name "
                    "describes an entity"
                )
            if feature_cols:
                share, baseline, note, n_searched = _neighbour_same_value_share(
                    s.astype(str).to_numpy(), df[feature_cols].drop(columns=[col],
                                                                    errors="ignore"),
                    scan_limit=scan_limit,
                )
                entry["same_value_share_nearest_neighbour"] = share
                entry["chance_baseline_share"] = baseline
                events = round(share * n_searched, 1) if share is not None else None
                entry["rows_searched"] = n_searched
                entry["rows_whose_neighbour_shares_the_value"] = events

                if share is None or not baseline:
                    entry["lift_withheld_because"] = "the neighbour search did not run"
                elif entry["rows_per_value_mean"] < 2:
                    # Near-unique values make the ratio a trap rather than a finding: "the
                    # nearest neighbour shares this value" degenerates into "the nearest
                    # neighbour is this row's near-duplicate", the baseline goes to ~0, and
                    # the quotient explodes. A continuous score column measured 59x on this
                    # dataset for exactly this reason, and 59x reads like a screaming entity
                    # key. The share and the baseline are still reported; only the
                    # misleading quotient is withheld.
                    entry["lift_withheld_because"] = (
                        f"only {entry['rows_per_value_mean']:g} rows per value on average, "
                        f"so 'same value' nearly always means 'the same row' and the ratio "
                        f"measures nothing. Read the share and the baseline directly."
                    )
                elif events is not None and events < MIN_SHARE_EVENTS:
                    # Both numbers are near zero, so their ratio is noise. Two columns on
                    # this dataset came out at share 0.0000 against baseline 0.00018 - a
                    # "lift" of 0.0 computed from fewer than one event. A share estimated
                    # from under ~25 events carries a relative standard error above 20%,
                    # which is not a measurement.
                    entry["lift_withheld_because"] = (
                        f"only about {events:g} of the {n_searched:,} rows searched have a "
                        f"same-value nearest neighbour at all, so the ratio is noise rather "
                        f"than a measurement. Either the values are nearly all distinct - a "
                        f"continuous measurement rather than a grouping - or the rows "
                        f"sharing a value are simply not close to each other."
                    )
                else:
                    entry["lift_over_chance"] = round(share / baseline, 2)
                entry["search_note"] = note
            else:
                entry["search_note"] = (
                    "no feature columns to search in, so the neighbour test was skipped"
                )
            group_candidates.append(entry)
            if col in dropped:
                dropped_reviewed.append({"column": col, "n_unique": nunique,
                                         "rows_per_value_max": entry["rows_per_value_max"]})
        else:
            not_examined.append({
                "column": col,
                "n_unique": nunique,
                "why": ("one distinct value per row" if nunique == n
                        else "fewer than two distinct values"),
            })

    # Presentation order only - strongest lift first, nulls last. This is not a verdict:
    # read `rows_per_value_*` and the name alongside it before concluding anything.
    group_candidates.sort(
        key=lambda e: (e["lift_over_chance"] is None, -(e["lift_over_chance"] or 0))
    )

    return {
        "n_rows": n,
        "feature_columns_searched": feature_cols,
        "group_candidates": group_candidates,
        "time_candidates": time_candidates,
        "dropped_id_like_reviewed": dropped_reviewed,
        "columns_not_examined": not_examined,
        "how_to_read_this": (
            "A high lift_over_chance means rows sharing this column's value resemble each "
            "other in the OTHER columns - true of a real entity key and also of a strong "
            "ordinary predictor. Read rows_per_value_mean alongside it: a column with only "
            "a row or two per value measures as a huge lift for arithmetic reasons, so its "
            "lift is withheld and lift_withheld_because says why. A handful of repeats per "
            "value is what a plausible entity key looks like. If "
            "a column IS an entity key it must not enter numeric/categorical: pass it as "
            "groups= to the splitter instead. Nothing here is a verdict - decide and say "
            "which evidence decided it."
        ),
        "note": ("evidence, not a verdict - whether the rows are independent is a judgement "
                 "made in SKILL.md Step 3 and recorded in the report"),
    }


# A share of the rows is only a measurement if enough rows exhibit the event. Below roughly
# 25 events the relative standard error of the share passes 20%, and since the baseline can
# be just as small, their ratio is noise - which is how two ordinary columns on the shipped
# dataset came back with a "lift" of 0.0 computed from less than one event.
MIN_SHARE_EVENTS = 25


def _neighbour_same_value_share(values: np.ndarray, feats: pd.DataFrame,
                                scan_limit: int = 200_000):
    """(share, baseline, note, n_searched) - is a row's nearest neighbour a row with the same value?

    `feats` must already EXCLUDE the column `values` came from. Left in, the column makes
    every one of its own rows distance zero apart and the answer is 1.0 for every column
    in the frame, which is worse than no answer because it looks like a result.

    `baseline` is the chance that two distinct rows share the value at all; the share is
    only interpretable against it.
    """
    if feats.shape[1] == 0:
        return None, None, "no feature columns to search in", 0

    n = len(values)
    note = None
    if n > scan_limit:
        # Sample by VALUE, never uniformly by row. A uniform sample of 5,000 rows from a
        # frame whose entities hold 5 rows each leaves most entities represented zero or
        # once - it destroys the exact repeat structure this test exists to measure, and a
        # genuine entity key then scores at chance. Taking whole values keeps every entity
        # intact, and the per-value cap stops a single dominant value (a `region` covering
        # 90% of the frame) from eating the whole budget.
        rng = np.random.default_rng(0)
        uniq_all, _ = np.unique(values, return_counts=True)
        per_value_cap = max(2, scan_limit // max(len(uniq_all), 1))
        rows_of: dict = {}
        for i, v in enumerate(values):
            rows_of.setdefault(v, []).append(i)

        taken: list[int] = []
        for v in uniq_all[rng.permutation(len(uniq_all))]:
            rows = rows_of[v]
            if len(rows) > per_value_cap:
                rows = list(rng.choice(rows, per_value_cap, replace=False))
            taken.extend(rows)
            if len(taken) >= scan_limit:
                break

        keep = np.sort(np.asarray(taken, dtype=int))
        values, feats = values[keep], feats.iloc[keep]
        note = (f"frame has {n:,} rows, above the {scan_limit:,}-row search limit; searched "
                f"{len(keep):,} rows sampled by value (whole values, at most "
                f"{per_value_cap} rows each) so the repeat structure survives. The share "
                f"and the baseline both describe this sample, not the frame.")
        n = len(values)

    if n < 2:
        return None, None, "fewer than two rows", n

    num = [c for c in feats.columns if pd.api.types.is_numeric_dtype(feats[c])]
    cat = [c for c in feats.columns if c not in num]
    Z = np.asarray(make_preprocessor(num, cat, scale=True).fit_transform(feats), dtype=float)

    # scipy rather than sklearn.neighbors: the sklearn path routes through threadpoolctl,
    # older versions of which raise on MKL installs whose version they cannot parse.
    _, ind = cKDTree(Z).query(Z, k=2)
    share = float((values[ind[:, 1]] == values).mean())

    _, counts = np.unique(values, return_counts=True)
    baseline = float((counts * (counts - 1)).sum() / (n * (n - 1)))
    return round(share, 4), round(baseline, 6), note, n


def save_figures(df: pd.DataFrame, label: str, numeric: list[str], out_dir) -> Path:
    """Write the standard EDA figures into <out_dir>/figures and return the directory.

    `out_dir` is the experiment folder's `out/` - this function appends `figures/` itself.
    That is the opposite of `evaluate` and `save_confusion_pair`, which take the figures
    directory directly; see the table in SKILL.md Step 0.
    """
    fig_dir = Path(out_dir) / "figures"
    refuse_bad_output_dir(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    counts = df[label].value_counts()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar([str(i) for i in counts.index], counts.values, color="#4c78a8")
    ax.set_title(f"Class balance: {label}")
    ax.set_ylabel("count")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(fig_dir / "class_balance.png", dpi=140)
    plt.close(fig)

    if numeric:
        cols = numeric[:9]
        ncol = 3
        nrow = int(np.ceil(len(cols) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.3 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax, col in zip(axes, cols):
            ax.hist(df[col].dropna(), bins=30, color="#4c78a8")
            ax.set_title(col, fontsize=8)
        for ax in axes[len(cols):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(fig_dir / "numeric_histograms.png", dpi=140)
        plt.close(fig)

        if len(numeric) > 1:
            corr = df[numeric].corr()
            fig, ax = plt.subplots(figsize=(5.5, 4.6))
            im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr)))
            ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
            ax.set_yticks(range(len(corr)))
            ax.set_yticklabels(corr.columns, fontsize=7)
            ax.set_title("Numeric correlation")
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            fig.savefig(fig_dir / "correlation.png", dpi=140)
            plt.close(fig)

    return fig_dir
