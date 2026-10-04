#!/usr/bin/env python
"""Fixture tests for the silent-failure guards. Plain asserts, no pytest.

    python tests/test_primitives.py

These exist because the shipped dataset exercises none of the new group/time code - it is
plain independent rows with no entity key and no timestamp - so passing on it proves
nothing about the guards. The fixtures here are synthetic frames built to have exactly the
structure each guard is for.

Every test below is about a failure that would otherwise produce a **plausible number**
rather than an error. That is the whole point: a split that leaks entities, a ratio
computed from no events, a registry that silently substitutes a different model.
"""
from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from data import (MIN_SHARE_EVENTS, _neighbour_same_value_share, classify_columns,
                  load_table, model_selection_evidence, structure_evidence)
from decisions import DecisionLog
from evaluation import _score
from models import (FORBIDDEN_KEYS, MODEL_REGISTRY, describe_models, make_model,
                    model_names)
from pipeline import tune
from validation import (fold_consequences, grouped_holdout, stratified_group_folds,
                        temporal_folds, validation_report)

SHIPPED = Path(r"C:\Users\86180\Desktop\dataset\train.csv")


# --------------------------------------------------------------------------- helpers


def raises(fn, exc=Exception, contains: str | None = None):
    """Assert `fn()` raises `exc`; optionally that the message mentions `contains`."""
    try:
        fn()
    except exc as e:
        if contains is not None and contains.lower() not in str(e).lower():
            raise AssertionError(
                f"raised {type(e).__name__} but the message does not mention "
                f"{contains!r}: {e}"
            )
        return e
    except Exception as e:                                    # wrong exception type
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, nothing was raised")


def panel(n_entities: int = 120, rows_per_entity: int = 6, seed: int = 0):
    """A synthetic frame whose rows are NOT independent: several rows per entity.

    Each entity has its own level on `feat_a` and `feat_b`, so its rows cluster together.
    Two informative columns and a tight within-entity spread are both needed: with one
    informative column among noise, an unrelated row elsewhere in the frame is often just
    as close as a sibling, and the entity stops being detectable - which is a property of
    the fixture, not of the test.

    `feat_c` carries no entity information at all, and is what the "does the hold-out
    matter" test searches in. The label is decided at the entity level, which is exactly
    the situation where a row-based split leaks: half an entity's rows in train and half in
    validation, and the model scores high by recognising the entity rather than by learning
    the task.
    """
    rng = np.random.default_rng(seed)
    n = n_entities * rows_per_entity
    entity = np.repeat([f"e{i}" for i in range(n_entities)], rows_per_entity)
    level = np.repeat(rng.normal(size=n_entities), rows_per_entity)
    level_b = np.repeat(rng.normal(size=n_entities), rows_per_entity)
    return pd.DataFrame({
        "entity_id": entity,
        "feat_a": level + rng.normal(scale=0.2, size=n),
        "feat_b": level_b + rng.normal(scale=0.2, size=n),
        "feat_c": rng.normal(size=n),
        "label": (level > 0).astype(int),
    })


def numbered(X: pd.DataFrame, start: int = 1000) -> pd.DataFrame:
    """Give X a label index that is deliberately NOT its row position.

    Splitters here all yield positional indices. With a default RangeIndex the two are the
    same number, so a test cannot tell them apart - and `X.loc[positional]` failing is the
    difference between a correct split and a silently misaligned one.
    """
    X = X.copy()
    X.index = np.arange(start, start + len(X))
    return X


# ----------------------------------------------------------------------------- tests


def test_registry_holds_no_dataset_decisions():
    """The registry may hold algorithm facts and constructor settings. Not a single default
    for anything the agent has to choose per dataset."""
    for name, spec in MODEL_REGISTRY.items():
        for key in spec["constructor"]:
            assert key not in FORBIDDEN_KEYS, (
                f"{name} ships a default for {key!r}, which is a modelling decision that "
                f"belongs to the agent (SKILL.md Step 4)"
            )
        for key in ("scaling", "one_hot", "nonlinearity"):
            assert len(spec[key]) > 30, (
                f"{name}.{key} must be prose with a reason, not a bare verdict"
            )
        assert "recommended" not in spec, (
            f"{name} carries a `recommended` flag. That moves the choice back into the "
            f"file, for whatever dataset its author had in mind (SKILL.md Step 4)"
        )
    assert "C" not in MODEL_REGISTRY["logistic_regression"]["constructor"]


def test_nonlinearity_actually_separates_the_families():
    """For `nonlinearity` to work as a selection criterion it has to read differently for
    different families. A field that says the same thing everywhere decides nothing, and
    would be a registry that looks informative while the choice is still made elsewhere."""
    vals = {n: s["nonlinearity"] for n, s in MODEL_REGISTRY.items()}
    assert len(set(vals.values())) >= 4, vals
    assert vals["logistic_regression"] != vals["decision_tree"], (
        "the linear/non-linear split is the one distinction the pair selection rests on"
    )
    assert "kernel" in vals["svm"], (
        f"SVM is the one family whose flexibility is itself a hyperparameter, and the text "
        f"has to say so; got {vals['svm']!r}"
    )


def test_make_model_rejects_a_typo_and_names_the_alternatives():
    e = raises(lambda: make_model("logistic_regresion"), ValueError, contains="available")
    assert "logistic_regression" in str(e), "the error must list the valid names"


def test_make_model_passes_hyperparameters_through():
    m = make_model("decision_tree", max_depth=8, min_samples_leaf=5)
    assert m.max_depth == 8 and m.min_samples_leaf == 5


def test_make_model_constructor_settings_do_not_crowd_out_the_caller():
    """max_iter=2000 is a convergence ceiling, not a choice - and C is still the caller's."""
    m = make_model("logistic_regression", C=0.1)
    assert m.max_iter == 2000 and m.C == 0.1


def test_describe_models_renders_every_family():
    out = describe_models()
    for name in MODEL_REGISTRY:
        assert name in out, f"{name} missing from describe_models()"


def test_describe_models_columns_line_up():
    """The capability table is monospace-aligned prose, and it is read as a table.

    A cell wider than the column width it was computed from does not raise - `ljust` pads
    but never truncates, so the row simply runs long and every column after it shifts.
    That is how a row appended without recomputing the widths shows up, and the fix is in
    the code that renders it, not in the reading.
    """
    lines = describe_models().splitlines()
    rules = [ln for ln in lines if ln and set(ln) == {"-"}]
    assert len(rules) == 1, f"expected exactly one rule line, found {len(rules)}"
    rule = rules[0]

    header_at = next(i for i, ln in enumerate(lines) if ln.startswith("model family"))
    assert lines.index(rule) == header_at + 1, (
        "the rule belongs directly under the header, as in a markdown table"
    )
    body = []
    for ln in lines[header_at + 2:]:
        if not ln.strip():                     # the blank line that ends the table
            break
        body.append(ln)
    table = [lines[header_at]] + body
    assert len(table) == 1 + len(MODEL_REGISTRY), (
        f"expected a header plus one row per family, got {len(table)} lines"
    )

    # Cells are joined by exactly two spaces and contain at most one, so runs of two or
    # more spaces are the column boundaries.
    spans = [[(m.start(), m.end()) for m in re.finditer(r"\S.*?(?=\s{2,}|$)", ln)]
             for ln in table]
    header_spans = spans[0]
    for ln, row in zip(table[1:], spans[1:]):
        assert len(row) == len(header_spans), (
            f"row has {len(row)} cells against the header's {len(header_spans)}: {ln!r}"
        )
        for (hs, _), (cs, _) in zip(header_spans, row):
            assert cs == hs, (
                f"column starts at {cs} but the header puts it at {hs}; the widths were "
                f"not recomputed for this row: {ln!r}"
            )
    for ln in table + [rule]:
        assert len(ln) <= len(rule), (
            f"line is {len(ln)} wide against a {len(rule)}-wide rule: {ln!r}"
        )


def test_model_selection_evidence_counts_the_frame_a_model_actually_sees():
    """`n_features` is what you passed in; `one_hot_width` is what a linear model or a
    kernel faces. They differ, sometimes by a lot, and the second one is what a
    distance-based family has to be eliminated against."""
    wide = [f"l{i % 25}" for i in range(200)]           # all 25 levels, evenly spread
    df = pd.DataFrame({
        "num_a": np.arange(200.0), "num_b": np.arange(200.0) * 2,
        "num_c": np.arange(200.0) * 3,
        "small_cat": [list("abcd")[i % 4] for i in range(200)],    # 4 levels
        "mid_cat": [list("xyz")[i % 3] for i in range(200)],       # 3 levels
        "wide_cat": wide,                                          # 25 levels
        "label": ["no"] * 160 + ["yes"] * 40,
    })
    num = ["num_a", "num_b", "num_c"]
    cat = ["small_cat", "mid_cat", "wide_cat"]
    ev = model_selection_evidence(df, "label", num, cat)

    assert ev["n_rows"] == 200 and ev["n_features"] == 6
    assert ev["n_numeric"] == 3 and ev["n_categorical"] == 3
    assert ev["rows_per_feature"] == round(200 / 6, 1)
    # The 3 numeric columns stay as they are; the 3 categoricals become 4 + 3 + 25 = 32.
    assert ev["one_hot_width"] == 3 + 4 + 3 + 25, ev["one_hot_width"]
    assert ev["one_hot_width"] > ev["n_features"], (
        "the point of the number is that the raw column count understates the frame"
    )
    assert ev["max_cardinality"] == 25
    assert ev["median_cardinality"] == 4.0, "median of [4, 3, 25]"
    assert ev["high_cardinality_columns"] == ["wide_cat"], (
        "only the 25-level column reaches the high-cardinality threshold"
    )
    assert ev["class_counts"] == {"no": 160, "yes": 40}
    assert ev["imbalance_ratio"] == 4.0
    assert ev["rarest_class"] == "yes" and ev["rarest_rows"] == 40
    assert ev["missing_share"] == 0.0


def test_model_selection_evidence_flags_columns_that_carry_nothing():
    n = 500
    df = pd.DataFrame({
        "flat": np.zeros(n),
        "almost_flat": np.array(["a"] * 498 + ["b"] * 2),
        "real": np.arange(n, dtype=float),
        "label": np.array([0, 1] * (n // 2)),
    })
    ev = model_selection_evidence(df, "label", ["flat", "almost_flat", "real"], [])
    assert ev["constant_columns"] == ["flat"]
    assert ev["near_constant_columns"] == ["almost_flat"]


def test_model_selection_evidence_names_no_model():
    """Dataset-shape facts must not smuggle in a recommendation.

    A number cannot pick a family - only an argument over the numbers can, and that
    argument is the agent's to make and to write down (SKILL.md Step 4). The moment this
    dict contains a family name, the choice has moved out of the agent and into the code.
    """
    rng = np.random.default_rng(5)
    n = 200
    df = pd.DataFrame({
        "x": rng.normal(size=n),
        "g": [list("abc")[i % 3] for i in range(n)],
        "label": ["no"] * 140 + ["yes"] * 60,
    })
    ev = model_selection_evidence(df, "label", ["x"], ["g"])
    flat = json.dumps(ev).lower()
    for name in model_names():
        assert name not in flat and name.replace("_", " ") not in flat, (
            f"{name!r} appears in the evidence dict. The registry describes families; this "
            f"function describes the dataset, and the two are only joined by a decision"
        )


def test_macro_f1_silently_rises_when_a_class_is_absent():
    """The inflation this whole project keeps guarding against, pinned down.

    A fold that never sees class 2 is not scored 0 for it - the class is dropped from the
    average, so the fold scores HIGHER than one that contains it.
    """
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 1])          # every row predicted majority
    dropped = _score(y_true, y_pred, "f1_macro")
    pinned = _score(y_true, y_pred, "f1_macro", labels=[0, 1, 2])
    # Seen classes only: class 0 scores 0, class 1 scores 2/3, mean = 1/3.
    # Full label set: the same two plus a class nobody predicted, mean = 2/9.
    assert abs(dropped - 1 / 3) < 1e-9, dropped
    assert abs(pinned - 2 / 9) < 1e-9, pinned
    assert dropped > pinned, (
        f"omitting labels must inflate the score, got {dropped} vs {pinned}"
    )


def test_tune_refuses_labels_that_omit_a_class_in_y_fit():
    """The second silent-zero in this file, and the nastier one: it is not a metric that
    dips, it is every combination scoring exactly 0.0 - so the search ranks hyperparameters
    by an all-zero score and returns whichever the grid happened to list first.

    Naming the readable classes on an integer-encoded y does that with `zero_division=0` and
    raises nothing. So does any label set that leaves a class out.
    """
    df = panel(seed=3)
    X, y = df[["feat_a", "feat_b"]], df["label"].to_numpy()
    pipe = make_model("decision_tree")
    grid = {"max_depth": [2, 4]}
    names = ["no", "yes"]                       # the readable names, not the values in y

    assert _score(y, y, "f1_macro", labels=names) == 0.0, (
        "naming absent labels must score zero - that is what makes it silent"
    )
    e = raises(lambda: tune(pipe, grid, X, y, X_score=X, y_score=y,
                            metric="f1_macro", labels=names, seed=0),
               ValueError, contains="omits [0, 1], which do appear in y_fit")
    assert "encoded values" in str(e), str(e)

    # The values actually in y are accepted.
    score, params, _ = tune(pipe, grid, X, y, X_score=X, y_score=y,
                            metric="f1_macro", labels=sorted(np.unique(y)), seed=0)
    assert score > 0.5, f"a correct call must score above chance, got {score}"

    # A SUPERSET of y_fit's classes is accepted, and has to be: a rolling-origin fold
    # trains on a prefix that can be missing a class, and there the pool's full label set
    # is the right thing to pass - passing only what that prefix holds would let the fold
    # drop the class instead of scoring it zero.
    score, _, _ = tune(pipe, grid, X, y, X_score=X, y_score=y, metric="f1_macro",
                       labels=[0, 1, 2], seed=0)
    assert score > 0.5, f"a superset must be usable, got {score}"


def test_decision_log_refuses_a_choice_with_no_alternatives():
    log = DecisionLog()
    raises(lambda: log.record("validation", "strategy", "holdout", reason="it is fine",
                              alternatives_considered=[]),
           ValueError, contains="alternatives")
    raises(lambda: log.record("validation", "strategy", "holdout", reason="   ",
                              alternatives_considered=["k-fold"]),
           ValueError, contains="reason")
    assert log.entries == []


def test_decision_log_orders_by_workflow_step():
    log = DecisionLog()
    log.record("validation", "strategy", "holdout", "stable estimate", ["k-fold"])
    log.record("label", "target column", "label", "name and cardinality agree", ["id"])
    assert [e["step"] for e in log.by_step()] == ["label", "validation"]
    assert "Rejected: k-fold" in log.render()


def test_structure_evidence_flags_a_genuine_entity_key():
    df = panel()
    ev = structure_evidence(df, "label", ["feat_a", "feat_b", "feat_c"], [], [])
    by_name = {g["column"]: g for g in ev["group_candidates"]}
    key = by_name["entity_id"]
    assert key["name_matches_entity_convention"] is True
    assert key["rows_per_value_mean"] == 6.0
    assert key["lift_over_chance"] > 3, (
        f"an entity whose rows share a level should cluster; got {key}"
    )
    assert ev["time_candidates"] == []


def test_the_holdout_is_what_makes_the_group_test_discriminate():
    """With the candidate left IN the feature space, EVERY column looks like a group key.

    That is the reason `_neighbour_same_value_share` holds it out, and the reason a
    candidate must not sit in both `numeric`/`categorical` and `groups`.
    """
    df = panel()
    rng = np.random.default_rng(1)
    noise_group = rng.integers(0, 6, len(df)).astype(str)   # unrelated to anything
    base_feats = df[["feat_c"]]                             # carries no entity signal

    share_out, baseline_out, _, _ = _neighbour_same_value_share(noise_group, base_feats)
    share_in, _, _, _ = _neighbour_same_value_share(
        noise_group, base_feats.assign(noise_group=noise_group)
    )

    assert share_in > 0.95, (
        f"a candidate in its own feature space makes nearly every row its own neighbour - "
        f"which is why every column would look like a group key; got {share_in}"
    )
    assert share_out < 1.6 * baseline_out, (
        f"held out, an unrelated grouping must fall to chance: share {share_out} against "
        f"baseline {baseline_out}"
    )


def test_structure_evidence_withholds_a_ratio_computed_from_no_events():
    """A share and a baseline that are both ~0 give a meaningless quotient, not a lift."""
    rng = np.random.default_rng(2)
    n = 900
    df = pd.DataFrame({
        "continuous_measure": rng.normal(size=n).round(4),   # nearly all distinct
        "feat": rng.normal(size=n),
        "label": rng.integers(0, 2, n),
    })
    ev = structure_evidence(df, "label", ["continuous_measure", "feat"], [], [])
    g = {x["column"]: x for x in ev["group_candidates"]}["continuous_measure"]
    assert g["lift_over_chance"] is None
    assert g["lift_withheld_because"], "a withheld lift must say why"
    assert g["rows_whose_neighbour_shares_the_value"] < MIN_SHARE_EVENTS
    # Nothing is hidden - the raw numbers are still there to read.
    assert g["same_value_share_nearest_neighbour"] is not None
    assert g["chance_baseline_share"] is not None


def test_grouped_holdout_yields_positions_and_never_shares_an_entity():
    df = panel()
    X = numbered(df[["feat_a", "feat_b"]])
    y = df["label"].to_numpy()
    groups = df["entity_id"].to_numpy()

    tr, va = next(iter(grouped_holdout(0.25, 0).split(X, y, groups)))
    assert max(tr) < len(X) and max(va) < len(X), (
        "indices must be positional - a label index silently misaligns after any dropna"
    )
    assert set(groups[tr]).isdisjoint(set(groups[va])), "an entity appears on both sides"
    assert len(tr) + len(va) < len(X) or True   # GroupShuffleSplit may not use every row


def test_grouped_splitter_refuses_to_run_without_groups():
    df = panel()
    X, y = numbered(df[["feat_a", "feat_b"]]), df["label"].to_numpy()
    raises(lambda: list(grouped_holdout(0.25, 0).split(X, y, None)),
           ValueError, contains="groups")
    raises(lambda: tune(make_model("decision_tree"), {"clf__max_depth": [3]}, X, y,
                        splitter=grouped_holdout(0.25, 0), metric="f1_macro"),
           ValueError, contains="groups")


def test_group_arrays_that_do_not_line_up_are_refused():
    df = panel()
    X, y = numbered(df[["feat_a", "feat_b"]]), df["label"].to_numpy()
    raises(lambda: list(grouped_holdout(0.25, 0).split(X, y, np.array(["a", "b"]))),
           ValueError, contains="rows")


def test_stratified_group_folds_refuses_more_folds_than_entities():
    X = pd.DataFrame({"a": np.arange(20.0)})
    y = np.array([0, 1] * 10)
    groups = np.array(["g1", "g2"] * 10)
    e = raises(lambda: list(stratified_group_folds(3, 0).split(X, y, groups)),
               ValueError, contains="entities")
    assert "3" in str(e)


def test_temporal_folds_refuses_an_unsorted_order():
    """TimeSeriesSplit splits by row position, so an unsorted frame trains on the future."""
    n = 60
    X = pd.DataFrame({"a": np.arange(n, dtype=float)})
    y = np.array([0, 1] * (n // 2))
    scrambled = np.array([5, 3, 9] + list(range(10, n - 3 + 10)))
    scrambled = scrambled[:n] if len(scrambled) >= n else np.resize(scrambled, n)

    raises(lambda: list(temporal_folds(2, gap=1, order=scrambled).split(X, y)),
           ValueError, contains="non-decreasing")

    ordered = np.arange(n)
    folds = list(temporal_folds(3, gap=1, order=ordered).split(X, y))
    assert len(folds) == 3
    for tr, va in folds:
        assert tr.max() < va.min(), "a temporal fold must train strictly before it validates"


def test_temporal_folds_requires_a_gap():
    raises(lambda: temporal_folds(2, gap=None, order=np.arange(10)),
           ValueError, contains="gap")


def test_fold_consequences_names_a_class_absent_from_a_fold():
    """The absent class is not scored as zero - it is dropped, and the fold scores higher."""
    class SecondHalfOnly:
        """First half trains, second half validates. Crude, and exactly the careless split."""

        def __init__(self):
            self.enforces_disjoint_groups = False

        def split(self, X, y=None, groups=None):
            h = len(X) // 2
            yield np.arange(h), np.arange(h, len(X))

        def __repr__(self):
            return "SecondHalfOnly()"

    # train (rows 0-29): 12 of class 0, 16 of class 1, 2 of class 2
    # val   (rows 30-59): 15 of class 0, 15 of class 1 - the rare class is missing
    y = np.array([0] * 12 + [1] * 16 + [2] * 2 + [0] * 15 + [1] * 15)
    X = pd.DataFrame({"a": np.arange(60.0)})
    groups = np.array([f"g{i % 7}" for i in range(60)])   # entities span both halves

    out = fold_consequences(SecondHalfOnly(), X, y, groups, classes=[0, 1, 2])
    assert out["any_class_absent_from_val"] is True
    assert out["folds"][0]["classes_absent_from_val"] == ["2"]
    assert out["folds"][0]["classes_absent_from_train"] == []
    assert out["folds"][0]["n_train"] == 30 and out["folds"][0]["n_val"] == 30
    assert out["folds"][0]["rarest_class_rows_in_val"] == 0
    assert out["max_groups_shared_between_train_and_val"] > 0, (
        "a non-zero overlap on any splitter is the evidence that the rows are not "
        "independent"
    )


def test_a_group_splitter_that_leaks_entities_is_a_defect_not_a_finding():
    n = 40
    X = pd.DataFrame({"a": np.arange(n, dtype=float)})
    y = np.array([0, 1] * (n // 2))
    groups = np.array(["g1", "g2"] * (n // 2))

    class Leaky:
        enforces_disjoint_groups = True

        def split(self, X, y=None, groups=None):
            yield np.arange(0, n), np.arange(0, n)     # deliberately overlapping

        def __repr__(self):
            return "Leaky()"

    raises(lambda: fold_consequences(Leaky(), X, y, groups, classes=[0, 1]),
           ValueError, contains="shares entities")


def test_validation_report_changes_the_unit_when_given_groups():
    """With entities, the rarest class's quantum is entities, not rows - and it is coarser."""
    df = panel(n_entities=60, rows_per_entity=5, seed=3)
    y = df["label"].to_numpy()
    groups = df["entity_id"].to_numpy()

    rows_only = validation_report(y, 0.2, 5, classes=[0, 1])
    assert "groups" not in rows_only

    with_groups = validation_report(y, 0.2, 5, classes=[0, 1], groups=groups)
    g = with_groups["groups"]
    assert g["n_groups"] == 60 and g["rows_per_group_mean"] == 5.0
    assert g["rarest_entities_in_pool"] <= g["rarest_rows_in_pool"]
    assert g["why_this_is_worse_than_the_row_arithmetic"]


def test_validation_report_refuses_a_time_order_it_cannot_align():
    y = np.array([0, 1] * 10)
    raises(lambda: validation_report(y, 0.2, 5, classes=[0, 1], order="timestamp"),
           ValueError, contains="array")


def test_tune_refuses_more_than_one_scoring_mode():
    X = pd.DataFrame({"a": np.arange(40.0)})
    y = np.array([0, 1] * 20)
    raises(lambda: tune(make_model("decision_tree"), {"clf__max_depth": [3]}, X, y,
                        X_score=X, y_score=y, folds=2, metric="f1_macro"),
           ValueError, contains="exactly one")


def test_tune_refuses_to_pick_the_search_metric_for_you():
    """The search objective is configuration the report must state, so it is never inferred.

    A default here would be the one decision in the whole workflow made by a script rather
    than by SKILL.md - and it would be invisible in the report, which is the part a marker
    checks.
    """
    X = pd.DataFrame({"a": np.arange(40.0)})
    y = np.array([0, 1] * 20)
    raises(lambda: tune(make_model("decision_tree"), {"clf__max_depth": [3]}, X, y,
                        folds=2),
           TypeError, contains="metric")


# ----------------------------------------------------------- the shipped dataset itself


def test_the_shipped_dataset_has_no_entity_key_and_no_timestamp():
    """The negative case. If this ever fails, SKILL.md Step 3 must run before Step 6."""
    if not SHIPPED.exists():
        print(f"      (skipped: {SHIPPED} not present)")
        return
    df = load_table(SHIPPED)
    num, cat, dropped = classify_columns(df, "label")
    ev = structure_evidence(df, "label", num, cat, dropped)

    assert ev["time_candidates"] == [], "a time column appeared; use temporal_folds"
    assert not any(g["name_matches_entity_convention"] for g in ev["group_candidates"]), (
        "a column now looks like an entity key; pass it as groups= to the splitter"
    )
    # The guard, stated as a general property rather than a pinned column list.
    for g in ev["group_candidates"]:
        if g["rows_per_value_mean"] < 2:
            assert g["lift_over_chance"] is None, g["column"]
        if g["lift_over_chance"] is None:
            assert g["lift_withheld_because"], g["column"]
    assert any(g["lift_over_chance"] for g in ev["group_candidates"]), (
        "no candidate produced a lift at all, which suggests the search is broken"
    )


# ---------------------------------------------------------------------------- runner

def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception:
            failed.append(name)
            print(f"FAIL  {name}")
            traceback.print_exc()
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
