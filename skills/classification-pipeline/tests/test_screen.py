#!/usr/bin/env python
"""Checks for `screen.py` - the module that replaced an argument with a measurement.

    python tests/test_screen.py

Every test here pins one way the *old* Step 4 went wrong. It asked for each candidate to be
eliminated in prose, with a number cited; that rule checked the form of a reason and not its
substance, so a model could always reach the pair it had already picked. Five runs on four
datasets across three LLMs came back `logistic_regression` + `random_forest`, twice with
rejection reasons that measurement contradicts. The tests below pin the properties that make
that impossible to repeat:

  * the shortlist is arithmetic on measured scores, and a family under the floor is out;
  * a family that was never *measured* is never reported as if it had lost;
  * the test rows are gone before anything is fitted, so nothing is ranked on them;
  * the scaling flags the screen fits under agree with the registry's own prose.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

import screen as screen_mod
from models import MODEL_REGISTRY, model_names
from screen import (BOTH_METRICS, _excluded, _subsample, choose_protocol, run_screen)
from validation import make_splitter


# ---------------------------------------------------------------- tiny fixtures


def toy(n=90, p=3, seed=0) -> tuple[pd.DataFrame, np.ndarray]:
    """A frame small enough that seven families fit in a second, with signal in it."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(p)})
    y = ((X["x0"] + 0.5 * X["x1"] ** 2) > 0).astype(int).to_numpy()
    return X, y


def screen_toy(**over) -> dict:
    X, y = toy()
    kwargs = dict(numeric=list(X.columns), categorical=[], splitter=make_splitter(2, 42),
                  folds=2, seed=42, labels=[0, 1], sample=10_000)
    kwargs.update(over)
    return run_screen(X, y, **kwargs)


# ---------------------------------------------------------------- the shortlist


def test_the_shortlist_is_arithmetic_on_the_measured_means():
    """Pins: the floor is one std below the best mean - not a judgement, and not the top two.

    The old rule let a model argue any candidate out. Here, a family whose mean is under
    `best - best_std` cannot appear, and every family at or above it must.
    """
    ev = screen_toy()
    scored = {r["family"]: r for r in ev["families"] if r["n_folds"]}
    assert scored, "the screen scored nothing at all"
    for metric in BOTH_METRICS:
        best = max(scored.values(), key=lambda r: r["mean"][metric])
        floor = best["mean"][metric] - best["std"][metric]
        expected = sorted([f for f, r in scored.items() if r["mean"][metric] >= floor],
                          key=lambda f: -scored[f]["mean"][metric])
        assert ev["shortlist"][metric] == expected, metric
        for family in ev["shortlist"][metric]:
            assert scored[family]["mean"][metric] >= floor, family


def test_the_shortlist_holds_only_families_that_were_measured():
    """Pins: a family that errored or was cost-excluded cannot be on the shortlist.

    This is the whole point of keeping the three lists apart. If a family that never ran
    could reach the shortlist, "did not run" would silently become "good enough", which is
    the same class of mistake as the prose elimination it replaced.
    """
    ev = screen_toy()
    absent = {e["family"] for e in ev["excluded_without_fitting"]} | \
             {e["family"] for e in ev["errored"]}
    for metric in BOTH_METRICS:
        assert not (set(ev["shortlist"][metric]) & absent), metric


def test_every_family_lands_in_exactly_one_of_the_three_lists():
    """Pins: seven in, seven accounted for - none quietly dropped.

    A family that appears in no list would be invisible to Step 4, and the report would
    neither choose it nor explain leaving it out.
    """
    ev = screen_toy()
    seen = [r["family"] for r in ev["families"]] \
        + [e["family"] for e in ev["excluded_without_fitting"]] \
        + [e["family"] for e in ev["errored"]]
    assert sorted(seen) == model_names(), f"{sorted(seen)} != {model_names()}"
    assert len(seen) == len(set(seen)), "a family was reported twice"


def test_two_tree_ensembles_are_not_offered_as_a_pair():
    """Pins: spambase, where the shortlist was `random_forest` + `gradient_boosting`.

    Both reach the same axis-aligned staircase - one averages trees, the other fits them to
    residuals - so the pair tells the report nothing about where the boundary lies, which is
    the whole reason Step 4 asks for two biases. The file has to say the list is empty
    rather than leave a reader to believe two names make a pair.
    """
    ev = screen_toy()
    for metric in BOTH_METRICS:
        names = set(ev["shortlist"][metric])
        for a, b in ev["pairs_with_different_bias"][metric]:
            assert {a, b} <= names, "a pair was offered from outside the shortlist"
            assert ev["bias"][a] != ev["bias"][b], (metric, a, b)
        # The two tree ensembles are the case that matters: when both are shortlisted they
        # must not be offered together, however close their scores.
        if {"random_forest", "gradient_boosting"} <= names:
            assert ["gradient_boosting", "random_forest"] not in \
                ev["pairs_with_different_bias"][metric]
            assert ["random_forest", "gradient_boosting"] not in \
                ev["pairs_with_different_bias"][metric]


def test_the_bias_grouping_matches_the_registry():
    """Pins: the report's claim about inductive bias and the screen's check on it agree."""
    from models import MODEL_REGISTRY
    for name in model_names():
        assert MODEL_REGISTRY[name]["inductive_bias"], name
    trees = {MODEL_REGISTRY[n]["inductive_bias"]
             for n in ("decision_tree", "random_forest", "gradient_boosting")}
    assert len(trees) == 1, "the three tree families must share one bias group"
    assert MODEL_REGISTRY["logistic_regression"]["inductive_bias"] not in trees
    assert MODEL_REGISTRY["svm"]["inductive_bias"] not in trees


def test_both_metrics_are_scored_on_the_same_folds():
    """Pins: whichever metric Step 7 picks, its shortlist already exists.

    Scoring one metric would force the metric choice at screen time - before Step 7 has
    read the dataset - which is a decision made in the wrong place for the wrong reason.
    """
    ev = screen_toy()
    assert set(ev["shortlist"]) == set(BOTH_METRICS)
    for r in ev["families"]:
        assert set(r["mean"]) == set(BOTH_METRICS)
        assert len(r["per_fold"]["f1_macro"]) == len(r["per_fold"]["accuracy"]) == r["n_folds"]


# ---------------------------------------------------------------- not-measured


def test_a_family_whose_fit_raises_is_recorded_not_swallowed(monkeypatch=None):
    """Pins: an exception in one family leaves the other six measured, and is visible.

    Found in the wild before it was found in a test - the Anaconda here shipped
    `threadpoolctl` 2.2.0, which cannot read this MKL's version, so sklearn's brute-force
    neighbour search raised and KNN died on
    every dataset wide enough for `algorithm="auto"` to pick brute. Uncaught, that took the
    entire screen down. Swallowed without a record, it would have read as "KNN lost" - and
    that is the one thing it must never read as.
    """
    real = screen_mod.SCALING_REQUIRED
    broken = dict(real)
    broken["not_a_real_family"] = False      # `make_model` refuses this name, and says so
    screen_mod.SCALING_REQUIRED = broken
    try:
        ev = screen_toy()
    finally:
        screen_mod.SCALING_REQUIRED = real

    assert [e["family"] for e in ev["errored"]] == ["not_a_real_family"]
    err = ev["errored"][0]
    assert "unknown model" in err["error"], err["error"]
    assert "not scored" in err["note"]
    assert "not_a_real_family" not in [r["family"] for r in ev["families"]]
    assert len(ev["families"]) == len(real), "one bad family cost the others their screen"


def test_a_missing_reason_never_reads_as_a_lost_one():
    """Pins: the two absence lists say *why*, so neither can be written up as a defeat."""
    X, y = toy()
    ev = run_screen(X, y, numeric=list(X.columns), categorical=[],
                    splitter=make_splitter(2, 42), folds=2, seed=42, labels=[0, 1],
                    sample=10_000, one_hot_width=10_000_000)
    assert [e["family"] for e in ev["excluded_without_fitting"]] == ["knn"], \
        "the cell bound should have caught knn and nothing else"
    reason = ev["excluded_without_fitting"][0]["reason"]
    assert "not screened" in reason and "cell bound" in reason


# ---------------------------------------------------------------- cost bounds


def test_the_cost_bounds_are_the_only_grounds_for_excluding_without_fitting():
    """Pins: nothing may leave the screen for a reason that is about difficulty.

    "SVM is probably noisy at this size" is exactly the guess this module exists to replace
    with a number, and it is the guess that was wrong twice - on spambase and on ionosphere,
    where SVM was rejected in prose and came first by measurement.
    """
    # Under both bounds, nothing is excluded - every family is fitted.
    assert _excluded("svm", 14_999, 1_000) is None
    assert _excluded("knn", 1_000, 5_999) is None
    # Over one, exactly that family is, and the reason names the bound.
    assert "15,000 row bound" in _excluded("svm", 15_001, 10)
    assert "cell bound" in _excluded("knn", 1_000, 6_001)
    # Families with no bound are never excluded here, however large the frame.
    for family in ("decision_tree", "random_forest", "gradient_boosting",
                   "logistic_regression", "naive_bayes"):
        assert _excluded(family, 10 ** 7, 10 ** 4) is None, family


# ---------------------------------------------------------------- protocol


def test_an_explicit_group_or_time_column_overrides_what_structure_json_inferred():
    """Pins: the caller wins - the search cannot see what a person knows.

    `structure.json` reports candidates with evidence, not verdicts. A dataset whose entity
    column exists but whose lift fell just under the threshold is a real case, and passing
    the column must settle it rather than be re-argued by the inference.
    """
    structure = {"time_candidates": [{"column": "when", "is_non_decreasing": True}],
                 "group_candidates": [{"column": "who", "rows_per_value_mean": 9.0,
                                       "lift_over_chance": 40.0}]}
    assert choose_protocol(structure)["unit"] == "time"
    assert choose_protocol(structure, group_column="patient")["strategy"] == \
        "stratified_group_kfold"
    assert choose_protocol(structure, group_column="patient")["group_column"] == "patient"
    assert choose_protocol(structure, time_column="ts")["time_column"] == "ts"


def test_a_strong_predictor_is_not_mistaken_for_an_entity_key():
    """Pins: repeating values alone do not make a column a group.

    A binary column repeats (so `rows_per_value_mean` is 2) and a predictive one makes rows
    that share it resemble each other - which is the same *shape* of evidence an entity key
    gives. Grouping on it would cut folds through the classes instead of through the
    entities. The lift threshold is what separates them.
    """
    weak = {"group_candidates": [{"column": "is_senior", "rows_per_value_mean": 2.0,
                                  "lift_over_chance": 1.9}]}
    assert choose_protocol(weak)["unit"] == "row"

    unique = {"group_candidates": [{"column": "row_id", "rows_per_value_mean": 1.0,
                                    "lift_over_chance": 900.0}]}
    assert choose_protocol(unique)["unit"] == "row"


def test_a_column_that_is_also_a_feature_is_not_an_entity_key():
    """Pins: spambase, where a count column passed every other test and grouped the folds.

    `capital_run_length_total` has 919 distinct values over 4,601 rows, 5.01 rows per value
    and a lift of 46.6 - it looks exactly like an entity key by the repeat and lift rules.
    It is not one: it is a column the model is given, and its lift comes from *predicting
    the label*, not from any entity. Folds cut on it are cut on a predictor, which is a
    different experiment from the one the report describes.

    the real mice case is the control - `mouse_id` passes this test as well as the others.
    """
    looks_like_a_key = {"group_candidates": [
        {"column": "capital_run_length_total", "rows_per_value_mean": 5.01,
         "lift_over_chance": 46.57, "is_in_feature_set": True},
    ]}
    p = choose_protocol(looks_like_a_key)
    assert p["unit"] == "row"
    assert "capital_run_length_total" in p["why"], \
        "the rejection must name the candidate, or nobody can override it"
    assert "--group-column" in p["why"] or "features" in p["why"]

    real_key = {"group_candidates": [
        {"column": "mouse_id", "rows_per_value_mean": 15.0,
         "lift_over_chance": 76.64, "is_in_feature_set": False},
    ]}
    p = choose_protocol(real_key)
    assert p["unit"] == "entity" and p["group_column"] == "mouse_id"


def test_no_candidate_at_all_falls_back_to_independent_rows():
    """Pins: the default is the honest one, and it is *stated*, not assumed."""
    p = choose_protocol({})
    assert (p["unit"], p["strategy"]) == ("row", "stratified_kfold")
    assert p["why"], "the protocol must carry the reason it was chosen"


# ---------------------------------------------------------------- subsampling


def test_a_subsample_keeps_whole_entities_and_every_class():
    """Pins: the recorded protocol is not a lie about the rows actually used.

    Cutting rows out of an entity would leave its remainder more alike than a real fold ever
    sees. That flatters every family equally, so it would not change the ranking - but it
    would misdescribe what was run, and a protocol that misdescribes itself is the kind of
    wrongness that raises nothing.
    """
    groups = np.repeat(np.arange(60), 5)          # 60 entities of 5 rows
    y = np.tile([0, 1, 2, 3, 4], 60)
    keep = _subsample(groups, y, cap=100, seed=0)

    kept_groups = groups[keep]
    for g in np.unique(kept_groups):
        assert (groups == g).sum() == (kept_groups == g).sum(), \
            f"entity {g} was split across the subsample boundary"
    assert set(np.unique(y[keep])) == {0, 1, 2, 3, 4}, "a class fell out of the subsample"
    assert len(keep) < len(y)


def test_a_frame_under_the_cap_is_used_whole():
    """Pins: no subsample when none is needed - the screen fits on everything it was given."""
    groups = np.repeat(np.arange(10), 3)
    y = np.tile([0, 1, 2], 10)
    keep = _subsample(groups, y, cap=100, seed=0)
    assert len(keep) == len(y)


# ---------------------------------------------------------------- the registry


def test_the_scaling_flags_cover_exactly_the_registry():
    """Pins: a family added to the registry cannot be silently missing from the screen.

    `run_screen` iterates `SCALING_REQUIRED`, not the registry, so a family absent from it
    would never be fitted and never appear in any list - invisible to Step 4 rather than
    reported as unmeasured.
    """
    assert sorted(screen_mod.SCALING_REQUIRED) == model_names()


def test_the_scaling_flags_agree_with_the_registry_prose():
    """Pins: two copies of one claim cannot drift.

    The registry states scaling as prose with a reason, because "generally recommended" and
    "effectively required" are different claims. The screen needs a boolean. A boolean that
    quietly disagreed with the prose would make the screen fit a family under the wrong
    preprocessing - and it would raise nothing, it would just score it differently.
    """
    for name, spec in MODEL_REGISTRY.items():
        says_not_required = spec["scaling"].startswith("not required")
        assert screen_mod.SCALING_REQUIRED[name] is not says_not_required, (
            f"{name}: registry says {spec['scaling']!r} but the flag is "
            f"{screen_mod.SCALING_REQUIRED[name]}"
        )


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
