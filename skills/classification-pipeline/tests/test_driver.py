#!/usr/bin/env python
"""Checks for `driver.py` - the execution path itself.

    python tests/test_driver.py

`test_guards.py` pins the guards inside the library and `test_plan.py` pins the plan
contract. This file pins the thing that was actually missing: a path the workflow cannot
leave. Every test here stands in for a way a run *looks* finished without being finished.

  * A report left over from an earlier run, sitting in `out/` while the plan that would
    replace it is refused - the folder reads as "done" and the numbers are from a plan
    nobody is looking at any more. So the stale report is deleted **first**, before the
    plan is even parsed.
  * Evidence staged anywhere but `out/evidence/` - `make_pdf.require_evidence` resolves
    that one path, so evidence written elsewhere is evidence the PDF step never finds.
  * A `prepare` run under one interpreter and a `run` under another, fitting different
    library defaults than the evidence describes.

The last test runs `run` end to end on a small synthetic frame and then hands the result
to the real `require_evidence`, so the figure, the markdown and the evidence are checked
together rather than each in isolation.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import driver
from make_pdf import parse, require_evidence


# --------------------------------------------------------------------------- fixtures


def fixture_csv(folder: Path, name: str = "train.csv", n: int = 240, seed: int = 0) -> Path:
    """A small frame with a learnable signal, three columns and two balanced classes.

    Small on purpose: the end-to-end test tunes two models on it, and a test that takes
    ten seconds is a test that stops being run.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    region = rng.choice(["north", "south", "east"], size=n)
    signal = 1.5 * a + 1.0 * b + 1.2 * (region == "north")
    label = np.where(signal > np.median(signal), "yes", "no")
    path = Path(folder) / name
    pd.DataFrame({"a": a.round(3), "b": b.round(3), "region": region,
                  "label": label}).to_csv(path, index=False)
    return path


def valid_plan(dataset: Path, **over) -> dict:
    """A complete plan for the fixture frame. `over` applies dotted-path overrides.

    Dotted paths rather than whole blocks, so a test that changes one field still exercises
    a plan that is valid everywhere else - which is the only way to know the refusal it
    pins is caused by the field it changed.
    """
    p = {
        "schema_version": 1,
        "dataset": str(dataset).replace("\\", "/"),
        "test_dataset": None,
        "seed": 42,
        "target": {
            "column": "label",
            "reason": "`label` is the only column holding a class name; `a`, `b` and "
                      "`region` are measurements.",
            "alternatives_considered": ["`region` - three levels, and its levels are "
                                        "attributes of a row rather than its outcome"],
        },
        "cleaning": {
            "drop_columns": [],
            "drop_rows_with_missing_label": True,
            "rarest_class_action": "keep",
            "outlier_policy": {"method": "keep"},
            "reason": "the fixture frame has no missing cells and no column that needs "
                      "holding out.",
            "alternatives_considered": ["an IQR fence - there is nothing to remove from a "
                                        "frame drawn from a normal distribution"],
        },
        "models": [
            {
                "family": "logistic_regression",
                "scale": True,
                "scale_reason": "required - the solver penalises the coefficients, so an "
                                "unscaled column carries more weight than its importance.",
                "stopping": "a fixed iteration cap on the solver.",
                "overfitting_control": "an L2 penalty whose strength the search chooses.",
                "grid": {"C": [1.0]},
                "reason": "a linear boundary is the low-variance end of the comparison.",
                "alternatives_considered": ["naive_bayes - its independence assumption is "
                                            "false here, `a` and `b` are correlated"],
            },
            {
                # Deliberately a family with a bootstrap: a fixture built from two
                # effectively-deterministic families let the determinism test below pass
                # while `seed` was reaching nothing at all.
                "family": "random_forest",
                "scale": False,
                "scale_reason": "not required - a tree splits on order within a column, "
                                "and multiplying a column by a constant does not move a "
                                "split.",
                "stopping": "a depth cap, so no leaf is grown to hold a single row.",
                "overfitting_control": "the depth cap and the bootstrap resample, so no "
                                       "single tree can memorise the frame.",
                "grid": {"max_depth": [2], "n_estimators": [10]},
                "reason": "threshold splits averaged over resamples are a different "
                          "inductive bias from a weighted sum, which is the point of "
                          "comparing two families.",
                "alternatives_considered": ["decision_tree - the same bias with far more "
                                            "variance, which at 240 rows the fixture "
                                            "cannot distinguish"],
            },
        ],
        "validation": {
            "unit": "row",
            "strategy": "stratified_kfold",
            "test_size": 0.25,
            "val_fraction": 0.25,
            "n_splits": 3,
            "gap": None,
            "group_column": None,
            "time_column": None,
            "reason": "no column repeats an entity and none is a date, so rows are the "
                      "unit and order carries nothing.",
            "alternatives_considered": ["a single holdout - one estimate instead of three, "
                                        "and one split can land unluckily"],
        },
        "tuning": {
            "metric": "f1_macro",
            "final_fit_scope": "pool",
            "final_fit_reason": "every row of the pool is validated against exactly once "
                                "under k-fold, so the final fit takes all of them.",
            "reason": "macro-F1 weights both classes equally, which is what the task asks.",
            "alternatives_considered": ["accuracy - a constant prediction already scores "
                                        "about half on a balanced frame"],
        },
        "primary_metric": {
            "name": "f1_macro",
            "rejected": "accuracy",
            "reason": "the report leads with macro-F1 because the two classes are treated "
                      "as equally important.",
            "rejected_reason": "accuracy is in the table, but leading with it would let a "
                               "majority-class prediction look like a working model.",
        },
        "report": {
            "student_id": "G0000000A",
            "student_name": "DRIVER TEST FIXTURE",
            "variant": "Variant-2",
            "model_version": "driver.py @ schema 1 (test fixture)",
            "harness": "pytest",
            "repository": "https://github.com/example/fixture",
            "dataset_label": "synthetic fixture - 240 rows",
        },
        "narrative": {
            "candidate_models": "Both families were kept small so the test is quick; the "
                                "argument is the same one the real plan makes.",
            "feature_engineering": "No features were constructed.",
            "cleaning_why": [{"issue": "no missing cells", "action": "nothing",
                              "why": "there is nothing to impute or drop"}],
            "validation_argument": "rows are the unit, so the split is stratified.",
            "primary_metric_argument": "macro-F1, because the classes are equally "
                                       "important.",
            "performance_comparison": "Both models were scored on the same held-out rows.",
            "limitations": "The fixture frame is synthetic and supports no claim about "
                           "anything.",
        },
    }
    for path, value in over.items():
        node = p
        parts = path.split(".")
        for part in parts[:-1]:
            node = node[int(part)] if isinstance(node, list) else node[part]
        last = parts[-1]
        if isinstance(node, list):
            node[int(last)] = value
        else:
            node[last] = value
    return p


def write_plan(experiment: Path, plan: dict) -> Path:
    path = Path(experiment) / "plan.json"
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return path


def experiment_with_plan(tmp: str, **over) -> tuple[Path, Path]:
    """A folder holding a `train.csv` and a `plan.json` for it. Returns (folder, csv)."""
    folder = Path(tmp) / "experiment"
    folder.mkdir()
    csv = fixture_csv(folder)
    write_plan(folder, valid_plan(csv, **over))
    return folder, csv


def run(experiment: Path) -> int:
    return driver.main(["run", "--experiment", str(experiment)])


class cwd:
    """Run a block with the process in `path`, then put it back."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.old = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, *exc):
        os.chdir(self.old)


def prepare(experiment: Path, csv: Path, *extra: str) -> int:
    return driver.main(["prepare", "--dataset", str(csv), "--experiment",
                        str(experiment), *extra])


def evidence(folder: Path) -> Path:
    return Path(folder) / "out" / "evidence"


def metrics_path(folder: Path) -> Path:
    """`metrics.json` lives under `out/evidence/`, not in `out/`. A test asserting on
    `out/metrics.json` passes whatever the run does, because that path never exists -
    which is how three assertions in this file were first written."""
    return evidence(folder) / "metrics.json"
    """`metrics.json` lives under `out/evidence/`, not in `out/`. A test asserting on
    `out/metrics.json` passes whatever the run does, because that path never exists -
    which is how three assertions in this file were first written."""
    return evidence(folder) / "metrics.json"


# ---------------------------------------------------------------------- the two commands


def test_prepare_writes_the_profile_evidence_once_it_knows_the_label():
    """Step 0's whole purpose: the model reads evidence rather than guessing at a frame."""
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d) / "experiment"
        folder.mkdir()
        csv = fixture_csv(folder)
        assert prepare(folder, csv, "--label", "label") == 0
        ev = evidence(folder)
        for name in ("candidates.json", "eda.json", "model_selection.json",
                     "structure.json", "rarest_class.json", "environment.json"):
            assert (ev / name).exists(), f"prepare did not write {name}"
        eda = json.loads((ev / "eda.json").read_text(encoding="utf-8"))
        assert eda["label_column"] == "label", eda
        assert eda["_evidence"]["producer"], eda["_evidence"]


def test_prepare_without_a_label_writes_the_candidates_and_says_what_is_missing():
    """Step 1 asks the model to choose a label *from evidence*. With no label it can still
    be told which columns are candidates - but the profile of the target cannot exist yet,
    and a placeholder written here would be indistinguishable from a real one."""
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d) / "experiment"
        folder.mkdir()
        csv = fixture_csv(folder)
        assert prepare(folder, csv) == 0
        ev = evidence(folder)
        assert (ev / "candidates.json").exists()
        assert not (ev / "eda.json").exists(), "a profile was written with no label given"


def test_prepare_refuses_a_label_that_is_not_a_column():
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d) / "experiment"
        folder.mkdir()
        csv = fixture_csv(folder)
        try:
            prepare(folder, csv, "--label", "nope")
        except driver.DriverError as e:
            assert "not a column of" in str(e), str(e)
            return
        raise AssertionError("expected DriverError")


def test_prepare_refuses_an_experiment_inside_a_dot_opencode_tree():
    """The failure that started this: cwd was `.opencode/`, so `out/` landed outside the
    folder being submitted, where nothing looks wrong."""
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d) / ".opencode"
        folder.mkdir()
        csv = fixture_csv(Path(d))
        try:
            prepare(folder, csv, "--label", "label")
        except SystemExit as e:
            assert ".opencode" in str(e), str(e)
            return
        raise AssertionError("expected SystemExit")


def test_run_refuses_an_experiment_inside_the_skill_directory():
    try:
        run(ROOT / "out")
    except SystemExit as e:
        assert "skill directory" in str(e).lower(), str(e)
        return
    raise AssertionError("expected SystemExit")


def test_a_run_started_inside_dot_opencode_still_writes_to_the_experiment_folder():
    """The failure this whole design exists to close, played out.

    The harness had `cd`-ed into `.opencode/`, and the output folder was computed from the
    working directory - so it resolved to `.opencode/out/`: outside the folder being
    submitted, outside the skill directory, and indistinguishable from success. Here the
    process really is inside `.opencode/` when `run` is called, and the output has to land
    in the experiment folder anyway, because `--experiment` names it and the working
    directory is never consulted."""
    with tempfile.TemporaryDirectory() as d:
        experiment, _ = experiment_with_plan(d)
        harness = experiment / ".opencode"
        harness.mkdir()
        with cwd(harness):
            assert run(experiment) == 0
        assert (experiment / "out" / "report.pdf").exists(), \
            "the report did not land in the experiment folder"
        assert not (harness / "out").exists(), \
            "output landed inside .opencode - the escape this design exists to close"


def test_run_refuses_a_missing_plan_and_does_no_work():
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d) / "experiment"
        folder.mkdir()
        assert run(folder) == 1
        assert not (folder / "out" / "report.md").exists()


# ------------------------------------------------------------------- what a failure leaves


def test_a_failed_run_leaves_no_report_and_removes_a_stale_one():
    """The invariant that makes the folder readable: `report.pdf` is there if and only if a
    run reached the end. Here the run dies at the loader - long after the stale report was
    cleared - and both stale files are gone."""
    with tempfile.TemporaryDirectory() as d:
        folder, _ = experiment_with_plan(d, **{"cleaning.drop_columns": ["not_a_column"]})
        out = folder / "out"
        out.mkdir()
        (out / "report.md").write_text("# an earlier run\n", encoding="utf-8")
        (out / "report.pdf").write_bytes(b"%PDF-1.4 stale")
        assert run(folder) == 1
        assert not (out / "report.md").exists(), "a stale report.md survived a failed run"
        assert not (out / "report.pdf").exists(), "a stale report.pdf survived a failed run"
        assert not (out / "metrics.json").exists(), \
            "a failed run left metrics.json - a result-shaped file from a run that stopped"


def test_a_refused_plan_also_removes_the_previous_report():
    """Clearing happens *before* the plan is parsed. Otherwise a typo in plan.json is
    refused and the folder still holds the last run's PDF, which reads as current."""
    with tempfile.TemporaryDirectory() as d:
        folder, _ = experiment_with_plan(d)
        out = folder / "out"
        out.mkdir()
        (out / "report.pdf").write_bytes(b"%PDF-1.4 stale")
        write_plan(folder, valid_plan(folder / "train.csv", **{"models": [
            {"family": "decision_tree", "scale": False, "scale_reason": "x",
             "stopping": "y", "overfitting_control": "z", "grid": {"max_depth": [2]},
             "reason": "r", "alternatives_considered": ["a"]}]}))
        assert run(folder) == 1
        assert not (out / "report.pdf").exists(), \
            "a refused plan left the previous run's report looking current"


def test_run_refuses_a_plan_with_one_model_before_any_work():
    """`evaluate_pair` refuses one model too, but only after both tuning runs. The plan is
    the cheap place to say it."""
    with tempfile.TemporaryDirectory() as d:
        folder, _ = experiment_with_plan(d, **{"models.1": {
            "family": "logistic_regression", "scale": True, "scale_reason": "x",
            "stopping": "y", "overfitting_control": "z", "grid": {"C": [1.0]},
            "reason": "r", "alternatives_considered": ["a"]}})
        assert run(folder) == 1
        assert not metrics_path(folder).exists()


def test_run_refuses_an_interpreter_that_is_not_the_one_prepare_ran_under():
    """Evidence describes a frame read, and library versions resolved, in one environment.
    Executing in another fits different defaults and the report never says so."""
    with tempfile.TemporaryDirectory() as d:
        folder, _ = experiment_with_plan(d)
        ev = evidence(folder)
        ev.mkdir(parents=True)
        (ev / "environment.json").write_text(json.dumps({
            "_evidence": {"type": "environment", "schema_version": 1,
                          "producer": "driver"},
            "interpreter": "D:/elsewhere/python.exe",
            "interpreter_real": "D:/elsewhere/python.exe",
        }), encoding="utf-8")
        assert run(folder) == 1
        assert not metrics_path(folder).exists()

        # ...and the escape hatch turns exactly this off.
        os.environ[driver.ALLOW_INTERPRETER_ENV] = "1"
        try:
            assert run(folder) == 0
        finally:
            os.environ.pop(driver.ALLOW_INTERPRETER_ENV, None)


def test_run_refuses_more_folds_than_the_rarest_class_can_fill():
    """`fold_cap` again, but reached through the plan rather than called directly: the
    refusal has to be the run's, and it has to happen before anything is fitted."""
    with tempfile.TemporaryDirectory() as d:
        folder, _ = experiment_with_plan(d, **{"validation.n_splits": 500})
        assert run(folder) == 1
        assert not metrics_path(folder).exists()


def test_run_refuses_a_test_file_without_the_label():
    """A test file with no outcome column cannot score anything, and `evaluate` would
    silently produce numbers from an empty comparison."""
    with tempfile.TemporaryDirectory() as d:
        folder, csv = experiment_with_plan(d)
        holder = pd.read_csv(csv).drop(columns=["label"])
        blind = Path(d) / "blind.csv"
        holder.to_csv(blind, index=False)
        write_plan(folder, valid_plan(csv, **{"test_dataset": str(blind).replace("\\", "/")}))
        assert run(folder) == 1
        assert not metrics_path(folder).exists()


def test_run_refuses_a_test_file_carrying_a_class_the_training_frame_never_saw():
    with tempfile.TemporaryDirectory() as d:
        folder, csv = experiment_with_plan(d)
        other = pd.read_csv(csv)
        other.loc[0, "label"] = "maybe"
        path = Path(d) / "other.csv"
        other.to_csv(path, index=False)
        write_plan(folder, valid_plan(csv, **{"test_dataset": str(path).replace("\\", "/")}))
        assert run(folder) == 1
        assert not metrics_path(folder).exists()


# ------------------------------------------------------------------- the whole thing


def test_run_end_to_end_produces_a_pdf_the_evidence_guard_accepts():
    """The one test that runs the path the workflow actually takes, and then asks the real
    guard - not a copy of its rules - whether what came out is acceptable."""
    with tempfile.TemporaryDirectory() as d:
        folder, _ = experiment_with_plan(d)
        assert run(folder) == 0
        out = folder / "out"
        md, pdf = out / "report.md", out / "report.pdf"

        assert md.exists() and pdf.exists(), "the run reported success without a report"
        assert pdf.read_bytes()[:4] == b"%PDF", "report.pdf is not a PDF"

        require_evidence(md)                # must not raise

        for name in ("metrics.json", "decisions.json", "validation.json", "tuning.json",
                     "cleaning.json", "eda.json"):
            assert (evidence(folder) / name).exists(), f"no {name}"
        metrics = json.loads(metrics_path(folder).read_text(encoding="utf-8"))
        # `evaluate_pair`'s shape: one entry per family, keyed by family name.
        assert set(metrics) == {"logistic_regression", "random_forest"}, sorted(metrics)
        for family, m in metrics.items():
            assert 0.0 <= m["f1_macro"] <= 1.0, (family, m)
            assert m["n_scored"] == 60, (family, m["n_scored"])
        # The fixture's signal is learnable, so a model that cannot beat chance means the
        # path from label encoding to scoring is broken somewhere in the middle.
        assert max(m["accuracy"] for m in metrics.values()) > 0.7, metrics

        text = md.read_text(encoding="utf-8")
        assert "DRIVER TEST FIXTURE" in text, "the plan's report block never reached the page"
        assert "logistic_regression" in text and "random_forest" in text

        # The figure has to reach the renderer as a figure. This is the assertion that was
        # missing while every report this skill produced printed its confusion matrix as the
        # literal line `Confusion matrices: ![...](figures/confusion_pair.png)`: the run
        # exited 0, the PDF passed the evidence guard, and no matrix was on the page.
        assert (out / "figures" / "confusion_pair.png").exists(), "no confusion pair drawn"
        blocks = parse(text)
        assert [p for k, p in blocks if k == "image"] == ["figures/confusion_pair.png"], (
            [(k, p) for k, p in blocks if k == "image"]
        )
        assert not [p for k, p in blocks if isinstance(p, str) and "![" in p], (
            "image syntax reached the page as text instead of as a figure"
        )


def test_the_plan_seed_reaches_every_family_that_has_somewhere_to_put_it():
    """The bug this pins was live: every family is built with scikit-learn's defaults, none
    of which sets `random_state`, so a plan declaring `"seed": 42` printed that seed in the
    report and nothing honoured it. Two runs on the real dataset gave `random_forest`
    macro-F1 0.7520 and 0.7538.

    Checked across the whole registry rather than the two families the fixture happens to
    use, so a family added later is covered without an edit here."""
    from models import make_model, model_names

    for family in model_names():
        family_has_one = "random_state" in make_model(family).get_params()
        seeded = driver._seeded(family, 7)
        if family_has_one:
            assert seeded.get_params()["random_state"] == 7, \
                f"{family} takes a random_state and the plan's seed did not reach it"
        else:
            assert "random_state" not in seeded.get_params(), family


def test_two_identical_runs_produce_byte_identical_metrics():
    """The seed in the plan has to be the seed that decides everything. If a run is not
    reproducible, a number in the report cannot be checked against anything.

    The fixture's second family is a forest on purpose: with two deterministic families
    this test passed while the seed was reaching no estimator at all.
    """
    with tempfile.TemporaryDirectory() as d:
        folder, _ = experiment_with_plan(d)
        assert run(folder) == 0
        first = metrics_path(folder).read_bytes()
        assert run(folder) == 0
        assert metrics_path(folder).read_bytes() == first


def test_a_plan_naming_a_test_file_uses_it_instead_of_splitting_one_off():
    with tempfile.TemporaryDirectory() as d:
        folder, csv = experiment_with_plan(d)
        frame = pd.read_csv(csv)
        test_path = Path(d) / "test.csv"
        frame.iloc[:60].to_csv(test_path, index=False)
        frame.iloc[60:].to_csv(csv, index=False)
        write_plan(folder, valid_plan(csv,
                                      **{"test_dataset": str(test_path).replace("\\", "/")}))
        assert run(folder) == 0
        # The report has to say where its test rows came from. A run that quietly ignored
        # the file and split the frame instead produces the same table and the same page.
        text = (folder / "out" / "report.md").read_text(encoding="utf-8")
        assert "`test.csv`" in text, text
        assert "held out from the start" in text, text
        metrics = json.loads(metrics_path(folder).read_text(encoding="utf-8"))
        assert all(m["n_scored"] == 60 for m in metrics.values()), metrics


def test_a_pair_outside_the_screens_shortlist_is_announced_not_refused():
    """Pins: the divergence is allowed and *visible*, which is what the old flow lacked.

    Five runs on four datasets came back `logistic_regression` + `random_forest`, twice with
    rejection reasons the measurements contradicted - and nothing anywhere recorded that the
    pair had not come from the evidence. A refusal would be wrong (a person may know
    something the screen cannot), and silence is what let it happen. So: a warning, naming
    the families and the shortlist they are missing from.
    """
    import contextlib
    import io

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        ev = folder / "evidence"
        ev.mkdir()
        (ev / "candidate_screen.json").write_text(json.dumps({
            "_evidence": {"type": "candidate_screen", "schema_version": 1,
                          "producer": "screen"},
            "families": [], "excluded_without_fitting": [], "errored": [],
            "shortlist": {"f1_macro": ["svm", "random_forest"], "accuracy": ["svm"]},
        }), encoding="utf-8")

        def captured(plan) -> str:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                driver._check_pair_against_screen(plan, ev)
            return buf.getvalue()

        # On the shortlist: nothing to say.
        quiet = captured({"models": [{"family": "svm"}, {"family": "random_forest"}]})
        assert "WARNING" not in quiet, quiet

        # Chosen anyway, and not measured either way: allowed, and said out loud.
        loud = captured({"models": [{"family": "logistic_regression"},
                                    {"family": "random_forest"}]})
        assert "logistic_regression" in loud and "WARNING" in loud, loud

        # A family that was never measured is called *that*, not "eliminated".
        (ev / "candidate_screen.json").write_text(json.dumps({
            "shortlist": {"f1_macro": [], "accuracy": []},
            "excluded_without_fitting": [], "families": [],
            "errored": [{"family": "knn", "error": "AttributeError: ...",
                         "note": "not scored"}],
        }), encoding="utf-8")
        never = captured({"models": [{"family": "knn"}, {"family": "svm"}]})
        assert "not measured" in never, never
        assert "eliminated" in never, never

        # No screen at all - the run-without-prepare path - says nothing rather than
        # inventing a shortlist to disagree with.
        (ev / "candidate_screen.json").unlink()
        assert captured({"models": [{"family": "svm"}, {"family": "knn"}]}).strip() == ""


# ------------------------------------------------------------- a label with missing values


def test_encode_refuses_a_missing_label_rather_than_casting_it():
    """Pins: the silent half of the bug, which is the dangerous half.

    `y.astype(str)` makes a missing value the *string* `"nan"`, which is in no mapping, so
    `map` leaves NaN and `to_numpy(dtype=int)` casts it to whatever the hardware produces -
    large and negative in practice, and the cast warns rather than raises. Downstream that
    surfaced as `np.bincount` saying `'list' argument must have no negative elements`, which
    names neither the column nor the cause. On a frame where those NaNs happened to cast
    non-negative, nothing would have raised at all and the screen would have ranked families
    on a class that does not exist.
    """
    y = pd.Series(["no", "yes", None, "yes", None])
    try:
        driver._encode(y)
    except driver.DriverError as e:
        assert "2 row(s) have no label" in str(e), e
        assert "drop_rows_with_missing_label" in str(e), e
    else:
        raise AssertionError("a missing label was encoded instead of refused")


def test_prepare_screens_a_label_with_missing_values_and_records_how_many_left():
    """Pins: the provided dataset's own shape. `train.csv` carries 3 unlabelled rows, and
    `prepare` - Step 0, the first command anyone runs - died on them with a numpy message
    about a list. A run cannot train on a row with no label, so the screen drops them; what
    it must not do is drop them silently, because the report's cleaning table has to be able
    to say how many rows left and why."""
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d) / "experiment"
        folder.mkdir()
        csv = fixture_csv(folder)
        frame = pd.read_csv(csv)
        frame.loc[[3, 17], "label"] = np.nan          # two rows with no label at all
        frame.to_csv(csv, index=False)

        assert prepare(folder, csv, "--label", "label") == 0

        screen = json.loads(
            (evidence(folder) / "candidate_screen.json").read_text(encoding="utf-8"))
        assert screen["rows_dropped_missing_label"] == 2, screen["rows_dropped_missing_label"]
        assert screen["class_names"] == ["no", "yes"], screen["class_names"]
        # They left *before* the split, not out of one side of it: pool + test is the frame
        # minus the two unlabelled rows.
        assert screen["n_pool"] + screen["n_test_held_out"] == 238, screen
        assert screen["subsampled"] is False, screen
        # Every family was still measured - the missing labels cost rows, not the screen.
        assert len(screen["families"]) >= 5, screen["families"]


def test_a_negative_label_code_is_refused_by_name_not_by_numpy():
    """Pins the message a reader gets. `np.bincount`'s own words were `'list' argument must
    have no negative elements` - true, and useless: they name no column, no file, and not
    the thing that went wrong. `driver._encode` no longer produces a negative code, so the
    guard's job is to say what a negative code *means* if some new producer makes one."""
    from validation import class_counts

    try:
        class_counts(np.array([0, 1, 1, -1]))
    except ValueError as e:
        assert "negative" in str(e), e
        assert "missing label" in str(e), e
    else:
        raise AssertionError("a negative label code was counted as a class")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
