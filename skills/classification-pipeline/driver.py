#!/usr/bin/env python
"""The one entry point: `prepare` the evidence, then `run` the plan.

    python driver.py prepare --dataset data/train.csv --experiment <folder>
    python driver.py run     --experiment <folder>

Why this file exists
--------------------
The workflow used to be a script the agent wrote. Three runs failed the same way, and the
last one is the diagnostic: the agent wrote `out/run.py` with the correct Step 0 imports
and then **never executed it**. It drifted into five hand-rolled scripts, and its final
artefact imported nothing from this package at all. Every guard in `scripts/` - the
output-directory refusal, the evidence contract, `evaluate_pair`'s two-model rule, the
`tune` label check - lives *inside* a module, and a guard can only refuse a call that is
made. A door nobody walks through is not a door.

So the execution path moved here. The agent's output is a validated `plan.json`, and this
file owns everything else: load, clean, encode, split, fit, tune, evaluate, figure, log,
render. Nothing in the plan can be code - `plan.validate_plan` refuses a `code`/`script`/
`custom_*` key at any depth - so the only way to get a report out of this package is to go
through this file.

Two properties worth stating outright, because they are the point of the design:

  * **`--experiment` is required and cwd is never consulted.** The failure that started
    this was `OUT = Path.cwd() / "out"` resolving into `.opencode/out/` - beside the
    machinery, outside the submitted folder, where nothing looks wrong. "Start the harness
    in the right directory" is an operational fix; taking the root as an argument is the
    one that holds whatever directory the harness happens to be in.
  * **A failure is loud and leaves no report.** `out/report.pdf` exists only if `run`
    exits 0. A failed run leaves the evidence it managed to write, and never a
    report-shaped pile of JSON that reads as a finished workflow.

What this file deliberately does NOT decide: the label, the two families, whether each is
scaled, the validation unit and strategy, the tuning metric, the headline metric, the
final-fit scope, the grids, and the cleaning policy. All of those arrive in the plan. If
you are about to add a default for one of them here, it belongs in SKILL.md instead.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd

import make_pdf
import report as report_mod
from data import (EVIDENCE_SCHEMA, classify_columns, label_candidates, load_table,
                  model_selection_evidence, profile, rarest_class_evidence, save_figures,
                  structure_evidence)
from evaluation import evaluate_pair, save_confusion_pair
from models import make_model
from paths import (EVIDENCE_SUBDIR, FIGURES_SUBDIR, OUT_NAME, PDF_NAME, PLAN_NAME,
                   REPORT_NAME, experiment_root)
from pipeline import build_pipeline, tune
from plan import PlanError, load_plan, plan_to_decisions
from screen import (BOTH_METRICS, SCREEN_FOLDS, SCREEN_SAMPLE, SCREEN_TEST_SIZE,
                    choose_protocol, run_screen)
from preprocessing import make_preprocessor
from validation import (fold_cap, fold_consequences, grouped_holdout, make_splitter,
                        stratified_group_folds, stratified_split, temporal_folds,
                        validation_report)

ALLOW_INTERPRETER_ENV = "SKILL_ALLOW_OTHER_INTERPRETER"
PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib")

# A reference number only, used for `validation_report`'s k-fold arithmetic when the plan
# chose a holdout and `n_splits` is therefore null. It is not a recommendation; the value
# actually used is recorded in validation.json so the report's numbers can be traced.
KFOLD_REFERENCE = 5


class DriverError(RuntimeError):
    """A refusal from this file. Reported with the step that raised it."""


class _Trace:
    """Remembers the step in progress, so a failure can name it.

    The workflow is long enough that a bare traceback says which line broke but not which
    *decision* led there. A failure that reads "FAILED at: tuning random_forest" is one the
    agent can act on without reading this file.
    """

    def __init__(self) -> None:
        self.step = "start"

    def __call__(self, name: str) -> None:
        self.step = name
        print(f"  [driver] {name}", flush=True)


# ------------------------------------------------------------------ writing artefacts


def _stamp(obj: dict, kind: str, producer: str) -> dict:
    """Attach the provenance `make_pdf.require_evidence` reads, without overwriting one."""
    obj.setdefault("_evidence", {"type": kind, "schema_version": EVIDENCE_SCHEMA,
                                 "producer": producer})
    return obj


def _write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _dirs(experiment: Path) -> dict:
    """Every path this run reads or writes, named once.

    Everything evidence-shaped lives under `out/evidence/` - including `decisions.json` and
    `metrics.json` - so `make_pdf.require_evidence` has exactly one rule and one spelling
    to check, and `out/` itself holds only the report and the figures it references.
    """
    out = experiment / OUT_NAME
    ev = out / EVIDENCE_SUBDIR
    return {"out": out, "evidence": ev, "figures": out / FIGURES_SUBDIR,
            "report_md": out / REPORT_NAME, "report_pdf": out / PDF_NAME,
            "plan": experiment / PLAN_NAME, "metrics": ev / "metrics.json",
            "environment": ev / "environment.json"}


def _environment(extra: dict | None = None) -> dict:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    env = {
        "interpreter": sys.executable,
        "interpreter_real": os.path.realpath(sys.executable),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }
    env.update(extra or {})
    return _stamp(env, "environment", "driver")


# ------------------------------------------------------------------ preparing evidence


def prepare(args) -> int:
    """Write the evidence the plan's decisions have to be argued from. Writes no plan."""
    experiment = experiment_root(args.experiment)
    d = _dirs(experiment)
    d["evidence"].mkdir(parents=True, exist_ok=True)

    print(f"  [driver] reading {args.dataset}")
    df = load_table(args.dataset)

    _write_json(d["evidence"] / "candidates.json",
                _stamp({"candidates": label_candidates(df),
                        "note": "every column that could plausibly be the target, with the "
                                "numbers behind it. Not a pick - choosing is yours, and if "
                                "two candidates are close, ask the user."},
                       "candidates", "label_candidates"))

    label = _resolve_label(args, d, df)

    if label is None:
        print(f"  [driver] no label given and no {PLAN_NAME} to read one from - wrote "
              f"candidates.json only. Re-run with --label <column> once you have chosen.")
    else:
        numeric, categorical, dropped_like = classify_columns(df, label)
        _write_profile_evidence(df, label, numeric, categorical, dropped_like, d["evidence"])
        save_figures(df, label, numeric, d["out"])
        print(f"  [driver] wrote the evidence for label {label!r}")
        if getattr(args, "no_screen", False):
            print("  [driver] --no-screen given, so no candidate screen was run. Step 4 "
                  "needs one; run `driver.py screen --experiment ...` before writing the plan.")
        else:
            ev = _screen_evidence(d, df, label)
            _write_json(d["evidence"] / "candidate_screen.json", ev)
            _print_shortlist(ev)

    _write_json(d["environment"], _environment(
        {"dataset": str(Path(args.dataset).resolve()),
         "test_dataset": str(Path(args.test_dataset).resolve()) if args.test_dataset
                         else None,
         "dataset_rows": int(len(df))}))
    print(f"  [driver] evidence in {d['evidence']}")
    return 0


def _write_profile_evidence(df: pd.DataFrame, label: str, numeric, categorical,
                            dropped_like, evidence_dir: Path,
                            label_reason: str = "") -> dict:
    """The four evidence files a plan's Steps 1-4 are argued from. Returns the profile."""
    eda = profile(df, label, numeric, categorical, dropped_like, label_reason=label_reason)
    _write_json(evidence_dir / "eda.json", eda)
    _write_json(evidence_dir / "model_selection.json",
                _stamp(model_selection_evidence(df, label, numeric, categorical),
                       "model_selection", "model_selection_evidence"))
    _write_json(evidence_dir / "structure.json",
                _stamp(structure_evidence(df, label, numeric, categorical, dropped_like),
                       "structure", "structure_evidence"))
    _write_json(evidence_dir / "rarest_class.json",
                _stamp(rarest_class_evidence(df, label, numeric, categorical),
                       "rarest_class", "rarest_class_evidence"))
    return eda


def _resolve_label(args, d, df) -> str | None:
    """The target column: `--label`, else the existing plan's, else None.

    Shared by `prepare` and `screen` so the two cannot disagree about which column is the
    target - a screen run under one label and a plan written against another would produce
    a shortlist that silently has nothing to do with the plan.
    """
    label = getattr(args, "label", None)
    if label is None:
        existing = _read_json(d["plan"])
        if isinstance(existing, dict):
            label = (existing.get("target") or {}).get("column")
            if label:
                print(f"  [driver] label {label!r} taken from the existing {PLAN_NAME}")
    if label is not None and label not in df.columns:
        raise DriverError(
            f"label {label!r} is not a column of the dataset. Columns: "
            f"{[str(c) for c in df.columns]}"
        )
    return label


def _screen_evidence(d, df: pd.DataFrame, label: str, *, group_column=None,
                     time_column=None, folds: int = SCREEN_FOLDS,
                     sample: int = SCREEN_SAMPLE, seed: int = 42) -> dict:
    """Fit every family that can run, on one protocol, and return the evidence.

    The split is formed first and the screen is then run on the pool alone. That ordering is
    the point: a family ranked on rows the final model will later be scored against would
    make the report's held-out number optimistic, and nothing would raise.
    """
    # The screen scores every family, so every row it screens needs a label. `run` drops
    # these rows too - `clean`, by the plan's `drop_rows_with_missing_label`, which is true
    # in every plan written so far - so dropping them here keeps the shortlist and the
    # report about the same rows. The count is recorded rather than merely dropped: the
    # report's cleaning table has to be able to say how many rows left and why, and a table
    # that cannot cite this number is describing a different frame from the one screened.
    n_missing_label = int(df[label].isna().sum())
    if n_missing_label:
        df = df[df[label].notna()].reset_index(drop=True)
        print(f"  [driver] {n_missing_label} row(s) carry no label and cannot be screened; "
              f"dropped, {len(df):,} rows left")
    y, class_names = _encode(df[label])
    structure = _read_json(d["evidence"] / "structure.json") or {}
    selection = _read_json(d["evidence"] / "model_selection.json") or {}
    existing = _read_json(d["plan"]) or {}

    # Screen on the same feature set the run will use. When a plan already exists it may
    # have held a leaky column out in `cleaning.drop_columns`; leaving that column in the
    # screen would rank the families on columns the final models never see, and the
    # shortlist would be about a different question than the report answers.
    drop = list((existing.get("cleaning") or {}).get("drop_columns") or [])
    numeric, categorical, _ = classify_columns(df, label)
    numeric = [c for c in numeric if c not in drop]
    categorical = [c for c in categorical if c not in drop]
    if drop:
        print(f"  [driver] the existing {PLAN_NAME} holds out {drop}; screening without "
              f"them, so the shortlist is about the same columns the run will use")

    protocol = choose_protocol(structure, group_column=group_column,
                               time_column=time_column)
    def plan_with(n_splits: int) -> dict:
        return {"seed": seed, "target": {"column": label},
                "cleaning": {"drop_columns": drop},
                "validation": {
                    "unit": protocol["unit"], "strategy": protocol["strategy"],
                    "test_size": SCREEN_TEST_SIZE, "val_fraction": SCREEN_TEST_SIZE,
                    "n_splits": n_splits, "gap": 0 if protocol["unit"] == "time" else None,
                    "group_column": protocol["group_column"],
                    "time_column": protocol["time_column"]}}

    print(f"  [driver] screening under {protocol['unit']}/{protocol['strategy']} - "
          f"{protocol['why']}")
    # The fold cap is a property of the *pool*, not of the whole frame, because the test
    # rows leave before any fold is cut. So `split_frame` runs once at the floor to learn
    # the pool's rarest-class count and once at the folds actually used - the test split
    # depends only on `test_size` and the seed, so both calls cut the same one.
    sp = split_frame(df, y, plan_with(2))
    folds = max(2, min(int(folds), fold_cap(sp["y_pool"])))
    if folds > 2:
        sp = split_frame(df, y, plan_with(folds))
    print(f"  [driver] {sp['n_pool']:,} pool rows screened, {sp['n_test']:,} test rows "
          f"held out of the screen")
    ev = dict(run_screen(
        sp["X_pool"], sp["y_pool"], numeric=numeric, categorical=categorical,
        splitter=sp["consequences_splitter"], groups=sp["groups_pool"],
        one_hot_width=selection.get("one_hot_width"), protocol=protocol, folds=folds,
        seed=seed, labels=list(range(len(class_names))), metrics=BOTH_METRICS,
        sample=sample))
    ev["class_names"] = class_names
    ev["rows_dropped_missing_label"] = n_missing_label
    ev["n_test_held_out"] = int(sp["n_test"])
    ev["test_rows_excluded"] = (
        "no family was fitted or scored on the test rows - they were split off before the "
        "screen started")
    return _stamp(ev, "candidate_screen", "screen")


def _check_pair_against_screen(plan: dict, evidence_dir: Path) -> None:
    """Say so when the plan's two families are not the ones the screen shortlisted.

    Not a refusal. A person may have a reason the screen cannot see - a cost bound, a
    departmental requirement, a family that errored - and refusing would make the screen a
    rule rather than evidence. But the divergence has to be *visible*, because the failure
    this whole mechanism replaced was exactly a pair chosen first and justified afterwards:
    it left no trace, and every artifact still looked consistent.

    Silent when there is no screen at all, which is the `run`-without-`prepare` path.
    """
    ev = _read_json(evidence_dir / "candidate_screen.json")
    if not isinstance(ev, dict):
        return
    chosen = [m["family"] for m in plan.get("models") or []]
    shortlisted = set(ev.get("shortlist", {}).get("f1_macro") or []) | \
        set(ev.get("shortlist", {}).get("accuracy") or [])
    absent = {e["family"] for e in (ev.get("excluded_without_fitting") or [])} | \
        {e["family"] for e in (ev.get("errored") or [])}

    unmeasured = [f for f in chosen if f in absent]
    if unmeasured:
        print(f"  [driver] WARNING: {', '.join(unmeasured)} was in neither the screen's "
              f"scored list nor its shortlist - it was not measured. The report must say so "
              f"and must not describe it as eliminated.")
    outside = [f for f in chosen if f not in shortlisted and f not in absent]
    if outside:
        print(f"  [driver] WARNING: {', '.join(outside)} is not on the screen's shortlist "
              f"(shortlisted: {', '.join(sorted(shortlisted)) or 'none'}). That is allowed - "
              f"a person may know something the screen cannot - but the plan's "
              f"narrative.candidate_models has to say why.")


def _print_shortlist(ev: dict) -> None:
    """Print the screen the way a person reads it: ranked, with the spread, then the pool."""
    rows = [r for r in ev["families"] if r["n_folds"]]
    print(f"  [driver] screened {len(rows)} families over {ev['n_screened']:,} rows"
          + (f" (subsampled from {ev['n_pool']:,})" if ev["subsampled"] else ""))
    for r in sorted(rows, key=lambda r: -r["mean"]["f1_macro"]):
        print(f"  [driver]   {r['family']:20} f1_macro {r['mean']['f1_macro']:.4f} "
              f"(±{r['std']['f1_macro']:.4f})   accuracy {r['mean']['accuracy']:.4f}")
    for ex in ev["excluded_without_fitting"]:
        print(f"  [driver]   {ex['family']:20} {ex['reason']}")
    for err in ev.get("errored") or []:
        print(f"  [driver]   {err['family']:20} NOT SCORED - {err['error']}")
        print(f"  [driver]   {'':20} This family was neither measured nor ruled out. Fix the "
              f"error and re-run `screen`, or if it is an environment fault, say so in the "
              f"plan - do not treat it as eliminated.")
    for metric in BOTH_METRICS:
        print(f"  [driver] shortlist ({metric}, within one std of the best): "
              f"{', '.join(ev['shortlist'][metric]) or 'none'}")
        biases = ev.get("bias") or {}
        for a, b in ev.get("pairs_with_different_bias", {}).get(metric) or []:
            print(f"  [driver]   a legal pair: {a} ({biases.get(a)}) + "
                  f"{b} ({biases.get(b)})")
        if not (ev.get("pairs_with_different_bias", {}).get(metric) or []):
            print(f"  [driver]   no pair inside this shortlist differs in inductive bias "
                  f"- take the best of it plus the highest-scoring family outside it, and "
                  f"say in the plan that the shortlist could not supply a pair.")
    print(f"  [driver] Step 4 takes TWO from the shortlist, differing in inductive bias.")


def screen(args) -> int:
    """Re-run the candidate screen on its own, to change the protocol or the sample size."""
    experiment = experiment_root(args.experiment)
    d = _dirs(experiment)
    d["evidence"].mkdir(parents=True, exist_ok=True)
    print(f"  [driver] reading {args.dataset}")
    df = load_table(args.dataset)

    label = _resolve_label(args, d, df)
    if label is None:
        raise DriverError(
            "screening needs a target column, and neither --label nor an existing "
            f"{PLAN_NAME} named one. Run `prepare --dataset ... --experiment ...` first and "
            "read candidates.json."
        )
    structure_path = d["evidence"] / "structure.json"
    if not structure_path.exists():
        raise DriverError(
            f"{structure_path} is missing and the screen reads it to decide the protocol. "
            f"Run `prepare --dataset {args.dataset} --experiment {args.experiment} "
            f"--label {label}` first."
        )
    ev = _screen_evidence(d, df, label, group_column=args.group_column,
                          time_column=args.time_column, folds=args.folds,
                          sample=args.sample, seed=args.seed)
    _write_json(d["evidence"] / "candidate_screen.json", ev)
    _print_shortlist(ev)
    print(f"  [driver] wrote {d['evidence'] / 'candidate_screen.json'}")
    return 0


def _features(df: pd.DataFrame, label: str, plan: dict) -> tuple[list, list, list, list]:
    """(numeric, categorical, id-like dropped automatically, held out by the plan).

    A name in `cleaning.drop_columns` that is not a column is refused rather than ignored:
    the decision was to hold that column out of the model, and a silently-ignored typo
    leaves it in the feature set - the decision made and not taken.
    """
    numeric, categorical, dropped_like = classify_columns(df, label)
    requested = list((plan.get("cleaning") or {}).get("drop_columns") or [])
    unknown = [c for c in requested if c not in df.columns]
    if unknown:
        raise DriverError(
            f"cleaning.drop_columns names {unknown}, which are not columns of the dataset. "
            f"Columns: {[str(c) for c in df.columns]}. An unrecognised name here means the "
            f"column you meant to hold out is still in the model."
        )
    keep = lambda cols: [c for c in cols if c not in requested]     # noqa: E731
    return keep(numeric), keep(categorical), dropped_like, requested


# ---------------------------------------------------------------------- cleaning a frame


def clean(df: pd.DataFrame, plan: dict) -> tuple[pd.DataFrame, dict]:
    """Apply the plan's cleaning policy. Returns the frame and a record of what it did.

    Every action here is one of the plan's closed-menu values, so the report's cleaning
    table always has something behind it. The two refusals are cases where the decision was
    made but cannot be carried out; carrying on would produce a frame that does not match
    the plan while nothing says so.
    """
    cl = plan["cleaning"]
    label = plan["target"]["column"]
    numeric, _, _, _ = _features(df, label, {"cleaning": {}})
    log = {"label": label,
           "n_rows_loaded": int(len(df)),
           "rows_dropped_missing_label": 0, "rows_dropped_duplicate_rarest": 0,
           "rows_dropped_outliers": 0,
           "outlier_policy": cl.get("outlier_policy"),
           "rarest_class_action": cl.get("rarest_class_action")}

    # 1. rows with no label. Not imputable and not scorable, so the plan has to have said
    #    what happens to them; "keep" is a decision the frame cannot honour.
    n_missing_label = int(df[label].isna().sum())
    if n_missing_label and not cl["drop_rows_with_missing_label"]:
        raise DriverError(
            f"{n_missing_label} row(s) have no label, and "
            f"cleaning.drop_rows_with_missing_label is false. A row with no label cannot be "
            f"encoded, fitted or scored - there is nothing to learn from it and nothing to "
            f"compare a prediction against. Set it to true, or explain in the plan why the "
            f"column you chose is not the label."
        )
    if cl["drop_rows_with_missing_label"] and n_missing_label:
        df = df[df[label].notna()]
        log["rows_dropped_missing_label"] = n_missing_label

    # 2. the rarest class. Bounded on purpose: `keep`, or drop the rows of that class that
    #    are exact duplicates of each other, so only the unique count is the sample size.
    if cl["rarest_class_action"] == "drop_duplicated_rows":
        counts = df[label].value_counts()
        rare = counts.idxmin()
        sub = df[df[label] == rare]
        dup = sub.duplicated(keep="first").to_numpy()
        log["rare_class_deduplicated"] = str(rare)
        log["rows_dropped_duplicate_rarest"] = int(dup.sum())
        df = df.drop(index=sub.index[dup])

    # 3. outliers, from the IQR fence the plan asked for.
    policy = cl.get("outlier_policy") or {}
    if policy.get("method") == "iqr":
        cols = [c for c in (policy.get("columns") or numeric) if c in df.columns]
        if not cols:
            raise DriverError(
                "cleaning.outlier_policy is an IQR filter but names no numeric column that "
                "exists. There is nothing to filter, so the policy cannot be carried out."
            )
        factor = float(policy["factor"])
        keep = pd.Series(True, index=df.index)
        for c in cols:
            q1, q3 = df[c].quantile([0.25, 0.75])
            width = q3 - q1
            keep &= df[c].isna() | df[c].between(q1 - factor * width, q3 + factor * width)
        log["rows_dropped_outliers"] = int((~keep).sum())
        log["outlier_columns"] = list(cols)
        df = df.loc[keep]

    df = df.reset_index(drop=True)
    if not len(df):
        raise DriverError("cleaning removed every row; there is nothing left to model")
    return df, log


def _encode(y: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Sorted distinct labels -> 0..k-1, with the display names kept in that order.

    Encoding here rather than asking the plan for an encoding has one purpose: it removes
    the trap where a string target makes `f1_score(..., average="macro")` average over a
    class set that silently drops a rare class, or where a hand-written mapping reorders
    the classes so a per-class table is keyed by the wrong name. Sorted order fixes the
    mapping, and `class_names` carries the names back out.

    A missing label is refused, not encoded. `y.astype(str)` turns a missing value into the
    *string* `"nan"`, which is in no mapping, so `map` leaves a NaN behind and
    `to_numpy(dtype=int)` casts it to whatever the hardware makes of it - observed values
    are large and negative, and the cast raises nothing, it only warns. `np.bincount` then
    dies on the negative code with `'list' argument must have no negative elements`, which
    names neither the column nor the cause. Worse, a frame whose missing labels happened to
    cast non-negative would raise nothing at all: the screen would rank families on rows
    belonging to a class that does not exist. Deciding the row set is the caller's job,
    because only the caller can say why.
    """
    n_missing = int(y.isna().sum())
    if n_missing:
        raise DriverError(
            f"{n_missing} row(s) have no label. A row with no label cannot be trained on or "
            f"scored, and casting it to a class index gives a code that belongs to no class. "
            f"Drop those rows first - `clean` does it when the plan's "
            f"cleaning.drop_rows_with_missing_label is true."
        )
    names = sorted(str(v) for v in pd.unique(y))
    mapping = {name: i for i, name in enumerate(names)}
    return y.astype(str).map(mapping).to_numpy(dtype=int), names


def _seeded(family: str, seed: int):
    """The plan's seed, handed to the estimator wherever it has somewhere to put it.

    Every family in the registry is built with scikit-learn's own defaults and none of them
    sets `random_state`, which is deliberate - a registry that seeded its estimators would
    be making the seed decision for the plan. The consequence is that the seed has to be
    applied *here*, by whatever executes the plan, and until it was, `plan.json` declaring
    `"seed": 42` was a claim the report printed and nothing honoured: two identical runs on
    the real dataset produced `random_forest` macro-F1 0.7520 and 0.7538, and the forest's
    own bootstrap is what moved.

    A family with no such parameter (knn, naive_bayes) is returned untouched rather than
    raising - the seed is the plan's, not the family's, and there is nothing to disagree
    with. `get_params()` is what decides, so this stays correct for a family added to the
    registry later without a matching edit here.
    """
    model = make_model(family)
    if "random_state" in model.get_params():
        model.set_params(random_state=seed)
    return model


# ------------------------------------------------------------------------ splitting


class _OneSplit:
    """A stand-in splitter for the holdout case, so one split reads like many.

    `fold_consequences` reports what each fold contains, which is how a class that a fold
    never validates becomes visible. A holdout has exactly one fold, and wrapping it means
    the entity-disjoint check still applies to it instead of being skipped because the
    holdout came from `train_test_split` rather than from a splitter object.
    """

    def __init__(self, train_idx, val_idx, *, enforces_disjoint_groups: bool = False):
        self._split = [(np.asarray(train_idx), np.asarray(val_idx))]
        self.enforces_disjoint_groups = enforces_disjoint_groups

    def split(self, X=None, y=None, groups=None):
        return list(self._split)

    def __repr__(self) -> str:
        return "the single holdout actually used"


def _positions(X: pd.DataFrame, idx) -> pd.DataFrame:
    """`X` restricted to `idx`, renumbered so that label and position agree again.

    Every index a splitter or `fold_consequences` hands back is *positional*, and it is
    positional within the frame it was given. Rebuilding each sub-frame with a fresh
    RangeIndex is what keeps those two meanings the same number - without it, an index
    that is a row's position in the whole frame gets used as a position in a sub-frame,
    and the arithmetic still produces a number.
    """
    return X.iloc[idx].reset_index(drop=True)


def split_frame(df: pd.DataFrame, y: np.ndarray, plan: dict, *,
                external_test: tuple | None = None) -> dict:
    """Everything the workflow needs about the split, from the plan's `validation` block.

    The test set is formed first and is never returned to the selection path.
    `external_test` is the `(X_test, y_test)` pair when the plan named a separate labelled
    test file; in that case the whole training frame is the pool.
    """
    v = plan["validation"]
    seed = plan["seed"]
    unit, strategy = v["unit"], v["strategy"]
    label = plan["target"]["column"]

    numeric, categorical, _, _ = _features(df, label, plan)
    feature_cols = numeric + categorical
    X = df[feature_cols]

    groups_all = (df[v["group_column"]].astype(str).to_numpy()
                  if v.get("group_column") else None)
    order_all = None
    if v.get("time_column"):
        col = v["time_column"]
        if col not in df.columns:
            raise DriverError(
                f"validation.time_column is {col!r}, which is not a column of the dataset. "
                f"Rows cannot be ordered by a column that is not there."
            )
        parsed = pd.to_datetime(df[col], errors="coerce")
        n_bad = int(parsed.isna().sum() - df[col].isna().sum())
        if n_bad:
            raise DriverError(
                f"validation.time_column {col!r} does not parse as a date for {n_bad} "
                f"row(s). A rolling-origin split on a partly-unparseable column trains on "
                f"the future for those rows, and the score still looks fine."
            )
        order_all = parsed.reset_index(drop=True)

    # -- the test set: formed first, and never re-entered ------------------------------
    if external_test is not None:
        X_test, y_test = external_test
        pool_idx = np.arange(len(X))
        X_pool = _positions(X, pool_idx)
        y_pool = y
        test_split_prose = (f"`{Path(plan['test_dataset']).name}`, a labelled file kept "
                            f"aside from the start")
        test_source = "provided file"
    else:
        n = len(X)
        if unit == "row":
            X_pool, X_test, y_pool, y_test = stratified_split(X, y, v["test_size"], seed)
            pool_idx = X_pool.index.to_numpy()
            X_pool = X_pool.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            test_split_prose = "stratified by class"
        elif unit == "entity":
            splitter = grouped_holdout(v["test_size"], seed)
            pool_idx, test_idx = next(iter(splitter.split(X, y, groups_all)))
            X_pool, X_test = _positions(X, pool_idx), _positions(X, test_idx)
            y_pool, y_test = y[pool_idx], y[test_idx]
            test_split_prose = "held out by whole entity, so no entity is on both sides"
        else:
            if not order_all.is_monotonic_increasing:
                # `temporal_folds` refuses an unsorted frame for the folds; the test split
                # has to obey the same rule. Sorting by the column the plan named is what
                # the plan asked for, not a choice this file is making.
                order = order_all.sort_values(kind="stable")
                pos = order.index.to_numpy()
                X, y = _positions(X, pos), y[pos]
                if groups_all is not None:
                    groups_all = groups_all[pos]
                order_all = order.reset_index(drop=True)
            n_test = max(1, int(round(n * v["test_size"])))
            test_idx = np.arange(n - n_test, n)
            pool_idx = np.arange(0, n - n_test)
            X_pool, X_test = _positions(X, pool_idx), _positions(X, test_idx)
            y_pool, y_test = y[pool_idx], y[test_idx]
            test_split_prose = ("the last rows in time order, so the model is scored on "
                                "the future")
        test_source = "split from dataset"

    groups_pool = groups_all[pool_idx] if groups_all is not None else None
    order_pool = order_all.to_numpy()[pool_idx] if order_all is not None else None

    out = {
        "X_pool": X_pool, "y_pool": y_pool, "X_test": X_test, "y_test": y_test,
        "groups_pool": groups_pool, "order_pool": order_pool,
        "test_split": test_split_prose, "test_source": test_source,
        "n_pool": int(len(X_pool)), "n_test": int(len(X_test)),
    }

    # -- validation: a block held out of the pool, or folds over it --------------------
    if strategy in ("holdout", "grouped_holdout"):
        if unit == "row":
            X_tr, X_va, y_tr, y_va = stratified_split(X_pool, y_pool, v["val_fraction"], seed)
            tr_idx, va_idx = X_tr.index.to_numpy(), X_va.index.to_numpy()
        else:
            splitter = grouped_holdout(v["val_fraction"], seed)
            tr_idx, va_idx = next(iter(splitter.split(X_pool, y_pool, groups_pool)))
            y_tr, y_va = y_pool[tr_idx], y_pool[va_idx]
        out.update({
            "mode": "holdout", "X_tr": _positions(X_pool, tr_idx), "y_tr": y_tr,
            "y_va": y_va, "n_val": int(len(y_va)),
            "tune_kwargs": {"X_score": _positions(X_pool, va_idx), "y_score": y_va},
            "consequences_splitter": _OneSplit(tr_idx, va_idx,
                                               enforces_disjoint_groups=(unit == "entity")),
            "n_folds": None,
        })
    else:
        cap = fold_cap(y_pool)
        if v["n_splits"] > cap:
            raise DriverError(
                f"validation.n_splits is {v['n_splits']} but the rarest class holds only "
                f"{cap} row(s) of the pool. A fold cannot contain one row of every class, "
                f"and the fold that comes up short is the one that scores higher - see "
                f"`fold_cap`. Cap n_splits at {cap}."
            )
        if strategy == "stratified_kfold":
            splitter = make_splitter(v["n_splits"], seed)
            kwargs = {"folds": v["n_splits"]}
        elif strategy == "stratified_group_kfold":
            splitter = stratified_group_folds(v["n_splits"], seed)
            kwargs = {"splitter": splitter, "groups": groups_pool}
        else:
            splitter = temporal_folds(v["n_splits"], v["gap"], order_pool)
            kwargs = {"splitter": splitter}
        out.update({
            "mode": "folds", "X_tr": X_pool, "y_tr": y_pool, "y_va": None, "n_val": 0,
            "tune_kwargs": kwargs, "consequences_splitter": splitter,
            "n_folds": v["n_splits"],
        })

    out["n_train"] = int(len(out["X_tr"]))
    return out


# ------------------------------------------------------------------------------ run


def run(args) -> int:
    """Execute `plan.json` end to end. Any failure means no report."""
    trace = _Trace()
    experiment = experiment_root(args.experiment)
    d = _dirs(experiment)

    try:
        # 1. no stale report ------------------------------------------------------------
        # Before the plan is even read. If this came after validation, a plan with a typo
        # would be refused and leave the PREVIOUS run's report.pdf sitting in the folder
        # looking current - which is the shape of mistake this whole design is about. A
        # report exists if and only if a run reached the end; the cost is that a good
        # report is lost when a plan is refused, and it is reproducible by re-running.
        trace("clearing any report left by an earlier run")
        for stale in (d["report_md"], d["report_pdf"]):
            if stale.exists():
                stale.unlink()
                print(f"  [driver] removed a stale {stale.relative_to(experiment)}")

        # 2. the plan -------------------------------------------------------------------
        trace(f"loading and validating {d['plan'].name}")
        plan = load_plan(d["plan"])
        label = plan["target"]["column"]

        # 3. the interpreter the evidence was prepared with -----------------------------
        trace("checking the interpreter against the prepared evidence")
        preparer = _read_json(d["environment"])
        if preparer and os.environ.get(ALLOW_INTERPRETER_ENV) != "1":
            was = preparer.get("interpreter_real")
            now = os.path.realpath(sys.executable)
            if was and was != now:
                raise DriverError(
                    f"the evidence was prepared by a different interpreter.\n"
                    f"  prepared with: {preparer.get('interpreter')}\n"
                    f"  running with : {sys.executable}\n"
                    f"  the evidence describes a frame read, and versions resolved, in one "
                    f"environment; executing the plan in another fits models with different "
                    f"defaults and nothing says so.\n"
                    f"  re-run `prepare` with this interpreter, or set "
                    f"{ALLOW_INTERPRETER_ENV}=1 to proceed anyway."
                )

        # 4. the data ------------------------------------------------------------------
        trace(f"reading {plan['dataset']}")
        df = load_table(plan["dataset"])
        if label not in df.columns:
            raise DriverError(
                f"target.column is {label!r}, which is not a column of {plan['dataset']}. "
                f"Columns: {[str(c) for c in df.columns]}"
            )
        numeric, categorical, dropped_like, excluded = _features(df, label, plan)
        d["evidence"].mkdir(parents=True, exist_ok=True)

        # 5. the evidence the plan was argued from (re-derived where prepare never ran) --
        trace("writing the evidence")
        _write_profile_evidence(df, label, numeric, categorical, dropped_like,
                                d["evidence"], label_reason=plan["target"]["reason"])
        if not (d["evidence"] / "candidates.json").exists():
            _write_json(d["evidence"] / "candidates.json",
                        _stamp({"candidates": label_candidates(df)},
                               "candidates", "label_candidates"))
        if not d["environment"].exists():
            _write_json(d["environment"], _environment())
        _check_pair_against_screen(plan, d["evidence"])

        # 6. cleaning -------------------------------------------------------------------
        trace("applying the cleaning policy")
        df, cleaning = clean(df, plan)
        cleaning.update({"n_rows_modelled": int(len(df)),
                         "n_features_modelled": len(numeric) + len(categorical),
                         "numeric_features": numeric, "categorical_features": categorical,
                         "columns_excluded": excluded,
                         "columns_dropped_id_like": dropped_like})
        save_figures(df, label, numeric, d["out"])

        # 7. label encoding, owned here rather than asked for ---------------------------
        trace("encoding the label")
        y, class_names = _encode(df[label])
        print(f"  [driver] {len(class_names)} classes: "
              + ", ".join(f"{n}->{i}" for i, n in enumerate(class_names)))

        # 8. a separate test file, if the plan named one --------------------------------
        external_test = None
        if plan.get("test_dataset"):
            trace(f"reading the test file {plan['test_dataset']}")
            test_df = load_table(plan["test_dataset"])
            if label not in test_df.columns:
                raise DriverError(
                    f"test_dataset {plan['test_dataset']!r} has no {label!r} column, so "
                    f"nothing in it can be scored. Point `test_dataset` at a labelled file, "
                    f"or set it to null and let `test_size` split one off `dataset`."
                )
            n_test_before = int(len(test_df))
            test_df, _ = clean(test_df, plan)
            unknown = sorted({str(v) for v in test_df[label].unique()} - set(class_names))
            if unknown:
                raise DriverError(
                    f"the test file carries class(es) {unknown}, which do not appear in "
                    f"{plan['dataset']} ({class_names}). A model cannot predict a class it "
                    f"never saw, so those rows would be scored as errors of a kind the "
                    f"training run had no chance to avoid."
                )
            mapping = {name: i for i, name in enumerate(class_names)}
            external_test = (
                test_df[[c for c in numeric + categorical]].reset_index(drop=True),
                test_df[label].astype(str).map(mapping).to_numpy(dtype=int),
            )
            cleaning["test_file"] = {"path": plan["test_dataset"],
                                     "n_rows_loaded": n_test_before,
                                     "n_rows_modelled": int(len(test_df))}
        _write_json(d["evidence"] / "cleaning.json",
                    _stamp(cleaning, "cleaning", "driver"))

        # 9. splitting ------------------------------------------------------------------
        trace("forming the split")
        sp = split_frame(df, y, plan, external_test=external_test)
        print(f"  [driver] {sp['n_train']:,} train / {sp['n_val']:,} validation / "
              f"{sp['n_test']:,} test rows ({sp['test_source']})")

        # 10. tuning, per model ---------------------------------------------------------
        metric = plan["tuning"]["metric"]
        classes = list(range(len(class_names)))
        fitted, tuning_models = {}, {}
        for m in plan["models"]:
            family = m["family"]
            trace(f"tuning {family}")
            pipe = build_pipeline(make_preprocessor(numeric, categorical, m["scale"]),
                                  _seeded(family, plan["seed"]))
            grid = {f"clf__{k}": list(vals) for k, vals in m["grid"].items()}
            best, params, scores = tune(pipe, grid, sp["X_tr"], sp["y_tr"],
                                        metric=metric, labels=classes,
                                        seed=plan["seed"], **sp["tune_kwargs"])
            tuning_models[family] = {"family": family, "best_params": params,
                                     "grid": m["grid"], "best_score": best,
                                     "n_combinations": len(scores), "scores": scores,
                                     "metric": metric, "scale": m["scale"]}
            print(f"  [driver] {family}: best {metric} {best:.4f} at "
                  f"{params or 'the only combination'}")

        # 11. the final fit -------------------------------------------------------------
        trace("fitting the final models")
        scope = plan["tuning"]["final_fit_scope"]
        fit_X, fit_y = ((sp["X_tr"], sp["y_tr"]) if scope == "train"
                        else (sp["X_pool"], sp["y_pool"]))
        for m in plan["models"]:
            family = m["family"]
            pipe = build_pipeline(make_preprocessor(numeric, categorical, m["scale"]),
                                  _seeded(family, plan["seed"]))
            pipe.set_params(**tuning_models[family]["best_params"])
            fitted[family] = pipe.fit(fit_X, fit_y)
        n_fit = int(len(fit_X))
        print(f"  [driver] final fit on {n_fit:,} rows (scope: {scope})")

        # 12. scoring on the test set ---------------------------------------------------
        trace("scoring both models on the test set")
        d["figures"].mkdir(parents=True, exist_ok=True)
        metrics = evaluate_pair(fitted, sp["X_test"], sp["y_test"],
                                fig_dir=d["figures"], class_names=class_names)
        _write_json(d["metrics"], metrics)
        save_confusion_pair(sp["y_test"],
                            {f: fitted[f].predict(sp["X_test"]) for f in fitted},
                            d["figures"], labels=class_names)

        # 13. what the split actually contains ------------------------------------------
        trace("recording the validation arithmetic")
        folds_ref = sp["n_folds"] or max(2, min(KFOLD_REFERENCE, fold_cap(sp["y_pool"])))
        consequences = fold_consequences(sp["consequences_splitter"], sp["X_pool"],
                                         sp["y_pool"], groups=sp["groups_pool"],
                                         classes=classes)
        validation = _stamp({
            "class_names": class_names,
            "split": {
                "mode": sp["mode"], "test_source": sp["test_source"],
                "test_split": sp["test_split"], "n_pool": sp["n_pool"],
                "n_train": sp["n_train"], "n_val": sp["n_val"], "n_test": sp["n_test"],
                "test_size": plan["validation"]["test_size"],
                "val_fraction": plan["validation"]["val_fraction"], "seed": plan["seed"],
                "val_class_counts": ({name: int((sp["y_va"] == i).sum())
                                      for i, name in enumerate(class_names)}
                                     if sp["mode"] == "holdout" else {}),
            },
            "final_fit_rows": n_fit,
            "kfold_reference_used_for_the_comparison": folds_ref,
            "validation_report": validation_report(
                sp["y_pool"], plan["validation"]["val_fraction"], folds_ref, class_names,
                groups=sp["groups_pool"],
                order=(sp["order_pool"].astype("int64")
                       if sp["order_pool"] is not None else None)),
            "fold_consequences": consequences,
        }, "validation", "driver")
        _write_json(d["evidence"] / "validation.json", validation)

        # 14. the search record, so the report prints what was actually chosen ----------
        _write_json(d["evidence"] / "tuning.json",
                    _stamp({"metric": metric, "protocol": _tuning_protocol(plan, sp),
                            "final_fit_scope": scope, "final_fit_rows": n_fit,
                            "models": tuning_models}, "tuning", "driver"))

        # 15. the audit record -----------------------------------------------------------
        trace("writing the decision log")
        plan_to_decisions(plan).write(d["evidence"] / "decisions.json")

        # 16. the report -----------------------------------------------------------------
        trace("rendering report.md")
        evidence = {name: _read_json(d["evidence"] / f"{name}.json")
                    for name in ("eda", "cleaning", "validation", "tuning")}
        d["report_md"].write_text(report_mod.render_report(plan, evidence, metrics),
                                  encoding="utf-8")
        print(f"  [driver] wrote {d['report_md'].relative_to(experiment)}")

        trace("rendering report.pdf")
        r = plan["report"]
        try:
            rc = make_pdf.main([
                "--md", str(d["report_md"]), "--out", str(d["report_pdf"]),
                "--student-id", r["student_id"], "--student-name", r["student_name"],
                "--variant", r["variant"], "--model-version", r["model_version"],
                "--harness", r["harness"], "--dataset", r["dataset_label"],
            ])
        except SystemExit as exc:
            # make_pdf refuses a markdown file it cannot render, evidence that does not
            # match what the workflow writes, and a target outside the experiment folder.
            # Either way what is left on disk is not a finished artefact.
            d["report_pdf"].unlink(missing_ok=True)
            raise DriverError(f"the PDF renderer refused to run: {exc}") from None
        if rc != 0:
            # Two pages is the assignment's hard limit, and a report over it is not a
            # finished artefact - so it is not left on disk looking like one.
            d["report_pdf"].unlink(missing_ok=True)
            raise DriverError(
                f"the report came out over the two-page limit and has been removed. "
                f"`{REPORT_NAME}` is still there: trim the narrative prose - "
                f"`performance_comparison` and `limitations` are the longest - then re-run. "
                f"Do not shrink the font."
            )

        trace("done")
        _headline(metrics, plan, sp, experiment, d)
        return 0

    except (DriverError, PlanError) as exc:
        print(f"\nFAILED at: {trace.step}\n"
              f"{exc}\n"
              f"\nNo report was written. The evidence in "
              f"{d['evidence'].relative_to(experiment)} is what this run managed to "
              f"produce; read it, fix the plan, and run again.", file=sys.stderr)
        return 1
    except Exception as exc:                                    # noqa: BLE001
        print(f"\nFAILED at: {trace.step}\n"
              f"  {type(exc).__name__}: {exc}\n"
              f"\nNo report was written. This is a defect in the run rather than in the "
              f"plan; the traceback below says where.", file=sys.stderr)
        raise


def _tuning_protocol(plan: dict, sp: dict) -> str:
    """One sentence naming the design the search actually ran under."""
    v = plan["validation"]
    if sp["mode"] == "folds":
        design = f"{v['n_splits']}-fold cross-validation over the {sp['n_pool']:,} pool rows"
    else:
        design = (f"a {v['val_fraction']:.0%} holdout of the pool ({sp['n_val']:,} of "
                  f"{sp['n_pool']:,} rows)")
    return f"{design}, scored on {plan['tuning']['metric']}"


def _headline(metrics, plan, sp, experiment: Path, d: dict) -> None:
    primary = plan["primary_metric"]["name"]
    print(f"\nwrote {d['report_pdf'].relative_to(experiment)}")
    print(f"test set: {sp['n_test']:,} rows | primary metric: {primary}")
    for m in plan["models"]:
        got = metrics[m["family"]]
        value = got.get(primary)
        shown = "n/a" if value is None else f"{value:.4f}"
        print(f"  {m['family']:<22} {primary:<16} {shown:<8} "
              f"accuracy {got['accuracy']:.4f}  macro-F1 {got['f1_macro']:.4f}")


# ------------------------------------------------------------------------------ cli


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="write the evidence the plan's decisions need")
    p.add_argument("--dataset", required=True, help="the labelled training file")
    p.add_argument("--test-dataset", default=None,
                   help="a separate labelled test file, held out untouched")
    p.add_argument("--experiment", required=True,
                   help="the experiment folder - the one CONTAINING the harness directory, "
                        "not the harness directory and not the skill directory")
    p.add_argument("--label", default=None,
                   help="the target column; omit it to see the candidates first")
    p.add_argument("--no-screen", action="store_true",
                   help="write the evidence but do not fit the candidate families - use it "
                        "if the screen is too slow, then run `screen` on its own")
    p.set_defaults(func=prepare)

    s = sub.add_parser(
        "screen", help="fit every candidate family on one protocol and report the spread",
        description="Step 4 needs the candidate families measured, not argued about. This "
                    "fits every family in the registry that can run, on one split, with its "
                    "own defaults, and writes out/evidence/candidate_screen.json. It is run "
                    "for you at the end of `prepare` unless you pass --no-screen.")
    s.add_argument("--dataset", required=True, help="the same file prepare was given")
    s.add_argument("--experiment", required=True, help="the same folder prepare was given")
    s.add_argument("--label", default=None,
                   help="the target column; defaults to the existing plan.json's")
    s.add_argument("--group-column", default=None,
                   help="force entity-disjoint folds on this column, overriding the search")
    s.add_argument("--time-column", default=None,
                   help="force rolling-origin folds on this column, overriding the search")
    s.add_argument("--folds", type=int, default=SCREEN_FOLDS,
                   help=f"folds per family (default {SCREEN_FOLDS}); lowered automatically "
                        f"when the rarest class in the pool cannot fill that many")
    s.add_argument("--sample", type=int, default=SCREEN_SAMPLE,
                   help=f"subsample the pool to this many rows when it is larger "
                        f"(default {SCREEN_SAMPLE:,}); whole entities are kept together")
    s.add_argument("--seed", type=int, default=42, help="seed for the split and the fits")
    s.set_defaults(func=screen)

    r = sub.add_parser("run", help="execute plan.json and write the report")
    r.add_argument("--experiment", required=True, help="the same folder prepare was given")
    r.set_defaults(func=run)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
