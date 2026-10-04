#!/usr/bin/env python
"""The contract between the agent's decisions and the skill's execution.

The workflow used to be a script the agent wrote: it imported these modules, decided
things, and called them in order. When that went wrong it went wrong *silently* - the run
wrote its output beside the machinery, or hand-rolled its own scikit-learn code and never
imported this package at all, so none of the guards inside `scripts/` were ever reached. A
guard can only refuse a call that is made.

So the split moved. The agent decides; this package executes. The agent's entire output is
one `plan.json`, validated here before any work starts.

What the plan may contain
-------------------------
Only values from **closed menus**, plus the prose that justifies them. There is no field
that takes an expression, a snippet, or a callable - `_forbidden_keys` refuses a key named
`code`, `script`, `query` or `custom_*` anywhere in the tree, at any depth. That is the
whole point: a plan that could carry code would put us back where we started, with the
guards bypassable by anyone who writes their own loader.

What this module checks, and why each check is here
--------------------------------------------------
Every refusal below stands in for a failure that is otherwise **invisible**:

  * *Grid keys* are checked against `make_model(family)().get_params()`. scikit-learn's own
    error for a bad key is `__init__() got an unexpected keyword argument`, which names
    neither the model nor the valid keys - so this message names both.
  * *The two families must differ.* `evaluate_pair` also refuses two of one family, but
    only after both have been fitted and tuned. Catching it here costs nothing and says so
    before the expensive part.
  * *`rejected` must differ from `name`.* A primary metric whose rejected alternative is
    itself reads as a decision that was never made.
  * *Reason and alternatives are required on every decision*, mirroring
    `DecisionLog.record`. A blank reason column still renders and still paginates.

Validation is deliberately **exhaustive, not fail-fast**: `_plan_problems` reports every
problem it can find, because a plan fixed one error per run costs one run per error.

If you are about to add a default or a policy to this file, it belongs in SKILL.md instead.
"""

from __future__ import annotations

import json
from pathlib import Path

from models import make_model, model_names

SCHEMA_VERSION = 1

# The layout, named once. `driver.py` writes these and `report.py` reads them; a second
# copy of "evidence" spelled slightly differently is how the report ends up looking for
# files that are not there.
PLAN_NAME = "plan.json"
OUT_NAME = "out"
EVIDENCE_SUBDIR = "evidence"

# ------------------------------------------------------------------ closed menus

UNITS = ("row", "entity", "time")

# Each strategy belongs to exactly one unit. The pairing is the check that matters: a
# group-aware splitter on rows that carry no entity is a random split with extra steps,
# and a row-based splitter on rows that repeat an entity leaks that entity across the
# boundary - both silently, both producing a number that looks like a result.
STRATEGY_UNIT = {
    "holdout": "row",
    "stratified_kfold": "row",
    "grouped_holdout": "entity",
    "stratified_group_kfold": "entity",
    "temporal": "time",
}
FOLD_STRATEGIES = ("stratified_kfold", "stratified_group_kfold", "temporal")
GROUP_STRATEGIES = ("grouped_holdout", "stratified_group_kfold")
HOLDOUT_STRATEGIES = ("holdout", "grouped_holdout")

# `tune` can score these two and no others - see `evaluation._score`.
TUNING_METRICS = ("f1_macro", "accuracy")

# The headline may be any of these; `evaluate` computes all of them. It is deliberately a
# wider menu than TUNING_METRICS, which is why the plan carries both.
PRIMARY_METRICS = ("accuracy", "f1_macro", "precision_macro", "recall_macro",
                   "roc_auc", "pr_auc")

FINAL_FIT_SCOPES = ("train", "train+validation", "pool")

RAREST_CLASS_ACTIONS = ("keep", "drop_duplicated_rows")

OUTLIER_METHODS = ("keep", "iqr")

EXACTLY_TWO_MODELS = 2

# Key names that would let a plan carry behaviour rather than a decision. Checked at every
# depth, case-insensitively. A plan is a set of choices from menus; anything that reads as
# "run this" is refused by name so the refusal explains itself.
FORBIDDEN_KEY_NAMES = frozenset({
    "code", "python", "script", "exec", "eval", "expr", "expression", "lambda",
    "callable", "source", "snippet", "formula", "query", "sql", "func", "function",
    "command", "shell", "body", "statements", "import", "module", "apply", "pipeline_code",
})
FORBIDDEN_KEY_PREFIXES = ("custom_", "on_", "pre_", "post_")

# ------------------------------------------------------------ the plan's own shape

TOP_KEYS = ("schema_version", "dataset", "test_dataset", "seed",
            "target", "cleaning", "models", "validation", "tuning",
            "primary_metric", "report", "narrative")

BLOCK_KEYS = {
    "target": ("column", "reason", "alternatives_considered"),
    "cleaning": ("drop_columns", "drop_rows_with_missing_label", "rarest_class_action",
                 "outlier_policy", "reason", "alternatives_considered"),
    "model": ("family", "scale", "scale_reason", "stopping", "overfitting_control",
              "grid", "reason", "alternatives_considered"),
    "validation": ("unit", "strategy", "test_size", "val_fraction", "n_splits", "gap",
                   "group_column", "time_column", "reason", "alternatives_considered"),
    "tuning": ("metric", "final_fit_scope", "final_fit_reason", "reason",
               "alternatives_considered"),
    "primary_metric": ("name", "reason", "rejected", "rejected_reason"),
    "report": ("student_id", "student_name", "variant", "model_version", "harness",
               "repository", "dataset_label"),
    "outlier_policy": ("method", "factor", "columns"),
}

NARRATIVE_FIELDS = ("candidate_models", "feature_engineering", "cleaning_why",
                    "validation_argument", "primary_metric_argument",
                    "performance_comparison", "limitations")

REPORT_REQUIRED = ("student_id", "student_name", "variant", "model_version", "harness",
                   "repository", "dataset_label")


class PlanError(ValueError):
    """A plan that cannot be executed. Carries every problem found, not the first."""


# ------------------------------------------------------------------- small helpers


def _text(where: str, d: dict, key: str, problems: list[str]) -> str | None:
    """Require a non-empty string at `d[key]`. Returns it, or None and records why."""
    v = d.get(key)
    if not isinstance(v, str) or not v.strip():
        problems.append(f"{where}.{key}: missing or empty (needs a non-empty string)")
        return None
    return v


def _alts(where: str, d: dict, problems: list[str]) -> None:
    """Require a non-empty list of alternatives - the same rule `DecisionLog` enforces."""
    v = d.get("alternatives_considered")
    if not isinstance(v, list) or not [a for a in v if str(a).strip()]:
        problems.append(
            f"{where}.alternatives_considered: missing or empty. A choice with nothing "
            f"else on the table was not a decision - name the option you rejected and why."
        )


def _unknown(where: str, d: dict, allowed: tuple, problems: list[str]) -> None:
    """Refuse keys outside the menu. Ignoring them hides the typo that caused them."""
    if not isinstance(d, dict):
        return
    stray = sorted(k for k in d if k not in allowed)
    if stray:
        problems.append(
            f"{where}: unknown key(s) {stray}. Allowed here: {list(allowed)}. "
            f"An unrecognised key is never ignored - a misspelling would otherwise be "
            f"accepted and the decision it was meant to carry silently dropped."
        )


def _forbidden_keys(node, where: str, problems: list[str]) -> None:
    """Walk the whole plan and refuse any key that could carry behaviour."""
    if isinstance(node, dict):
        for k, v in node.items():
            low = str(k).lower()
            if low in FORBIDDEN_KEY_NAMES or low.startswith(FORBIDDEN_KEY_PREFIXES):
                problems.append(
                    f"{where}.{k}: not allowed. A plan carries decisions, never code - "
                    f"the skill executes. Pick from the documented fields instead."
                )
            _forbidden_keys(v, f"{where}.{k}", problems)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _forbidden_keys(v, f"{where}[{i}]", problems)


def _number(where: str, d: dict, key: str, problems: list[str],
            lo: float, hi: float) -> float | None:
    v = d.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        problems.append(f"{where}.{key}: missing or not a number")
        return None
    if not (lo <= float(v) <= hi):
        problems.append(f"{where}.{key}: {v} is outside [{lo}, {hi}]")
        return None
    return float(v)


# ------------------------------------------------------------------- the blocks


def _check_target(plan: dict, problems: list[str]) -> None:
    t = plan.get("target")
    if not isinstance(t, dict):
        problems.append("target: missing (needs an object with `column`, `reason`, "
                        "`alternatives_considered`)")
        return
    _unknown("target", t, BLOCK_KEYS["target"], problems)
    column = _text("target", t, "column", problems)
    _text("target", t, "reason", problems)
    _alts("target", t, problems)
    if column and column in (plan.get("cleaning") or {}).get("drop_columns", []):
        problems.append(
            f"target.column: {column!r} is also listed in cleaning.drop_columns. The "
            f"target has to be present to be predicted - dropping it makes every model "
            f"useless and raises nothing."
        )


def _check_cleaning(plan: dict, problems: list[str]) -> None:
    cl = plan.get("cleaning")
    if not isinstance(cl, dict):
        problems.append("cleaning: missing (needs `drop_columns`, "
                        "`drop_rows_with_missing_label`, `rarest_class_action`, "
                        "`outlier_policy`, `reason`, `alternatives_considered`)")
        return
    _unknown("cleaning", cl, BLOCK_KEYS["cleaning"], problems)

    drop = cl.get("drop_columns")
    if not isinstance(drop, list) or not all(isinstance(c, str) for c in drop):
        problems.append("cleaning.drop_columns: missing or not a list of column names "
                        "(use [] when nothing is dropped)")

    if not isinstance(cl.get("drop_rows_with_missing_label"), bool):
        problems.append("cleaning.drop_rows_with_missing_label: missing or not true/false. "
                        "A missing label cannot be imputed and cannot be scored, so this "
                        "has to be an explicit answer rather than a default.")

    act = cl.get("rarest_class_action")
    if act not in RAREST_CLASS_ACTIONS:
        problems.append(f"cleaning.rarest_class_action: {act!r} is not one of "
                        f"{list(RAREST_CLASS_ACTIONS)}")

    op = cl.get("outlier_policy")
    if not isinstance(op, dict):
        problems.append('cleaning.outlier_policy: missing (use {"method": "keep"} to keep '
                        "every row)")
    else:
        _unknown("cleaning.outlier_policy", op, BLOCK_KEYS["outlier_policy"], problems)
        method = op.get("method")
        if method not in OUTLIER_METHODS:
            problems.append(f"cleaning.outlier_policy.method: {method!r} is not one of "
                            f"{list(OUTLIER_METHODS)}")
        elif method == "iqr":
            _number("cleaning.outlier_policy", op, "factor", problems, 0.1, 10.0)
            cols = op.get("columns")
            if cols is not None and (not isinstance(cols, list)
                                     or not all(isinstance(c, str) for c in cols)):
                problems.append("cleaning.outlier_policy.columns: must be a list of column "
                                "names, or omitted for all numeric columns")

    _text("cleaning", cl, "reason", problems)
    _alts("cleaning", cl, problems)


def _valid_grid_keys(family: str) -> list[str]:
    """The estimator's own parameter names - what a grid key has to be."""
    return sorted(make_model(family).get_params())


def _check_models(plan: dict, problems: list[str]) -> None:
    models = plan.get("models")
    if not isinstance(models, list):
        problems.append("models: missing (needs a list of exactly two model objects)")
        return
    if len(models) != EXACTLY_TWO_MODELS:
        problems.append(
            f"models: {len(models)} given, exactly {EXACTLY_TWO_MODELS} required. The "
            f"report compares two columns side by side; with one, the second column is "
            f"blank and nothing about the table says so."
        )

    families: list[str] = []
    for i, m in enumerate(models):
        where = f"models[{i}]"
        if not isinstance(m, dict):
            problems.append(f"{where}: not an object")
            continue
        _unknown(where, m, BLOCK_KEYS["model"], problems)

        family = m.get("family")
        if family not in model_names():
            problems.append(f"{where}.family: {family!r} is not a known family. "
                            f"Available: {model_names()}")
        else:
            families.append(family)

        if not isinstance(m.get("scale"), bool):
            problems.append(f"{where}.scale: missing or not true/false. Whether a family "
                            f"needs scaled inputs is a property of the family, not of the "
                            f"dataset, and the report is graded on the difference.")
        # The registry states a *strength* of preference - "not required", "generally
        # recommended", "effectively required" - and SKILL.md says the report is graded on
        # telling those apart. A bare boolean cannot carry that, so the reason is required
        # alongside it rather than reconstructed from the registry afterwards.
        _text(where, m, "scale_reason", problems)

        # The report has a "Stopping criterion" and an "Overfitting control" cell for each
        # model. Both are facts about the fit that only the decision-maker holds - whether
        # max_iter was reached, which cap bounds the tree's growth, what limits capacity.
        # Left out of the contract the renderer would have to phrase them from the tuned
        # hyperparameters, which is inventing a claim the run never made.
        _text(where, m, "stopping", problems)
        _text(where, m, "overfitting_control", problems)

        grid = m.get("grid")
        if not isinstance(grid, dict) or not grid:
            problems.append(f"{where}.grid: missing or empty (needs at least one "
                            f"hyperparameter to search)")
        elif family in model_names():
            allowed = _valid_grid_keys(family)
            for key, values in grid.items():
                if key not in allowed:
                    problems.append(
                        f"{where}.grid[{key!r}]: {family} has no such parameter. Valid: "
                        f"{allowed}"
                    )
                if not isinstance(values, list) or not values:
                    problems.append(f"{where}.grid[{key!r}]: needs a non-empty list of "
                                    f"values to try")

        _text(where, m, "reason", problems)
        _alts(where, m, problems)

    if len(families) == EXACTLY_TWO_MODELS and families[0] == families[1]:
        problems.append(
            f"models: both entries are {families[0]!r}. Two of one family differ only in "
            f"hyperparameters, so the comparison cannot say anything about inductive bias "
            f"- which is the only thing a two-model comparison is for."
        )


def _check_validation(plan: dict, problems: list[str]) -> None:
    v = plan.get("validation")
    if not isinstance(v, dict):
        problems.append("validation: missing (needs `unit`, `strategy`, and the strategy's "
                        "own parameters)")
        return
    _unknown("validation", v, BLOCK_KEYS["validation"], problems)

    unit = v.get("unit")
    if unit not in UNITS:
        problems.append(f"validation.unit: {unit!r} is not one of {list(UNITS)}")

    strat = v.get("strategy")
    if strat not in STRATEGY_UNIT:
        problems.append(f"validation.strategy: {strat!r} is not one of "
                        f"{list(STRATEGY_UNIT)}")
    elif unit in UNITS and STRATEGY_UNIT[strat] != unit:
        problems.append(
            f"validation: strategy {strat!r} splits at the {STRATEGY_UNIT[strat]!r} unit, "
            f"but unit is {unit!r}. The wrong pairing is silent - a group-aware splitter "
            f"on rows with no entity is a random split with extra steps, and a row-based "
            f"splitter on repeated entities leaks one entity across the boundary."
        )

    _number("validation", v, "test_size", problems, 0.01, 0.99)
    _number("validation", v, "val_fraction", problems, 0.01, 0.99)

    n_splits = v.get("n_splits")
    if strat in FOLD_STRATEGIES:
        if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
            problems.append(f"validation.n_splits: required for {strat!r} and must be an "
                            f"integer >= 2")
    elif n_splits is not None:
        problems.append(f"validation.n_splits: {n_splits!r} given but strategy {strat!r} "
                        f"is a holdout - it takes one estimate, not folds")

    if strat == "temporal":
        gap = v.get("gap")
        if not isinstance(gap, int) or isinstance(gap, bool) or gap < 0:
            problems.append("validation.gap: required for 'temporal' and must be an "
                            "integer >= 0 (the number of rows between train and test)")
    elif v.get("gap") is not None:
        problems.append(f"validation.gap: only 'temporal' uses a gap, not {strat!r}")

    for key, needed in (("group_column", GROUP_STRATEGIES),
                        ("time_column", ("temporal",))):
        value = v.get(key)
        if strat in needed:
            if not isinstance(value, str) or not value.strip():
                problems.append(f"validation.{key}: required for {strat!r} - the splitter "
                                f"cannot group or order rows it cannot name a column for")
        elif value is not None:
            problems.append(f"validation.{key}: {value!r} given but strategy {strat!r} "
                            f"does not use it; set it to null")

    _text("validation", v, "reason", problems)
    _alts("validation", v, problems)


def _check_tuning(plan: dict, problems: list[str]) -> None:
    t = plan.get("tuning")
    if not isinstance(t, dict):
        problems.append("tuning: missing (needs `metric`, `final_fit_scope`, "
                        "`final_fit_reason`, `reason`, `alternatives_considered`)")
        return
    _unknown("tuning", t, BLOCK_KEYS["tuning"], problems)

    if t.get("metric") not in TUNING_METRICS:
        problems.append(
            f"tuning.metric: {t.get('metric')!r} is not one of {list(TUNING_METRICS)}. "
            f"The search can only optimise a metric the scorer implements. If the headline "
            f"metric you want is neither, name it in primary_metric and say in "
            f"tuning.reason which one the search used instead."
        )
    if t.get("final_fit_scope") not in FINAL_FIT_SCOPES:
        problems.append(f"tuning.final_fit_scope: {t.get('final_fit_scope')!r} is not one "
                        f"of {list(FINAL_FIT_SCOPES)}")
    _text("tuning", t, "final_fit_reason", problems)
    _text("tuning", t, "reason", problems)
    _alts("tuning", t, problems)


def _check_primary_metric(plan: dict, problems: list[str]) -> None:
    p = plan.get("primary_metric")
    if not isinstance(p, dict):
        problems.append("primary_metric: missing (needs `name`, `reason`, `rejected`, "
                        "`rejected_reason`)")
        return
    _unknown("primary_metric", p, BLOCK_KEYS["primary_metric"], problems)

    name = p.get("name")
    if name not in PRIMARY_METRICS:
        problems.append(f"primary_metric.name: {name!r} is not one of "
                        f"{list(PRIMARY_METRICS)}")
    rejected = p.get("rejected")
    if not isinstance(rejected, str) or not rejected.strip():
        problems.append("primary_metric.rejected: missing - name the metric you did not "
                        "lead with")
    elif rejected == name:
        problems.append(f"primary_metric: `rejected` is also {name!r}. Rejecting the "
                        f"metric you chose reads as a decision that was never made.")
    _text("primary_metric", p, "reason", problems)
    _text("primary_metric", p, "rejected_reason", problems)


def _check_report(plan: dict, problems: list[str]) -> None:
    r = plan.get("report")
    if not isinstance(r, dict):
        problems.append(f"report: missing (needs {list(REPORT_REQUIRED)})")
        return
    _unknown("report", r, BLOCK_KEYS["report"], problems)
    for key in REPORT_REQUIRED:
        _text("report", r, key, problems)
    url = r.get("repository")
    if isinstance(url, str) and url.strip() and not url.startswith(("http://", "https://")):
        problems.append(f"report.repository: {url!r} is not a URL. This is the link to the "
                        f"skill itself - if you were not given one, ask rather than "
                        f"writing a plausible-looking address.")


def _check_narrative(plan: dict, problems: list[str]) -> None:
    n = plan.get("narrative")
    if not isinstance(n, dict):
        problems.append(f"narrative: missing (needs {list(NARRATIVE_FIELDS)}). These are "
                        f"the report's argument - the driver splices them in verbatim.")
        return
    _unknown("narrative", n, NARRATIVE_FIELDS, problems)

    for key in NARRATIVE_FIELDS:
        value = n.get(key)
        if key == "cleaning_why":
            if not isinstance(value, list) or not value:
                problems.append("narrative.cleaning_why: needs a non-empty list of "
                                "{issue, action, why} - it is the report's cleaning table")
                continue
            for i, row in enumerate(value):
                if not isinstance(row, dict):
                    problems.append(f"narrative.cleaning_why[{i}]: not an object")
                    continue
                _unknown(f"narrative.cleaning_why[{i}]", row, ("issue", "action", "why"),
                         problems)
                for field in ("issue", "action", "why"):
                    _text(f"narrative.cleaning_why[{i}]", row, field, problems)
        elif not isinstance(value, str) or not value.strip():
            problems.append(f"narrative.{key}: missing or empty")


# ---------------------------------------------------------------------- the entry


def _plan_problems(plan) -> list[str]:
    """Every reason this plan cannot be executed. Never raises; may return many lines.

    Exhaustive rather than fail-fast on purpose: a plan repaired one error per run costs
    one full run per error, and the errors here are cheap to find all at once.
    """
    problems: list[str] = []
    if not isinstance(plan, dict):
        return [f"plan: expected a JSON object, got {type(plan).__name__}"]

    _forbidden_keys(plan, "plan", problems)

    if plan.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version: expected {SCHEMA_VERSION}, got "
                        f"{plan.get('schema_version')!r}")
    _unknown("plan", plan, TOP_KEYS, problems)

    dataset = plan.get("dataset")
    if not isinstance(dataset, str) or not dataset.strip():
        problems.append("dataset: missing - the path to the training file")
    test = plan.get("test_dataset")
    if test is not None and (not isinstance(test, str) or not test.strip()):
        problems.append("test_dataset: must be a path or null (null splits a test set off "
                        "`dataset` instead)")
    seed = plan.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        problems.append("seed: missing or not an integer")

    _check_target(plan, problems)
    _check_cleaning(plan, problems)
    _check_models(plan, problems)
    _check_validation(plan, problems)
    _check_tuning(plan, problems)
    _check_primary_metric(plan, problems)
    _check_report(plan, problems)
    _check_narrative(plan, problems)

    return problems


def validate_plan(plan) -> dict:
    """Return the plan if it can be executed, else raise `PlanError` listing every fault."""
    problems = _plan_problems(plan)
    if problems:
        raise PlanError(
            "this plan cannot be executed:\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\n  fix plan.json and run again; nothing was executed and no report was "
              "written."
        )
    return plan


def load_plan(path) -> dict:
    """Read and validate `plan.json`. A malformed file is reported as one, not as a crash."""
    path = Path(path)
    if not path.exists():
        raise PlanError(
            f"no plan at {path}\n"
            f"  the workflow is: `driver.py prepare` to compute the evidence, then write "
            f"plan.json from it, then `driver.py run`."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise PlanError(f"{path} is not valid JSON: {e}\n"
                        f"  line {e.lineno}, column {e.colno}") from None
    return validate_plan(raw)


# ------------------------------------------------------- plan -> the audit record


def _choice(parts: list[str]) -> str:
    return "; ".join(p for p in parts if p)


def plan_to_decisions(plan: dict):
    """Build the `DecisionLog` this plan describes.

    The log is what `make_pdf`'s evidence contract checks for, and what a marker reads to
    see the reasoning without opening the driver. It is derived from the plan rather than
    written separately, so the two cannot disagree - a decision recorded in the report
    that the run did not actually take is not a mistake this design can make.
    """
    from decisions import DecisionLog

    log = DecisionLog()

    t = plan["target"]
    log.record("label", "target column", t["column"],
               reason=t["reason"], alternatives_considered=t["alternatives_considered"],
               inputs={"n_classes_hint": "see eda.json"})

    cl = plan["cleaning"]
    drops = cl.get("drop_columns") or []
    op = cl.get("outlier_policy") or {}
    log.record(
        "cleaning", "cleaning policy",
        _choice([
            "keep outliers" if op.get("method") == "keep"
            else f"filter outliers by IQR x{op.get('factor')}",
            "drop rows with no label" if cl.get("drop_rows_with_missing_label")
            else "keep rows with no label",
            f"rarest class: {cl.get('rarest_class_action')}",
            f"exclude {len(drops)} column(s) from the features" if drops
            else "no columns excluded",
        ]),
        reason=cl["reason"],
        alternatives_considered=cl["alternatives_considered"],
        inputs={"drop_columns": drops},
    )

    ms = plan["models"]
    log.record(
        "model_selection", "the two families",
        f"{ms[0]['family']} and {ms[1]['family']}",
        reason=" ".join(f"{m['family']}: {m['reason']}" for m in ms),
        alternatives_considered=[a for m in ms for a in m["alternatives_considered"]],
        inputs={"families": [m["family"] for m in ms],
                "stopping": {m["family"]: m["stopping"] for m in ms}},
    )
    log.record(
        "preprocessing", "scaling, per model",
        _choice([f"{m['family']}: scale={m['scale']}" for m in ms]),
        reason=" ".join(f"{m['family']}: {m['scale_reason']}" for m in ms),
        alternatives_considered=[
            f"{m['family']} with scale={not m['scale']} - the registry's stated preference "
            f"for this family is in reference.md section 3"
            for m in ms
        ],
    )

    v = plan["validation"]
    log.record(
        "validation", "how validation is formed",
        _choice([f"{v['unit']}-unit", v["strategy"],
                 f"n_splits={v['n_splits']}" if v.get("n_splits") else "",
                 f"gap={v['gap']}" if v.get("gap") else "",
                 f"test_size={v['test_size']}", f"val_fraction={v['val_fraction']}"]),
        reason=v["reason"],
        alternatives_considered=v["alternatives_considered"],
        inputs={"unit": v["unit"], "strategy": v["strategy"]},
    )

    tu = plan["tuning"]
    log.record("metrics", "tuning metric", tu["metric"], reason=tu["reason"],
               alternatives_considered=tu["alternatives_considered"])

    pm = plan["primary_metric"]
    log.record("metrics", "primary metric", pm["name"], reason=pm["reason"],
               alternatives_considered=[f"{pm['rejected']} - {pm['rejected_reason']}"])

    log.record(
        "final_fit", "what the final model is fitted on", tu["final_fit_scope"],
        reason=tu["final_fit_reason"],
        alternatives_considered=[
            f"{s} - the scopes differ in how much data the final fit sees and, for 'pool', "
            f"in whether the test estimate is still untouched by selection"
            for s in FINAL_FIT_SCOPES if s != tu["final_fit_scope"]
        ],
        inputs={"final_fit_scope": tu["final_fit_scope"]},
    )

    return log
