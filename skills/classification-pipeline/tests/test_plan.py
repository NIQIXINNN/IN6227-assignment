#!/usr/bin/env python
"""Checks for the plan contract in `scripts/plan.py`.

    python tests/test_plan.py

`plan.json` is the agent's entire output, so every refusal here stands in for a failure
that used to be **silent** in a different form:

  * two families with the same inductive bias produce a comparison table that looks
    complete and says nothing;
  * a grid key that is not a parameter of that family used to surface as
    `TypeError: __init__() got an unexpected keyword argument`, which names neither the
    model nor the valid keys;
  * an illegal `unit`/`strategy` pair is a random split with extra steps on one side and a
    leaked entity on the other - both still return a number;
  * an empty `reason` still renders and still paginates.

The validator is exhaustive rather than fail-fast, so one test below checks that a plan
with several faults reports all of them - a plan repaired one error per run costs one run
per error.
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from models import make_model, model_names
from plan import (BLOCK_KEYS, NARRATIVE_FIELDS, SCHEMA_VERSION, TOP_KEYS, PlanError,
                  load_plan, plan_to_decisions, validate_plan)


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
    except Exception as e:
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, nothing was raised")


class _Skip:
    """Marker for the `over` dicts: remove this key rather than replace it."""


SKIP = _Skip()


def valid_plan() -> dict:
    """A plan that passes. Every test below is this object with one thing changed."""
    def model(family, scale, scale_reason, stopping, overfitting_control, grid, reason):
        return {"family": family, "scale": scale, "scale_reason": scale_reason,
                "stopping": stopping, "overfitting_control": overfitting_control,
                "grid": grid, "reason": reason,
                "alternatives_considered": ["the other family - rejected for a reason"]}

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "data/train.csv",
        "test_dataset": None,
        "seed": 42,
        "target": {"column": "label", "reason": "two classes, and the name says so",
                   "alternatives_considered": ["region - 21 levels reads as a feature"]},
        "cleaning": {
            "drop_columns": [], "drop_rows_with_missing_label": True,
            "rarest_class_action": "keep",
            "outlier_policy": {"method": "keep"},
            "reason": "0.01% of cells are missing; imputation inside the pipeline covers it",
            "alternatives_considered": ["drop every row with a missing cell"],
        },
        "models": [
            model("logistic_regression", True,
                  "effectively required - the solver weighs inputs by magnitude",
                  "a fixed iteration cap on the solver",
                  "an L2 penalty whose strength the search chooses",
                  {"C": [0.1, 1.0]},
                  "82 columns wide, so a linear boundary is the low-variance end"),
            model("random_forest", False,
                  "not required - trees split on order, so units do not matter",
                  "grown to a bounded depth rather than to purity",
                  "the depth cap and the bootstrap sample together",
                  {"max_depth": [6, 12]},
                  "region has 21 levels and the forest can split on them directly"),
        ],
        "validation": {
            "unit": "row", "strategy": "stratified_kfold", "test_size": 0.2,
            "val_fraction": 0.2, "n_splits": 5, "gap": None, "group_column": None,
            "time_column": None,
            "reason": "no column repeats an entity and none parses as a date",
            "alternatives_considered": ["a single holdout - one estimate instead of five"],
        },
        "tuning": {
            "metric": "f1_macro", "final_fit_scope": "pool",
            "final_fit_reason": "every row is validated against exactly once under k-fold",
            "reason": "3.2:1 balance, so accuracy rewards leaning on the majority class",
            "alternatives_considered": ["accuracy - a majority predictor already scores 0.76"],
        },
        "primary_metric": {
            "name": "f1_macro", "rejected": "accuracy",
            "reason": "the two classes are treated as equally costly",
            "rejected_reason": "leading with accuracy lets a constant prediction look fine",
        },
        "report": {
            "student_id": "G2500000A", "student_name": "A Student", "variant": "Variant-2",
            "model_version": "driver.py @ schema 1", "harness": "opencode",
            "repository": "https://github.com/example/skill",
            "dataset_label": "train.csv",
        },
        "narrative": {
            "candidate_models": "Two families with different inductive bias.",
            "feature_engineering": "Imputation and scaling inside the pipeline.",
            "cleaning_why": [{"issue": "missing labels", "action": "rows dropped",
                              "why": "an unscorable row"}],
            "validation_argument": "Rows are independent, so the unit is the row.",
            "primary_metric_argument": "Macro-F1 weights the two classes equally.",
            "performance_comparison": "The gap is small against the fold spread.",
            "limitations": "The grids were narrow and the search was not exhaustive.",
        },
    }


def plan_with(**over) -> dict:
    """`valid_plan()` with dotted-path overrides. `SKIP` removes a key.

    Paths look like `models.0.grid`, `cleaning.outlier_policy`, `narrative.limitations`.
    """
    plan = copy.deepcopy(valid_plan())
    for path, value in over.items():
        parts = path.split(".")
        node = plan
        for p in parts[:-1]:
            node = node[int(p)] if isinstance(node, list) else node[p]
        if value is SKIP:
            node.pop(int(parts[-1])) if isinstance(node, list) else node.pop(parts[-1])
        else:
            node[parts[-1]] = value
    return plan


def problems_of(plan) -> str:
    """The validator's message for this plan, asserting it is refused at all."""
    e = raises(lambda: validate_plan(plan), PlanError)
    return str(e)


# --------------------------------------------------------------- the two models


def test_a_plan_with_one_model_is_refused():
    """One model still renders a two-column comparison table; the second column is blank."""
    plan = plan_with(models=[valid_plan()["models"][0]])
    msg = problems_of(plan)
    assert "models: 1 given" in msg, msg
    assert "exactly 2 required" in msg, msg


def test_two_models_of_the_same_family_are_refused():
    """`evaluate_pair` refuses this too - but only after both have been fitted and tuned."""
    plan = plan_with(**{"models.1.family": "logistic_regression",
                        "models.1.grid": {"C": [1.0]}})
    assert "both entries are 'logistic_regression'" in problems_of(plan)


def test_an_unknown_family_is_refused_and_the_message_lists_the_real_ones():
    msg = problems_of(plan_with(**{"models.0.family": "xgboost"}))
    assert "models[0].family: 'xgboost' is not a known family" in msg, msg
    for name in model_names():
        assert name in msg, (name, msg)


def test_a_grid_key_the_family_does_not_have_is_refused():
    """scikit-learn's own error names neither the model nor the valid keys; this one does.

    Before this check the typo surfaced as
    `TypeError: __init__() got an unexpected keyword argument 'n_estimator'`, several
    steps later and with no hint that the family was the problem.
    """
    plan = plan_with(**{"models.1.grid": {"n_estimator": [300]}})
    msg = problems_of(plan)
    assert "grid['n_estimator']: random_forest has no such parameter" in msg, msg
    assert "n_estimators" in msg, msg


def test_a_grid_with_no_values_to_try_is_refused():
    msg = problems_of(plan_with(**{"models.0.grid": {"C": []}}))
    assert "needs a non-empty list of values" in msg, msg


# --------------------------------------------------------------- the decisions


def test_the_target_inside_drop_columns_is_refused():
    plan = plan_with(**{"cleaning.drop_columns": ["label"]})
    assert "is also listed in cleaning.drop_columns" in problems_of(plan)


def test_an_empty_reason_is_refused():
    msg = problems_of(plan_with(**{"models.0.reason": "   "}))
    assert "models[0].reason: missing or empty" in msg, msg


def test_a_decision_with_no_alternatives_is_refused():
    """A choice with nothing else on the table was not a decision."""
    msg = problems_of(plan_with(**{"validation.alternatives_considered": []}))
    assert "validation.alternatives_considered: missing or empty" in msg, msg


def test_a_rejected_metric_that_is_the_chosen_one_is_refused():
    plan = plan_with(**{"primary_metric.rejected": "f1_macro"})
    assert "reads as a decision that was never made" in problems_of(plan)


def test_a_primary_metric_outside_the_menu_is_refused():
    msg = problems_of(plan_with(**{"primary_metric.name": "auc"}))
    assert "primary_metric.name: 'auc' is not one of" in msg, msg


def test_a_missing_scale_reason_is_refused():
    """The registry states a strength of preference, and the report is graded on it -
    a bare boolean cannot carry 'not required' versus 'effectively required'."""
    msg = problems_of(plan_with(**{"models.1.scale_reason": SKIP}))
    assert "models[1].scale_reason: missing or empty" in msg, msg


def test_a_missing_stopping_or_overfitting_cell_is_refused():
    """Both are facts about the fit that only the decision-maker holds. Without them the
    renderer would phrase them from the tuned hyperparameters - inventing a claim."""
    msg = problems_of(plan_with(**{"models.0.stopping": SKIP,
                                   "models.1.overfitting_control": SKIP}))
    assert "models[0].stopping: missing or empty" in msg, msg
    assert "models[1].overfitting_control: missing or empty" in msg, msg


def test_a_non_boolean_scale_is_refused():
    msg = problems_of(plan_with(**{"models.0.scale": "yes"}))
    assert "models[0].scale: missing or not true/false" in msg, msg


def test_a_report_repository_that_is_not_a_url_is_refused():
    plan = plan_with(**{"report.repository": "github.com/example/skill"})
    assert "report.repository: 'github.com/example/skill' is not a URL" in problems_of(plan)


# --------------------------------------------------------------- the split


def test_a_group_strategy_paired_with_the_row_unit_is_refused():
    """Both pairings fail silently: a group splitter on independent rows is a random split
    with extra steps, and a row splitter on repeated entities leaks the entity."""
    msg = problems_of(plan_with(**{"validation.strategy": "stratified_group_kfold",
                                   "validation.unit": "row",
                                   "validation.n_splits": 5}))
    assert "splits at the 'entity' unit, but unit is 'row'" in msg, msg


def test_a_group_strategy_without_a_group_column_is_refused():
    """The splitter cannot group rows it cannot name a column for - and `tune` would fall
    back to splitting without groups, which still returns a score."""
    plan = plan_with(**{"validation.unit": "entity",
                        "validation.strategy": "grouped_holdout",
                        "validation.n_splits": None,
                        "validation.group_column": None})
    msg = problems_of(plan)
    assert "validation.group_column: required for 'grouped_holdout'" in msg, msg


def test_a_temporal_strategy_needs_its_gap_and_its_column():
    plan = plan_with(**{"validation.unit": "time", "validation.strategy": "temporal",
                        "validation.n_splits": 5, "validation.gap": None,
                        "validation.time_column": None})
    msg = problems_of(plan)
    assert "validation.gap: required for 'temporal'" in msg, msg
    assert "validation.time_column: required for 'temporal'" in msg, msg


def test_a_holdout_with_n_splits_is_refused():
    """A holdout takes one estimate, not folds; carrying n_splits reads as a design that
    was chosen and then not used."""
    plan = plan_with(**{"validation.strategy": "holdout", "validation.n_splits": 5})
    msg = problems_of(plan)
    assert "is a holdout - it takes one estimate, not folds" in msg, msg


def test_a_group_column_given_to_a_row_strategy_is_refused():
    plan = plan_with(**{"validation.group_column": "region"})
    msg = problems_of(plan)
    assert "does not use it; set it to null" in msg, msg


def test_a_test_size_outside_the_open_unit_interval_is_refused():
    assert "is outside [0.01, 0.99]" in problems_of(
        plan_with(**{"validation.test_size": 1.0}))


# --------------------------------------------------------------- no code in a plan


def test_a_code_key_anywhere_in_the_plan_is_refused():
    """The hole this design exists to close: a plan that could carry code would put the
    guards back inside the library, where a hand-rolled run never reaches them."""
    for where in ("target", "narrative"):
        plan = plan_with(**{f"{where}.code": "import sklearn"})
        msg = problems_of(plan)
        assert f"{where}.code: not allowed" in msg, msg
        assert "A plan carries decisions, never code" in msg, msg


def test_a_custom_prefixed_key_is_refused_at_any_depth():
    plan = plan_with(**{"models.0.custom_scaler": "StandardScaler()"})
    msg = problems_of(plan)
    assert "custom_scaler: not allowed" in msg, msg


def test_a_forbidden_key_nested_in_a_list_entry_is_refused():
    plan = plan_with(**{"narrative.cleaning_why": [
        {"issue": "x", "action": "y", "why": "z", "post_hook": "..."}]})
    msg = problems_of(plan)
    assert "cleaning_why[0].post_hook: not allowed" in msg, msg


def test_an_unknown_key_is_refused_rather_than_ignored():
    """Ignoring it would accept the misspelling and silently drop the decision it carried."""
    msg = problems_of(plan_with(**{"tuning.metricc": "f1_macro"}))
    assert "tuning: unknown key(s) ['metricc']" in msg, msg


# --------------------------------------------------------------- the contract itself


def test_a_valid_plan_passes():
    validate_plan(valid_plan())                                # must not raise


def test_every_problem_is_reported_at_once():
    """Exhaustive, not fail-fast: a plan repaired one error per run costs one run per
    error, and these are cheap to find together."""
    plan = plan_with(**{"models.0.scale": "yes",
                        "models.1.reason": "",
                        "validation.n_splits": None,
                        "primary_metric.name": "nope",
                        "narrative.limitations": ""})
    msg = problems_of(plan)
    for expected in ("models[0].scale", "models[1].reason", "validation.n_splits",
                     "primary_metric.name", "narrative.limitations"):
        assert expected in msg, (expected, msg)


def test_a_missing_block_names_its_own_fields():
    msg = problems_of(plan_with(**{"narrative": SKIP}))
    assert "narrative: missing" in msg, msg
    for field in NARRATIVE_FIELDS:
        assert field in msg, (field, msg)


def test_a_wrong_schema_version_is_refused():
    msg = problems_of(plan_with(schema_version=0))
    assert "schema_version: expected 1, got 0" in msg, msg


def test_a_plan_that_is_not_an_object_is_refused():
    msg = problems_of(["not", "a", "plan"])
    assert "expected a JSON object, got list" in msg, msg


def test_load_plan_reports_invalid_json_as_invalid_json():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "plan.json"
        p.write_text('{"schema_version": 1,,}', encoding="utf-8")
        e = raises(lambda: load_plan(p), PlanError, contains="not valid JSON")
        assert "line" in str(e), str(e)


def test_load_plan_names_the_two_step_workflow_when_there_is_no_plan():
    with tempfile.TemporaryDirectory() as d:
        e = raises(lambda: load_plan(Path(d) / "plan.json"), PlanError, contains="no plan")
        assert "driver.py prepare" in str(e), str(e)


# ------------------------------------------------ the schema and the validator together


def test_the_schema_and_the_validator_agree_on_every_block():
    """`plan.py` is what enforces the contract; `plan.schema.json` is what an editor reads
    and what the marker is pointed at. If the two drift, one of them is lying.

    The comparison allows the schema to leave a field out of `required` - `n_splits`,
    `gap`, `group_column` and `time_column` are required only by certain strategies - but
    not to name a field the validator does not have, nor to omit one it does.
    """
    schema = json.loads((ROOT / "schemas" / "plan.schema.json").read_text(encoding="utf-8"))
    props = schema["properties"]

    blocks = {"target": "target", "cleaning": "cleaning", "validation": "validation",
              "tuning": "tuning", "primary_metric": "primary_metric",
              "report": "report", "narrative": "narrative"}
    for key, block in blocks.items():
        allowed = set(BLOCK_KEYS.get(block, ())) or set(
            props[key].get("properties", {})) | set(props[key].get("required", []))
        if block in BLOCK_KEYS:
            declared = set(props[key]["properties"])
            assert set(BLOCK_KEYS[block]) == declared, (
                f"{block}: plan.py has {sorted(set(BLOCK_KEYS[block]) ^ declared)} "
                f"that the schema does not agree on")
        for name in props[key].get("required", []):
            assert name in allowed, (key, name)

    item_required = set(props["models"]["items"]["required"])
    assert item_required == set(BLOCK_KEYS["model"]), (
        "models items: "
        f"{sorted(item_required ^ set(BLOCK_KEYS['model']))} disagree between the schema "
        f"and plan.py")
    assert set(props["models"]["items"]["properties"]) == set(BLOCK_KEYS["model"])

    assert set(schema["required"]) <= set(TOP_KEYS), (
        f"the schema requires {sorted(set(schema['required']) - set(TOP_KEYS))}, which "
        f"plan.py does not know about")
    assert set(props["cleaning"]["properties"]["outlier_policy"]["properties"]) == \
        set(BLOCK_KEYS["outlier_policy"])


# --------------------------------------------------------------- the audit record


def test_the_decision_log_has_one_entry_per_decision():
    """`make_pdf` checks `decisions.json` for step/decision/choice/reason/alternatives,
    and a missing entry leaves a report section with nothing behind it."""
    log = plan_to_decisions(valid_plan()).to_dict()
    entries = log["decisions"]
    steps = [e["step"] for e in entries]
    for expected in ("label", "cleaning", "model_selection", "preprocessing", "validation",
                     "metrics", "final_fit"):
        assert expected in steps, (expected, steps)
    for e in entries:
        for field in ("step", "decision", "choice", "reason", "alternatives_considered"):
            assert e.get(field), (field, e)
    assert log["_evidence"]["producer"] == "DecisionLog"


def test_the_decision_log_carries_the_per_model_facts_the_plan_chose():
    """The log is derived from the plan, so a decision in the report that the run did not
    take is not a mistake this design can make."""
    entries = plan_to_decisions(valid_plan()).to_dict()["decisions"]
    selection = next(e for e in entries if e["step"] == "model_selection")
    assert selection["inputs"]["families"] == ["logistic_regression", "random_forest"]
    assert set(selection["inputs"]["stopping"]) == {"logistic_regression", "random_forest"}
    final = next(e for e in entries if e["step"] == "final_fit")
    assert final["inputs"]["final_fit_scope"] == "pool"


def test_a_grid_key_is_checked_against_the_family_itself_not_a_list_of_names():
    """The check reads `make_model(family)().get_params()`, so it stays right when a
    family is added or scikit-learn renames a parameter."""
    for family in model_names():
        params = set(make_model(family).get_params())
        assert params, family
        plan = plan_with(**{"models.0.family": family,
                            "models.0.grid": {"definitely_not_a_param": [1]}})
        if plan["models"][1]["family"] == family:
            plan["models"][1]["family"] = next(f for f in model_names() if f != family)
            plan["models"][1]["grid"] = {"max_depth": [3]} if family != "random_forest" \
                else {"C": [1.0]}
            plan["models"][1] = {**plan["models"][1], "family": "decision_tree",
                                 "grid": {"max_depth": [3]}}
        assert "definitely_not_a_param" in problems_of(plan), family


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
