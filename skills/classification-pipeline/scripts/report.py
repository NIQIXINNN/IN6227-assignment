#!/usr/bin/env python
"""Render `report.md` from the plan, the evidence and the metrics - and from nothing else.

Three sources, and no fourth:

  * **the plan** - the decisions, and the argument for them (`plan["narrative"]`);
  * **the evidence** - `eda.json`, `cleaning.json`, `validation.json`, `tuning.json`, every
    one of them written by `driver.py` during this run;
  * **`metrics.json`** - the two models' scores on the held-out test set.

Nothing here is composed from imagination, and that is the reason the renderer moved into
the skill. A renderer can only print a number it was handed: the report cannot cite a
figure the run did not compute, and a piece of evidence that is missing **raises** rather
than leaving a blank - because a blank cell reads as "nothing to report", which is the same
silent wrongness as a fabricated one.

What is deliberately not generated: the identity block (drawn by `make_pdf.identity_header`
from the plan's `report` block, so it shares page 1 instead of costing a cover sheet) and
the Reflection (the human writes it, and it is not part of the two pages).

The markdown subset is exactly what `make_pdf.parse` understands: `#`/`##`/`###` headings,
paragraphs, `-` bullets, pipe tables, images (`![alt](path)`, optionally with a `label:` in
front of it), `---` rules and `<!-- -->` comments. A construct outside that subset renders as
a paragraph - except image syntax, which renders as a visible `[unplaceable figure: ...]`
marker. Prose is the wrong answer for that one case: a paragraph is indistinguishable from a
report that simply has no figure, which is how a captioned `![...](...)` line stayed missing
from this skill's own report through several runs.
"""

from __future__ import annotations

from pathlib import Path

from models import MODEL_REGISTRY


class ReportError(RuntimeError):
    """A report slot with no evidence behind it. Raised rather than left blank."""


# How each strategy in the plan's closed menu is described in the report. This is a
# translation of an enum, not a claim: the strategy names are fixed in `plan.py`, and a
# reader of the report should not have to know them.
STRATEGY_PROSE = {
    "holdout": "a single holdout",
    "stratified_kfold": "stratified k-fold cross-validation",
    "grouped_holdout": "a single holdout of whole entities",
    "stratified_group_kfold": "stratified, entity-disjoint k-fold cross-validation",
    "temporal": "rolling-origin folds over the time-ordered rows",
}

UNIT_PROSE = {"row": "row", "entity": "entity", "time": "time window"}

# Row order of the report's metric table, and the key each row reads from `evaluate`.
METRIC_ROWS = (
    ("Accuracy", "accuracy"),
    ("Precision (macro)", "precision_macro"),
    ("Recall (macro)", "recall_macro"),
    ("F1 (macro)", "f1_macro"),
)
AUC_ROW = "ROC-AUC / PR-AUC"


# ------------------------------------------------------------------------ small helpers


def _need(mapping, key: str, where: str = "evidence"):
    value = mapping.get(key)
    if value is None:
        raise ReportError(
            f"the report needs `{key}` from {where} and it is not there. Every sentence in "
            f"the report is generated from something the run produced, so a missing piece "
            f"is a missing step, not a gap to fill in prose."
        )
    return value


def _int(x) -> str:
    return f"{x:,}" if isinstance(x, int) and not isinstance(x, bool) else str(x)


def _num(x, places: int = 4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    return f"{x:.{places}f}"


def _rate(x) -> str:
    """A fraction as a percentage, or n/a."""
    return "n/a" if x is None else f"{x:.2%}"


def _cell(text) -> str:
    """One table cell: no pipes and no newlines, either of which breaks the row."""
    return " ".join(str(text).split()).replace("|", "/") or "-"


def _sentence(text: str) -> str:
    """End a spliced fragment with a full stop, so the next sentence does not run into it.

    The narrative is the plan's prose, so it arrives with whatever punctuation its author
    used. `Chosen because <argument> Validation class counts: ...` reads as one broken
    sentence when the argument was left unterminated, and that is a defect a grader sees
    without being able to say where it came from.
    """
    text = str(text).strip()
    return text if text.endswith((".", "!", "?", ":")) else text + "."


def _lead(text: str) -> str:
    """A fragment that opens a paragraph: sentence-cased as well as terminated.

    Kept separate from `_sentence` on purpose - a fragment spliced mid-sentence must not be
    capitalised, and one that starts a paragraph and begins "the gap is 0.02" reads as a
    dropped line rather than a sentence.
    """
    text = _sentence(text)
    return text[0].upper() + text[1:] if text else text


def _rows(n) -> str:
    return f"{_int(n)} row" if n == 1 else f"{_int(n)} rows"


def _ratio(x) -> str:
    return f"{x:.2f}:1"


def _count(x) -> str:
    """A count that may be fractional - `validation_report` rounds to one decimal."""
    if isinstance(x, float):
        return f"{x:,.1f}".removesuffix(".0")
    return _int(x)


def _table(header, rows) -> list[str]:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines


def _basename(path) -> str:
    return Path(str(path)).name


# ------------------------------------------------------------------- section 1: the data


def _dataset_line(plan, eda, cleaning) -> str:
    counts = eda.get("class_counts") or {}
    distribution = " / ".join(f"`{k}` {_int(v)}" for k, v in counts.items())
    n_num = len(eda.get("numeric_features") or [])
    n_cat = len(eda.get("categorical_features") or [])
    excluded = cleaning.get("columns_excluded") or []

    text = (f"`{_basename(plan['dataset'])}` - {_int(eda.get('n_rows', 0))} rows, "
            f"{_int(eda.get('n_features', 0))} feature columns. The label is "
            f"`{eda.get('label_column')}` with {_int(eda.get('n_classes', 0))} classes, "
            f"distributed {distribution}")
    ratio = eda.get("imbalance_ratio")
    if ratio is not None:
        text += (f" - a majority:minority ratio of {_ratio(ratio)}, which is what the choice "
                 f"of metric in section 4 turns on")
    else:
        text += " - a single class, so no imbalance ratio is defined"
    text += (f". {n_num} numeric and {n_cat} categorical columns were modelled")
    if excluded:
        text += (f"; {len(excluded)} further column(s) were held out of the feature set: "
                 + ", ".join(f"`{c}`" for c in excluded))
    elif cleaning.get("columns_dropped_id_like"):
        text += ("; one identifier-like column was held out automatically "
                 f"({', '.join('`%s`' % c for c in cleaning['columns_dropped_id_like'])})")
    else:
        text += "; no column was held out"
    return text + "."


def _issues(plan, eda, cleaning) -> str:
    label = eda.get("label_column")
    per_column = eda.get("missing_per_column") or {}
    parts = []

    if per_column:
        worst = max(per_column.items(), key=lambda kv: kv[1])
        parts.append(f"missing values: {_int(eda.get('missing_total', 0))} cells across "
                     f"{len(per_column)} column(s), worst `{worst[0]}` ({_int(worst[1])})")
    else:
        parts.append("missing values: none")
    if per_column.get(label):
        parts.append(f"rows with no label: {_int(per_column[label])}")

    duplicates = eda.get("duplicate_rows") or 0
    parts.append(f"duplicate rows: {_int(duplicates)}" if duplicates
                 else "duplicate rows: none")

    policy = (plan.get("cleaning") or {}).get("outlier_policy") or {}
    if policy.get("method") == "iqr":
        cols = policy.get("columns")
        where = ("the numeric columns " + ", ".join(f"`{c}`" for c in cols)) if cols \
            else "every numeric column"
        removed = cleaning.get("rows_dropped_outliers")
        parts.append(
            f"outliers: {_rows(removed)} fell outside a {policy.get('factor'):g}x IQR "
            f"fence on {where} and were removed" if removed
            else f"outliers: none fell outside a {policy.get('factor'):g}x IQR fence on "
                 f"{where}")
    else:
        parts.append("outliers: kept - no row was filtered, so no fence was applied")

    ratio = eda.get("imbalance_ratio")
    if ratio is not None:
        parts.append(f"class imbalance: {_ratio(ratio)}")

    text = "; ".join(parts) + "."
    return text[0].upper() + text[1:]


# ----------------------------------------------------------------------- section 3: models


def _estimator_cell(family: str) -> str:
    spec = MODEL_REGISTRY.get(family)
    if not spec:
        return f"`{family}`"
    return f"{spec['family']} (`{spec['estimator'].__name__}` via `{family}`)"


def _hyperparameters_cell(tuned: dict) -> str:
    """The chosen values, with the pipeline's step prefix stripped.

    `tune` addresses the estimator through `build_pipeline`'s step name, so the keys come
    back as `clf__C`. `clf` is an implementation detail of this package - a reader of the
    report has no reason to know it exists, and `clf__C=1.0` reads as a parameter they
    would have to look up.
    """
    best = tuned.get("best_params") or {}
    chosen = ", ".join(f"`{k.split('__')[-1]}={v!r}`" for k, v in sorted(best.items()))
    searched = tuned.get("n_combinations")
    tail = f" - best of {searched} combination(s) scored" if searched else ""
    return f"{chosen or 'none'}{tail}"


def _preprocessing_cell(plan_model: dict) -> str:
    scale = ("median imputation, then standardised" if plan_model.get("scale")
             else "median imputation only, no scaling")
    return (f"numeric: {scale}; categorical: most-frequent imputation then one-hot "
            f"(a level unseen in training encodes as all-zeros)")


def _after_cleaning(cleaning, eda) -> str:
    """The arithmetic between "what the data looked like" and "what was modelled"."""
    parts = [f"{_int(cleaning.get('n_rows_loaded', eda.get('n_rows', 0)))} rows loaded"]
    removed = [
        (cleaning.get("rows_dropped_missing_label", 0), "with no label"),
        (cleaning.get("rows_dropped_duplicate_rarest", 0),
         "duplicate rows inside the rarest class"),
        (cleaning.get("rows_dropped_outliers", 0), "outside the IQR fence"),
    ]
    dropped = [(n, why) for n, why in removed if n]
    if dropped:
        parts.append(f"{_int(sum(n for n, _ in dropped))} removed ("
                     + "; ".join(f"{_int(n)} {why}" for n, why in dropped) + ")")
    else:
        parts.append("none removed")
    parts.append(f"{_int(cleaning.get('n_rows_modelled', 0))} rows modelled, "
                 f"{cleaning.get('n_features_modelled', '?')} features")
    return "; ".join(parts) + "."


def _split_line(plan, validation) -> str:
    s = validation["split"]
    v = plan["validation"]
    if s.get("test_source") == "provided file":
        test_file = f"`{_basename(plan['test_dataset'])}`"
        if s["mode"] == "folds":
            head = (f"{_int(s['n_pool'])} pool rows from "
                    f"`{_basename(plan['dataset'])}`, validated by {v['n_splits']}-fold "
                    f"cross-validation, plus {_int(s['n_test'])} test rows from {test_file}.")
        else:
            head = (f"{_int(s['n_train'])} train / {_int(s['n_val'])} validation rows from "
                    f"`{_basename(plan['dataset'])}`, plus {_int(s['n_test'])} test rows "
                    f"from {test_file}.")
        head += (" The test file was held out from the start and opened only for the final "
                 "scoring.")
    elif s["mode"] == "folds":
        head = (f"{_int(s['n_pool'])} pool rows, validated by {v['n_splits']}-fold "
                f"cross-validation, plus {_int(s['n_test'])} test rows. The test set is "
                f"{s['test_split']} ({v['test_size']:.0%} of "
                f"`{_basename(plan['dataset'])}`), seed {plan['seed']}, and was not touched "
                f"again until the final scoring.")
    else:
        head = (f"{_int(s['n_train'])} train / {_int(s['n_val'])} validation / "
                f"{_int(s['n_test'])} test rows. The test set is {s['test_split']} "
                f"({v['test_size']:.0%} of `{_basename(plan['dataset'])}`), seed "
                f"{plan['seed']}, and was not touched again until the final scoring.")
    return head


def _final_fit_prose(plan, validation) -> str:
    s = validation["split"]
    scope = plan["tuning"]["final_fit_scope"]
    if s["mode"] == "folds":
        # With folds there is no separate validation block to add back: every row of the
        # pool is validated against exactly once, so "train" and "pool" name the same rows.
        # Saying so is better than printing a distinction the design cannot make.
        return (f"the whole pool ({_int(s['n_pool'])} rows) - with "
                f"{plan['validation']['n_splits']}-fold cross-validation every row is "
                f"validated against exactly once, so `final_fit_scope: pool` fits the same "
                f"rows as `train` would, and the test set is the only estimate untouched by "
                f"selection")
    if scope == "train":
        return (f"the training rows only ({_int(s['n_train'])} rows) - the "
                f"{_int(s['n_val'])} validation rows chose the hyperparameters but are not "
                f"in the final fit")
    return (f"the whole pool ({_int(s['n_pool'])} rows) - the {_int(s['n_val'])} validation "
            f"rows were refitted into the final model, so the validation score above is a "
            f"training score for it and the test set is the only untouched estimate")


def _val_counts(plan, validation) -> str:
    s = validation["split"]
    if s["mode"] == "folds":
        per = validation["validation_report"]["kfold"]["rows_per_class_per_estimate"]
        where = (f"mean per fold, over {plan['validation']['n_splits']} folds - a mean "
                 f"rather than a count, so it need not be a whole number")
    else:
        per = s["val_class_counts"]
        where = "in the validation set"
    return (" / ".join(f"`{k}` {_count(v)}" for k, v in per.items())
            + f" ({where})")


# -------------------------------------------------------------------- the whole document


def render_report(plan, evidence, metrics, *,
                  confusion_figure: str = "figures/confusion_pair.png") -> str:
    """The report as markdown.

    `evidence` maps a name to the parsed contents of that evidence file:
    `{"eda": ..., "cleaning": ..., "validation": ..., "tuning": ...}`. `metrics` is
    `{family: evaluate(...)}` - the two models, scored on the test set.
    """
    eda = _need(evidence, "eda")
    cleaning = _need(evidence, "cleaning")
    validation = _need(evidence, "validation")
    tuning = _need(evidence, "tuning")
    narrative = _need(plan, "narrative", "the plan")
    models = plan["models"]
    families = [m["family"] for m in models]

    absent = [f for f in families if f not in metrics]
    if absent:
        raise ReportError(
            f"metrics.json has no entry for {absent}; it holds {sorted(metrics)}. The "
            f"comparison table needs both models - which is the shape `evaluate_pair` "
            f"guarantees, so a missing one means the report was not built from that."
        )

    v = plan["validation"]
    t = plan["tuning"]
    tuned_models = tuning.get("models") or {}
    per_model_tuned = [_need(tuned_models, f, "tuning.json") for f in families]
    protocol = _need(tuning, "protocol", "tuning.json")

    out: list[str] = []

    # -- cover: skipped by the renderer, kept so report.md stands on its own ------------
    r = _need(plan, "report", "the plan")
    out += ["## Cover", "",
            "This section is skipped by the renderer - the identity block is drawn as the "
            "page-1 header from these same values, so it costs no separate page.", ""]
    out += [f"- **Matric number:** {r['student_id']}",
            f"- **Full name:** {r['student_name']}",
            f"- **Variant:** {r['variant']}",
            f"- **Model name & version:** {r['model_version']}",
            f"- **LLM interface:** {r['harness']}",
            f"- **Dataset:** {r['dataset_label']}",
            f"- **Repository:** {r['repository']}", ""]

    # -- 1 ----------------------------------------------------------------------------
    out += ["---", "", "## 1. Data exploration and cleaning", "",
            f"**Dataset.** {_dataset_line(plan, eda, cleaning)}", "",
            f"**Issues found.** {_issues(plan, eda, cleaning)}", "",
            "**Cleaning decisions.**", ""]
    out += _table(("Issue", "Action", "Why"),
                  [(row["issue"], row["action"], row["why"])
                   for row in narrative["cleaning_why"]])
    out += ["", f"**After cleaning.** {_after_cleaning(cleaning, eda)}", ""]

    # -- 2 ----------------------------------------------------------------------------
    out += ["## 2. Feature selection / engineering", "",
            _lead(narrative["feature_engineering"]), ""]

    # -- 3 ----------------------------------------------------------------------------
    out += ["## 3. Model training", "",
            f"**Candidate-model reasoning.** {_lead(narrative['candidate_models'])}", ""]
    rows = [
        ("Estimator", *[_estimator_cell(f) for f in families]),
        ("Hyperparameters", *[_hyperparameters_cell(tm) for tm in per_model_tuned]),
        ("Tuned on", protocol, protocol),
        ("Validation score",
         *[f"{_num(tm.get('best_score'))} ({t['metric']})" for tm in per_model_tuned]),
        ("Stopping criterion", *[m["stopping"] for m in models]),
        ("Overfitting control", *[m["overfitting_control"] for m in models]),
        ("Preprocessing applied", *[_preprocessing_cell(m) for m in models]),
    ]
    out += _table(("", "Model A", "Model B"), rows)
    out += ["", "**Preprocessing, per model.**", ""]
    out += [f"- `{m['family']}`: {_sentence(m['scale_reason'])}" for m in models]
    split = validation["split"]
    test_clause = (f"test set is the labelled file `{_basename(plan['test_dataset'])}`, "
                   f"held aside from the start"
                   if plan.get("test_dataset") else
                   f"test fraction {v['test_size']:.0%} of the frame")
    val_clause = (f"validation by {v['n_splits']}-fold cross-validation over the pool"
                  if split["mode"] == "folds" else
                  f"validation holds out {v['val_fraction']:.0%} of "
                  + ("the pool" if plan.get("test_dataset") else "the remainder"))
    out += ["",
            f"**Fixed configuration (not tuned):** seed {plan['seed']}; {test_clause}; "
            f"{val_clause}; tuning metric `{t['metric']}`; "
            f"primary metric `{plan['primary_metric']['name']}`.", ""]

    # -- 4 ----------------------------------------------------------------------------
    s = validation["split"]
    cost = validation["validation_report"]["kfold" if s["mode"] == "folds" else "holdout"]["cost"]

    out += ["## 4. Evaluation and comparison", "",
            f"Split protocol: {_split_line(plan, validation)}", "",
            f"Validation strategy: {STRATEGY_PROSE[v['strategy']]}, splitting at the "
            f"{UNIT_PROSE[v['unit']]} unit. Chosen because "
            f"{_sentence(narrative['validation_argument'])} Validation class counts: "
            f"{_val_counts(plan, validation)}. Cost of the choice: {cost}.", "",
            f"Validation used for: hyperparameter selection. Final model fitted on: "
            f"{_final_fit_prose(plan, validation)}.", ""]

    primary = plan["primary_metric"]["name"]
    metric_rows = []
    for label, key in METRIC_ROWS:
        mark = " (primary)" if key == primary else ""
        metric_rows.append((label + mark, *[_num(metrics[f].get(key)) for f in families]))
    auc_cells = []
    for f in families:
        m = metrics[f]
        auc_cells.append(f"{_num(m.get('roc_auc'))} / {_num(m.get('pr_auc'))}")
    mark = " (primary)" if primary in ("roc_auc", "pr_auc") else ""
    metric_rows.append((AUC_ROW + mark, *auc_cells))
    out += _table(("Metric", "Model A", "Model B"), metric_rows)

    auc_notes = sorted({(metrics[f].get("roc_auc_note") or metrics[f].get("pr_auc_note"))
                        for f in families} - {None})
    if auc_notes:
        out += ["", "AUC notes: " + "; ".join(auc_notes) + "."]

    out += ["", f"Confusion matrices: ![{_cell('confusion matrices, one panel per model')}]"
                f"({confusion_figure})", "",
            f"**Primary metric and rationale.** `{primary}`. "
            f"{_sentence(narrative['primary_metric_argument'])} "
            f"`{plan['primary_metric']['rejected']}` was not led with: "
            f"{_sentence(plan['primary_metric']['rejected_reason'])}", ""]

    # -- 5 ----------------------------------------------------------------------------
    # Both paragraphs are labelled, and `limitations` is labelled in-line rather than as
    # its own heading: unlabelled it read as a continuation of the comparison, and the
    # reader looking for what this design cannot support - which the rubric asks for -
    # had to guess where the comparison stopped. In-line costs a few characters of the
    # same paragraph instead of a line of the page budget.
    out += ["## 5. Findings and discussion", "",
            f"**Performance comparison.** {_lead(narrative['performance_comparison'])}", "",
            f"**Limitations.** {_lead(narrative['limitations'])}", ""]

    # -- repository --------------------------------------------------------------------
    out += ["---", "", "## Repository", "", r["repository"], ""]

    return "\n".join(out) + "\n"
