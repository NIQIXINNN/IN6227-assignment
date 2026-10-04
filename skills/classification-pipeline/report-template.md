# The report's structure

**This file is a specification, not a form.** Nothing here is filled in by hand: the driver
renders `out/report.md` from your `plan.json` and the evidence, and `make_pdf.py` turns that
into `out/report.pdf`. Edit the plan, not the report — a hand-edited `report.md` is
overwritten by the next `run`, and a number typed in by hand is a number nothing produced.

It is written down because the structure is fixed and a marker should be able to see what
was asked for. Each section below names where its content comes from, so a maintainer
changing `scripts/report.py` can check it against this file and a reader can tell which
part of the page is the model's argument and which is the run's arithmetic.

Two rules the renderer enforces:

- **The whole PDF is ≤ 2 pages, identity header included.** The renderer warns, exits
  non-zero, and `driver.py run` deletes the PDF if it overflows. Trim prose; do not shrink
  the font.
- **Every number on the page is traceable to a file.** Section 1's counts come from
  `evidence/eda.json` and `evidence/cleaning.json`, section 4's from
  `evidence/metrics.json` and `evidence/validation.json`. The prose comes from
  `plan.narrative`. Nothing is computed at render time.

---

## Header (page 1)

Drawn by `make_pdf.py` from the plan's `report` block, so it costs no separate page:

| Field | Plan field |
|---|---|
| Matriculation number | `report.student_id` |
| Name | `report.student_name` |
| Model & version | `report.model_version` |
| LLM interface | `report.harness` |
| Variant | `report.variant` |
| Dataset | `report.dataset_label` |

Declare the model and interface that **actually produced this report**. The examples in the
assignment brief are not your values — a false declaration here is worse than none, because
this is the one field whose entire purpose is to prevent exactly that.

---

## 1. Data exploration and cleaning

**Dataset.** rows / features / classes, the label column and its distribution —
`evidence/eda.json`.

**Issues found.** missing values, duplicates, outliers, imbalance ratio (majority ÷
minority) — `evidence/eda.json`.

**Cleaning decisions.** one row per action actually taken, produced from
`evidence/cleaning.json` and the plan's `narrative.cleaning_why`:

| Issue | Action | Why |
|---|---|---|
| from `cleaning.json` | from `cleaning.json` | from `narrative.cleaning_why` |

## 2. Feature selection / engineering

`narrative.feature_engineering` — what was constructed, and one line of rationale each. If
nothing was constructed, say why nothing was needed. Columns held out of the model are
named in `cleaning.drop_columns` and appear here.

## 3. Model training

**Candidate-model reasoning.** `narrative.candidate_models`: the chain a marker has to be
able to follow —

- (a) every family considered, with what `evidence/candidate_screen.json` measured for it —
  each is fitted on one protocol, so the comparison is like-for-like;
- (b) each rejection and the number that killed it. The shortlist's floor is one standard
  deviation below the best mean, so anything under it is rejected by arithmetic, not by
  argument. A family on `excluded_without_fitting` or `errored` was **not measured**: say
  that, and say why, rather than describing it as eliminated;
- (c) the two that survived, and why their inductive biases differ. Both are from the
  shortlist; if it held fewer than two, say so.

| | Model A | Model B |
|---|---|---|
| Estimator | `models[0].family` | `models[1].family` |
| Hyperparameters | the winner from `evidence/tuning.json` | same |
| Tuned on | the plan's validation design, and `tuning.metric` | same |
| Validation score | `evidence/tuning.json` | same |
| Stopping criterion | `models[i].stopping` | same |
| Overfitting control | `models[i].overfitting_control` | same |
| Preprocessing applied | `models[i].scale` + `scale_reason`, and the encoding the driver applies | same |

**Fixed configuration (not tuned):** `seed`, `test_size`, `val_fraction`, `n_splits`,
`tuning.metric`, the test protocol — all read from the plan, printed so a marker can see
which knobs were *not* searched.

## 4. Evaluation and comparison

**Split protocol** — row counts and where the test rows came from, from the run.

**Validation strategy** — `narrative.validation_argument`, plus the arithmetic the driver
recorded in `evidence/validation.json`: the rarest class's rows in the pool, how many of
them land in each estimate, what one flipped row moves, the class counts (a mean per fold
under k-fold, which is why it need not be a whole number), and what the choice cost. Under
a group or time strategy the same arithmetic is reported at that unit — entities or windows
rather than rows.

**Validation used for:** hyperparameter selection. **Final model fitted on:**
`tuning.final_fit_scope`, with `tuning.final_fit_reason`.

| Metric | Model A | Model B |
|---|---|---|
| Accuracy | `evidence/metrics.json` | same |
| Precision (macro) | same | same |
| Recall (macro) | same | same |
| F1 (macro) | same | same |
| ROC-AUC / PR-AUC | same | same |

The row for `primary_metric.name` is marked as the headline. If `roc_auc` or `pr_auc` is
null, its note is printed beside it — never a null without its note.

**Confusion matrices:** `figures/confusion_pair.png`, both models side by side in one
figure. Two separate figures do not fit the page budget, and side by side is how a marker
compares them anyway.

**Primary metric and rationale.** `narrative.primary_metric_argument` — the four inputs
that decided it: the task objective; the class distribution, with the actual imbalance
ratio; the cost of each error type; and whether this validation design can estimate it
stably. Then `primary_metric.rejected` and `rejected_reason`.

## 5. Findings and discussion

**Performance comparison.** `narrative.performance_comparison` — both numbers on the
primary metric, the gap between them, the variability across folds or splits, and what the
gap is worth. *A difference smaller than the spread is not a win, and saying so is worth
more than the win.*

**Limitations.** `narrative.limitations` — the violated assumption (independence,
linearity, distance concentration), what the validation design could not estimate, and what
you would change on a re-run.

---

## Repository

`report.repository` — the link to this skill. The clone form ending in `.git` is the same
place; write it without the suffix in a report. If you were not given one, ask the user
rather than writing a plausible-looking address: the validator refuses a value that is not
a URL, and it cannot tell a real one from an invented one.

---

<!--
## Reflection — written by the human, NOT by this skill, and NOT part of the two pages.

The skill does not produce this and does not render it. It is here so the 20% does not get
forgotten. Three things it is marked on:

1. Human oversight     — where you checked, corrected or overruled the model's output, and
                         what made you look.
2. Critical evaluation — which of its decisions you would defend and which you would not,
                         with the reason.
3. Trustworthiness     — what a reader should treat with caution, and what would change your
                         confidence.

Write it yourself. A reflection that reads like the model wrote it scores as one.
-->
