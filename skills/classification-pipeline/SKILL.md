---
name: tabular-classification-report
description: Runs a complete tabular classification workflow on any dataset path - profiles and cleans the data, selects and trains two classifiers, evaluates both on one shared protocol, and emits a PDF report of at most two pages. Use when the user supplies a tabular dataset and asks for classification, model training, model comparison, or an auto-generated data-mining report.
---

# Tabular Classification Report

Turns a dataset path into a set of justified decisions plus a short PDF report.

**You write one file: `plan.json`. The skill executes it.**

That is the whole interface. You do not write a script, you do not import the modules, you
do not call the primitives. You read evidence, decide, and record the decisions as fields
in a plan; `driver.py` validates the plan and does the work. Two commands:

```bash
python "<skill>/driver.py" prepare --dataset <csv> --experiment <dir> [--test-dataset <csv>] [--label <column>]
python "<skill>/driver.py" run     --experiment <dir>
```

**Why it is built this way.** Every choice in this workflow is yours: which column is the
label, which two models, what preprocessing each one needs, how validation is formed, how
wide the grids are. `scripts/` holds only *mechanics* — the parts that **fail silently**
when they are got wrong. It has no defaults and no policy to fall back on, because a default
that suits one dataset is wrong for the next.

The split is deliberate, and worth a line in the report:

> **The plan records what you decided; the driver decides nothing.**

Mechanics are the things where a mistake produces a plausible number rather than an error:
macro-F1 versus micro-F1, imputation fitted before the split instead of inside it, a split
that quietly forgets to stratify, a random split of rows that are not independent.
Decisions are the things a marker can read and judge: which models, which strategy, how
many folds. A decision made at run time can be justified; one baked into a constant cannot.

Earlier versions of this skill asked the model to write an orchestration script that called
the primitives in order. When a run went wrong it went wrong *invisibly*: the script wrote
its output beside the machinery, or hand-rolled its own scikit-learn code and never
imported this package at all, so none of the guards inside `scripts/` were ever reached —
a locksmith's door in a building nobody enters. A guard can only refuse a call that is
made. So the entry point moved to a driver that always takes the same path, and the plan
contract refuses, before any work starts, anything that would take a different one.

## What you have

| File | What it is |
|---|---|
| `driver.py` | **The entry point.** `prepare` and `run`. Read its module docstring first. |
| `schemas/plan.schema.json` | The plan's shape, for editor completion. The validator in `scripts/plan.py` is the authority. |
| `SKILL.md` | this file — every decision, and the reasoning behind it |
| `reference.md` | Decision tables: model families, preprocessing, metrics, validation. **Read before Step 4.** |
| `scripts/plan.py` | the plan contract: `validate_plan`, `load_plan`, `plan_to_decisions` |
| `scripts/report.py` | renders `report.md` from the evidence + your plan's narrative |
| `scripts/make_pdf.py` | Markdown → ≤ 2-page PDF, identity header drawn on page 1 |
| `scripts/data.py`, `models.py`, `validation.py`, `preprocessing.py`, `evaluation.py`, `pipeline.py`, `decisions.py` | the mechanics: loading, profiling, model registry, splitters, leak-safe preprocessing, metrics, tuning, the decision log |

Where does the report's *prose* come from? From your plan's `narrative` block, spliced in
verbatim. So the report can never disagree with the run: you write the argument once, and
the numbers that sit next to it are the ones the run produced.

What the machinery guarantees, so that you can rely on it without checking:

| Guarantee | Why it exists |
|---|---|
| Imputation and scaling are fitted **inside the pipeline**, per fold | the classic silent leak — fitted on the whole frame first, every score looks better than it is |
| Splits are **always stratified**, with no option to turn it off | an unstratified split of a 3:1 dataset can hand a fold zero rows of the rare class |
| Group and time splitters exist and **must** be given their column | a random split of rows that repeat an entity scores high and is meaningless |
| `fold_consequences` reports what each fold contains, **including absent classes** | macro-F1 averages only over classes it can see, so a fold missing a class scores *higher* |
| Every metric is defined once, in `evaluation.py`, and shared by both models | two models on two rulers is not a comparison |
| The driver encodes the label to `0..k-1` itself, keeping the readable names | a string label reaching the arithmetic raises or, worse, scores 0.0 for everything |
| A report exists **if and only if** a run reached the end | now a property of the driver, not of your care |
| The renderer refuses to write outside the experiment folder | an `out/` written beside the machinery is not in the folder you are submitting |

Every module ends with the same line: *if you are about to add a default or a policy here,
it belongs in SKILL.md instead.* That is the boundary — hold it.

## Inputs

| Input | Required | Where it goes |
|---|---|---|
| Dataset path | yes | `dataset` — CSV / TSV / XLSX. |
| Test file | no | `test_dataset` — a **labelled** holdout supplied alongside the training file, or `null`. |
| Label column | decide | `target.column`. Infer from the evidence; **ask the user if it is ambiguous** (Step 1). |
| Model pair | decide | `models` — **derived** from the evidence, never a fixed shortlist (Step 4). |
| Split sizes | decide | `validation.test_size`, `validation.val_fraction` (Step 6). |
| Split unit | decide | `validation.unit` — row / entity / time, from the structure evidence, **not assumed** (Steps 3 and 6). |
| Validation strategy | decide | `validation.strategy` — **decide it, do not default it** (Step 6). |
| Tuning metric | decide | `tuning.metric` (Step 6). |
| Primary metric | decide | `primary_metric.name` (Step 7). |
| Student ID / name | **ask your user** | `report.student_id`, `report.student_name`. Not in the data, not in your environment, not inferable — **ask**. See Step 8. |
| Variant | **ask your user** | `report.variant`. Ask — do not take the example printed in this file as your answer. See Step 8. |
| Model version | read your environment | `report.model_version` — **the model you are actually running as**, never an example. See Step 8. |
| Harness | read your environment | `report.harness` — **the interface you are actually running in**, and its version. See Step 8. |
| Repository URL | **ask your user** | `report.repository`. Ask. **Never invent one, and never reuse a URL that appears in your environment** — that one belongs to your harness, not to this skill. See Step 8. |

## Where your files go

`--experiment` is **required and explicit**, and the current directory is never consulted.
That is the architectural point, not a convenience:

```bash
# the experiment folder is the one CONTAINING .opencode/, not .opencode itself
python "<skill>/driver.py" prepare --dataset "/data/train.csv" --experiment "/work/my-run"
python "<skill>/driver.py" run     --experiment "/work/my-run"
```

| | Directory | How you name it |
|---|---|---|
| **Skill directory** | holds `SKILL.md` and `scripts/`. Imports resolve here. | the base directory your harness printed for this skill |
| **Experiment folder** | the folder you are working in — the one that contains `.opencode/` | **you pass it as `--experiment`** |

**Everything the run produces goes to `<experiment>/out/`, and the plan goes at
`<experiment>/plan.json`.**

An earlier version of this skill told the model to derive the output folder from
`Path.cwd()`, and a run whose harness had changed directory into `.opencode/` wrote its
whole output to `.opencode/out/` — outside the folder being submitted, and outside the
skill directory, so nothing about the run looked wrong. "Start the harness in the right
folder" is an operational fix. Taking the experiment root as an argument is the fix that
holds regardless of where anything starts.

You do not have to remember to get this right: the driver refuses an `--experiment` that
resolves inside the skill directory, inside any `.opencode/` or `.claude/` tree, or into a
directory holding `SKILL.md`. If you see that refusal, the fix is the path you passed, not
the guard.

## Step 0 — Prepare the evidence

```bash
python "<skill>/driver.py" prepare --dataset "/data/train.csv" --experiment "/work/my-run" --label <column>
```

`prepare` reads the frame and writes the evidence Steps 1–6 are argued from, into
`<experiment>/out/evidence/`:

| File | Producer | What it answers |
|---|---|---|
| `candidates.json` | `label_candidates` | which columns could be the label, with the evidence for each |
| `eda.json` | `profile` | rows, features, class counts, imbalance, missingness, duplicates, numeric summary |
| `model_selection.json` | `model_selection_evidence` | the shape facts a family choice has to cite: size, `one_hot_width`, cardinality, balance |
| `structure.json` | `structure_evidence` | whether the rows are independent — candidate entity and time columns, with evidence |
| `rarest_class.json` | `rarest_class_evidence` | whether the rarest class supports the split you are considering |
| `candidate_screen.json` | `screen` | every candidate family fitted on one protocol and ranked — Step 4's evidence |
| `environment.json` | the driver | the interpreter and library versions the evidence was computed under |

**Read them. Do not re-derive any of it by hand** — the point of the split is that the
numbers in the report come from one computation, and running your own pandas snippet beside
it produces a second set of numbers nothing reconciles.

`candidate_screen.json` costs a few minutes because it fits seven families several times
over. If you want the evidence without the wait, add `--no-screen` and run
`driver.py screen --dataset ... --experiment ...` on its own later — same file, same
numbers, and Step 4 cannot be done without it. `screen` takes `--group-column` /
`--time-column` to override the split it inferred from `structure.json`, and `--folds` /
`--sample` / `--seed` to change its protocol; a `--sample` **does not** change how the final
models are fitted, only how many rows the screen looks at.

If you do not yet know the label, run `prepare` without `--label`: it writes
`candidates.json` and stops, telling you so. Run it again with `--label` once Step 1 has
decided. `run` re-derives anything missing, so `prepare` is a convenience and never a
prerequisite — but a plan argued from evidence you never read is a plan argued from nothing.

Use an interpreter that actually has the libraries — `numpy`, `pandas`, `scikit-learn`,
`scipy`, `matplotlib`. They are often not on the bare `python` on PATH:

```bash
python -c "import sklearn, scipy, pandas, matplotlib; print(sklearn.__version__)"
```

If that fails, find one that works (`py -0` lists them on Windows; a conda install is the
usual answer) and use it for **both** commands. `prepare` records the interpreter it ran
under, and `run` refuses to execute under a different one — the evidence describes a frame
read, and versions resolved, in one environment, and executing in another fits different
library defaults without saying so. (Override with `SKILL_ALLOW_OTHER_INTERPRETER=1` only
if you know why the interpreter changed.)

Then work through the steps below, writing `plan.json` as you decide. Every step names the
fields it fills; **The plan file** section at the end lists them all in one place.

## Step 1 — Identify the task

Read `candidates.json`. It lists every plausible target **with the evidence**:
cardinality, missing count, whether the name matches a convention (`label`, `class`,
`target`, `outcome`), whether it looks like an identifier.

Read that list and decide — do not take the first entry:

- A name match *plus* low cardinality is strong evidence; cardinality alone is weak.
- A numeric column with almost as many distinct values as rows is a measure, not a class.
- **If two candidates are close, ask the user.** A wrong label invalidates everything
  downstream and no amount of good modelling recovers from it.

Confirm the target is discrete and note how many classes there are. Note which class
matters more, and what each kind of error costs — those are two of the four inputs to the
metric decision in Step 7, not the decision itself.

→ **`target.column`**, `target.reason`, `target.alternatives_considered`.

## Step 2 — Profile

Read `eda.json` (Step 0 wrote it) and `out/figures/` (the driver drew the EDA figures).

Report: rows, columns, feature types (nominal / ordinal / interval / ratio), missingness
per column, duplicates, class counts, and the imbalance ratio (majority ÷ minority).

EDA exists to **decide Steps 3–5**, not to decorate the report. State in one line what each
finding implies for a decision you are about to make — that line belongs in
`narrative.feature_engineering` or `cleaning.reason`, not in a section of its own.

## Step 3 — Clean, driven by task intent

Missing values: drop rows / impute / let the model handle them. Outliers: **keep them if
they are the signal** — a rare class (fraud, disease) is made of outliers, and dropping
them destroys the task. Filter only when the goal is a "typical pattern" model.

Cleaning asks *is this data trustworthy*, not *is this feature convenient for the
algorithm*.

You have a **bounded** set of cleaning actions, and that is deliberate: each one is a field
value with a documented effect, so what the report says was done is what was done. Anything
outside the menu is not available — do it in the argument, not by inventing a field.

| Decision | Field | Menu |
|---|---|---|
| columns held out of the model | `cleaning.drop_columns` | a list of column names (`[]` for none) |
| rows with no label | `cleaning.drop_rows_with_missing_label` | `true` / `false`. A missing label cannot be imputed and cannot be scored — say which way |

**A missing label is not a missing feature, and it is not something to fix in the CSV.**
Imputing a feature guesses a measurement; imputing a label invents the answer the model is
being asked to predict. And a row with no label cannot be screened, trained on, or scored,
so `prepare` drops those rows from the screen and records how many in
`candidate_screen.json`'s `rows_dropped_missing_label` — cite that number in
`narrative.cleaning_why`. Do not pre-clean the file and point `dataset` at the copy: that
hands the driver a frame the evidence was never computed from, and it silently replaces the
class names in the report (`yes`/`no`) with whatever the copy encoded them to (`0`/`1`).
| the rarest class | `cleaning.rarest_class_action` | `"keep"`, or `"drop_duplicated_rows"` |
| outliers | `cleaning.outlier_policy` | `{"method": "keep"}`, or `{"method": "iqr", "factor": m, "columns": [...]}` |

`drop_columns` may not contain the target. Imputation *how* is not in the menu: the
pipeline always imputes, fitted inside the split, and always has. If median-or-mode is
wrong for the missingness you found, say so in `cleaning.reason` — but do not impute the
frame by hand, which is the leak the pipeline exists to prevent.

**Check the rarest class here, before Step 6 — not there.** Read `rarest_class.json`. A
thin class is not simply real or fake, and this evidence does not decide which it is. It
reports on four separate questions — is the name meaningful, are the rows duplicated, do
identical rows carry different labels, and does the class separate locally.

| Evidence | Reading |
|---|---|
| `looks_like_placeholder_name` — the class is called `other`, `unknown`, `n/a`, `?` | evidence about the **name** — a bucket for whatever did not fit, or a legitimate "other" the task really has. |
| `effective_unique_rows` far below `n_rows` | evidence about **duplication** — 12 rows may be 3 records repeated 4 times, so the sample is smaller than it looks. |
| `contradicted_rows` > 0 — identical features under a different label | evidence about **conflict** — *possible* label noise, duplicate conflict, genuine ambiguity, or a discriminative column that is missing. |
| `lift_over_chance` near 1 — a rare row's nearest neighbour is a majority row | evidence about **local separation** — *these columns* give the class little structure. A real class can be spread through the feature space just as easily. |
| `lift_over_chance` far above 1 — rare rows cluster together | evidence of **local structure** — the class is separable here, which is a fact about the columns and not about how meaningful the class is. |

**Turning that evidence into a decision is yours, and the decision belongs to this step**,
not to Step 6, and not to a constant in the scripts. Merging a catch-all, de-duplicating,
relabelling and dropping rows are different responses to different findings, and only some
of them delete data. The one action available to you here is
`rarest_class_action: "drop_duplicated_rows"` — for a class that is mostly duplicated
rows, where dropping them is the honest fix. Anything else, say what you found and why you
kept the class. Say which you did and why: keeping a class that is really a catch-all
drags down every macro-averaged metric in the report, which is the metric you selected on.

**Also here: are the rows independent?** Read `structure.json`. It reports every column
that could be an entity key or a timestamp, with the evidence for each — a name hint, how
many rows share a value, and whether rows sharing a value are actually near each other in
the other columns once the candidate is held out.

At this stage the question is only whether a **column** is a key, because that changes what
the model may see:

- **An entity key must not be a feature.** An identifier carries no transferable signal; a
  model handed one learns to recognise the entity. Put it in `cleaning.drop_columns` and
  name it in Step 6 as `validation.group_column`.
- **A timestamp is not a feature either** unless you derive something from it deliberately.
  Put it in `cleaning.drop_columns` and name it in Step 6 as `validation.time_column`.
- Ordinary categoricals will show some lift — rows sharing a `region` are genuinely a
  little alike. That is a predictor, not a group. Read the evidence table in
  [reference.md](reference.md) §8b; the lift is withheld with a stated reason when the
  question cannot be answered from the data, and a withheld lift is not a failure.

If the columns are clear but you suspect the *rows* still repeat (a panel shipped without an
id column, several rows per site), say so in `narrative.limitations` — Step 6 is where it
changes the split.

→ **`cleaning.*`**, and the unit note you will use in Step 6.

## Step 4 — Select two models

**Do not choose from a shortlist of favourites. Derive the pair from this dataset — and
from this dataset's measurements.**

The registry holds seven families and deliberately carries no ranking and no `recommended`
flag. A fixed pair written into this file would be the same decision made for every
dataset, which is the transfer this skill exists to prevent — it would just be hidden in
prose instead of in a constant.

1. **Read `out/evidence/candidate_screen.json`.** Step 0 wrote it: every family in the
   registry was fitted on the **same folds** of the pool, with its own defaults, and scored
   on the same two metrics. Nothing is argued; `families` is a ranked table of numbers, and
   `shortlist` is arithmetic on them — **every family within one standard deviation of the
   best mean**. Read `how_to_read_this` in that file before you use a number from it.

2. **Anything under the floor is eliminated, and the elimination is not yours to write.**
   It is behind on its own defaults at its own fold spread. Do not rescue it with an
   argument, and do not re-rank the survivors by score either — the shortlist is
   deliberately one standard deviation wide precisely so that it excludes what is clearly
   behind rather than crowning a winner.

3. **A family absent from `families` was not measured, and that is not the same as losing.**
   `excluded_without_fitting` lists the ones a hard cost bound kept out — an RBF kernel past
   15,000 rows, a neighbour search past six million cells — and `errored` lists the ones
   whose fit raised. Neither list is evidence about this dataset, and a family on either
   **must not be described in the report as eliminated**. Name it as not measured, and say
   why. If `errored` is non-empty, fix the cause and re-run `driver.py screen` before
   writing the plan.

4. **Take two from the shortlist that differ in *inductive bias*** — the shape of boundary
   they can express — and therefore usually in what they require. Two families that fail in
   the same way tell the report nothing. The file's `pairs_with_different_bias` lists the
   combinations that qualify, per metric, so this is checkable rather than asserted:
   `decision_tree`, `random_forest` and `gradient_boosting` share one bias (a forest is
   trees averaged, a boosted model is trees fitted to residuals — same staircase), so a
   shortlist of `random_forest` + `gradient_boosting` offers **no** legal pair. When that
   list is empty, or the shortlist holds fewer than two families, take the best family from
   the shortlist and add the highest-scoring family from `families` whose bias differs, and
   say in the narrative that the shortlist could not supply the pair.

5. **The judgement you are still making is *which two of the shortlist*, plus *why*.** The
   shortlist narrows the field from seven to a handful; it does not pick. A pair taken from
   it without a reason is two numbers, not a decision. Say what each survivor's bias buys
   here, and why the other shortlisted families were not taken.

6. **Record the whole chain**: every candidate and what the screen measured for it, why each
   survivor survived, why each rejection was rejected, and the final pair. The chain goes in
   `narrative.candidate_models`; each model's own argument goes in its `reason` and
   `alternatives_considered`.

**Contrast is a consequence, not a criterion.** A pair with different inductive biases makes
Step 5 easier to write — but that is a side effect of choosing well, never a reason to
choose. Selecting a model because it makes the report easier to compare is picking the
report over the task. The priority order is:

```
measured screen → dataset suitability → task suitability → computational feasibility
    → complementary models → ease of explanation
```

**Screening numbers are not report numbers.** A screen fits every family with plain defaults
and no grid, so a family that is merely untuned looks worse than it is — which is why the
shortlist is one standard deviation wide. Quote a screening number only as a screening
number, and never confuse it with the tuned validation score Step 6 produces or the test
score from Step 8. The test split is not involved in this at all: `screen` cuts it off
before it fits anything, so no family is ever ranked on rows the final model will be scored
against.

Screening is also not the whole decision. It tells you which families this dataset cannot
rule out; it cannot tell you which two to keep, because a margin neither of you measured.
Measured performance is **supporting evidence**, not the decision: a higher score does not
rescue a family that does not fit the task, and no shortlist replaces reading
`model_selection.json` from Step 0 — size, dimensionality, `one_hot_width`, categorical
cardinality, class balance, constant columns, missingness. Nor is accuracy usually the
number in question: the comparison is reported on the metric chosen in Step 7, and on an
uneven dataset that is rarely plain accuracy — which is why the screen records both
`f1_macro` and `accuracy` and gives a shortlist for each.

→ **`models`** — exactly two entries, each with `family`, `grid`, `reason` and
`alternatives_considered`. The grids are Step 5's business for `scale` and Step 6's for
width.

**Your grid keys must be that family's real parameter names.** The validator checks every
key against `make_model(family).get_params()` and refuses an unknown one by listing the
valid names — because scikit-learn's own error for a bad key names neither the model nor
the alternatives. Write `{"n_estimators": [200, 500]}`, not `{"trees": [...]}`.

## Step 5 — Preprocessing, per model

This is where most submissions lose points by scaling once and reusing it. Here it is
literal: `scale` is a **field on each model**, not a property of the run.

```json
{"family": "logistic_regression", "scale": true,  "scale_reason": "effectively required - ..."}
{"family": "random_forest",       "scale": false, "scale_reason": "not required - ..."}
```

Whether the two even differ is your decision, taken per estimator from the registry's
`scaling` entry ([reference.md](reference.md) §3). The registry states a **strength of
preference**, and the strength differs: "generally recommended" is not the same claim as
"effectively required", and the report is graded on the difference — which is why
`scale_reason` is a required field and why a bare `true`/`false` would not be enough. Write
the strength, not just the direction.

Imputation is different: *where* it happens is not a decision, because it is already inside
the pipeline, where it cannot leak. Never impute or scale the frame by hand before
splitting; that is the classic silent leak, and it makes the scores look better than they
are. There is no field for it, on purpose.

Feature selection or reduction only with a stated reason (identifier-like columns,
high-dimensional sparsity) — expressed as `cleaning.drop_columns`, with the reason in
`cleaning.reason`.

→ **`models[].scale`**, **`models[].scale_reason`**, and **`models[].stopping`** /
**`models[].overfitting_control`** (see Step 6).

## Step 6 — Decide how validation is formed

Three roles, each with exactly one job:

| Part | Job | Never used for |
|---|---|---|
| train | fit the model | — |
| validation | select hyperparameters | final reported numbers |
| test | final reported numbers, touched once | any selection |

**First of all: what unit does a split have?** Step 3 established whether a *column* is an
entity key or a timestamp. That answer decides which of these three questions you are even
asking:

| If the rows are… | The split must keep… | `unit` + `strategy` |
|---|---|---|
| independent | the class proportions | `unit: "row"` with `"stratified_kfold"` or `"holdout"` — **the normal case** |
| grouped (several rows per entity) | **whole entities on one side** | `unit: "entity"` with `"stratified_group_kfold"` or `"grouped_holdout"`, plus `group_column` |
| time-ordered | **the arrow of time** | `unit: "time"` with `"temporal"`, plus `time_column` and `gap` |

Getting this wrong is the quietest failure in the whole workflow: split grouped rows at
random and the entity appears on both sides, the model recognises it rather than learning
the task, and the score comes out **high and entirely plausible**. Nothing raises. So the
pairing is checked: `unit` and `strategy` must match, a group strategy must name its
`group_column`, `temporal` must name its `time_column` and `gap`, and a `n_splits` on a
holdout is refused rather than ignored. A group strategy with no group column is just a
random split with extra steps, and it is refused by name.

- **If you are in the grouped or time row, everything below changes.** The estimate is no
  longer one holdout of *n* rows — it is *g* entities or *w* windows, and the quantum is
  coarser. The driver's validation record redoes the arithmetic at that unit and says what
  it costs.
- **Confirm the guarantee, do not assume it.** The driver records the group overlap per
  fold — it must be zero — plus which classes are absent from each fold, in
  `out/evidence/validation.json`. Absence is worth checking because macro-F1 averages only
  over the classes it can *see*: a fold missing a class is not scored zero for it, it
  scores **higher** than a fold that has it.
- **Never leave the key in the feature set as well.** A column that is both a feature and a
  grouping tells the model the answer and then tests it on the same information.

**The options are wider than a two-way choice.** Plain holdout, k-fold, group-aware folds
and temporal folds are all on the table. Which one fits is a question about *this* dataset —
the structure row you landed on above, the size, the class distribution, how stable a
selection you need, and how much fitting time you will spend — not a ranking you can look
up. Four things worth weighing:

1. **The data comes before the split.** Step 3 already asked whether the rarest class is one
   you should be selecting on at all; if the evidence there pointed at a catch-all,
   duplicated rows, contradictions, or a name with no meaning, that is settled before this
   step opens it again. This is not an alternative to choosing a validation strategy — it
   happens first, and it is the one part no primitive can do for you. **A cleverer split is
   not a cure for an unsupported class:** averaging *k* estimates of a label with no signal
   gives a stable estimate of nothing, and it looks reassuring in the report.
2. **A plain holdout.** Cheapest — one estimate, and every row outside it trains the model.
   Comfortable when the rarest class is well populated; it gives you no sense of how much
   the estimate would move under a different split.
3. **k-fold.** Worth its cost when the rare class is thin but real — it averages *k*
   estimates and validates every rare row. **Not a default, not mandatory**, and not free
   (*k*× fitting time) — wasteful when the class is well populated.
4. **Admit when the design cannot support the class.** The validation record gives you the
   arithmetic: how many rare rows one estimate would rest on, and which folds would lose the
   class entirely. When that is too thin to carry the claim you want to make, say so in
   `narrative.limitations` rather than quoting the number as if it were stable. A confident
   figure resting on a handful of rows is the failure mode here — and what counts as too
   thin is a judgement you make from those numbers, not a cut-off you look up.

**You do not encode the label.** The driver does it, once, before anything is fitted:
sorted unique values → integers `0..k-1`, with the readable names kept for the report. This
removes the silent-`0.0`-macro-F1 trap permanently rather than guarding against it: a
string label reaching `tune` cannot happen, so the failure it used to cause cannot either.
The consequence for you is only that `labelled` counts in the evidence are counts of *rows*,
not of encoded values.

Then decide, and record it:

- **The smallest class sets the noise floor, not the row count.** Selection is on
  macro-F1, where a class with 12 rows carries the same weight in the score as one with
  7,000. A split that is 99% fine by row count can be pure noise for the class that
  decides the macro average. Look at the rarest class first, always, and quote its number.
- **Per-estimate size is not the whole story.** A holdout gives one estimate, scored on the
  `val_fraction` you kept back; k-fold gives the mean of *k* estimates, each fitted on
  approximately `1 − 1/k` of the pool and scored on approximately `1/k` of it — so a single
  fold is usually a *smaller* estimate than the holdout, while every rare row is validated
  against exactly once instead of only the held-out share being seen. Averaging is generally
  more stable than trusting one draw — the folds are not independent samples of the data, so
  this recovers a smaller share of the split-to-split variance rather than giving a √k gain
  in precision. That, plus the final model still training on the whole pool, is what you pay
  *k*× the fitting time for. It is why k-fold can win even when its per-fold rare count is
  *lower* than the holdout's.
- **How many candidates are you ranking?** Taking the max of a dozen noisy estimates
  overfits the split. A wider grid needs a tighter estimate.
- **What does the width cost?** Every extra combination costs `folds` more fits, and the
  search is the slowest part of `run` by a wide margin. Measured on ~25k rows: an
  eight-combination gradient-boosting grid at 5 folds took just over thirty minutes, while a
  three-combination logistic-regression grid beside it finished in seconds. Cost is
  superlinear on an axis like `n_estimators` — each fit re-grows the whole ensemble — so
  `[200, 500]` roughly doubles the bill for a difference that usually lands in the third
  decimal. **Give each axis a reason it could matter for this dataset's rows and columns, and
  drop the range you cannot justify.** From the same run: the cheapest corner of that
  grid won it (`n_estimators=200`, `max_depth=3`, `learning_rate=0.1` → 0.76790, against
  0.76681 for 500 trees), so the four combinations built on 500 trees bought 0.001 of
  `f1_macro` for most of the run's clock. You can check this claim after the fact —
  `tuning.json`'s `scores` holds one entry per combination, and the wait is visible in how
  long the command ran.
- **`val_fraction` is a lever, not a constant.** The report prints the sensitivity: a
  bigger holdout keeps more rare rows but trains on fewer. Pick the trade-off and say why.
- **Fold count** is capped by the rarest class's row count. A fold holding zero rows of a
  class cannot score it, and the run is refused with the cap in the message. The cap is a
  ceiling, not a target: at the cap each fold scores a single rare row, which is the worst
  estimate available, so the largest fold count is usually the wrong choice.

**What the report must record:** the strategy, the rarest-class count you based it on, how
many rare rows that puts in an estimate, and the limitation it leaves. The first three are
in `out/evidence/validation.json` — quote them, do not re-derive them; the last is
`narrative.validation_argument`.

Which rows feed the split depends on what the provider shipped:

- **A labelled test file** → `test_dataset`; held out untouched, validation comes from the
  training file.
- **A single file** → `test_dataset: null`, and `test_size` splits the test part off the
  frame before validation is formed from the rest.
- **Open the second file before assuming anything.** A file named `test.csv` may still
  carry the label, in which case it is a labelled holdout, not a submission template. A
  `test_dataset` without the label column is refused — nothing in it could be scored — as
  is one carrying a class the training frame never saw.

**Name the metric the search optimised: `tuning.metric`.** It has no default, because it
decides what the search is *for*. It is experimental configuration, not a hyperparameter:
you fix it, you report it, and it is not tuned. Take it from the same four inputs as the
primary metric in Step 7, and make the two agree — a search tuned on one metric and
reported on another is a comparison nobody can check. The search scores `f1_macro` or
`accuracy` only; if the metric you want to lead with is neither, name it in
`primary_metric` and say in `tuning.reason` which one the search used instead, and why. The
validator refuses any other value here rather than silently searching on something else.

**`seed` is applied to every estimator that has a `random_state`, by the driver.** You do
not pass it per model, and you do not put it in a grid: it is experimental configuration,
not a hyperparameter, and the driver hands it to each family's `random_state` whenever the
family has one. Without that, a forest's bootstrap — not the seed — would decide the
numbers while the report printed the seed beside them.

**Also decide what the winner is finally fitted on — `tuning.final_fit_scope`.** One of
`train`, `train+validation`, or `pool` (everything except the test set). Under k-fold the
usual answer is `pool`: every row was validated against exactly once, so holding any back
from the final fit throws away data for nothing. The report prints the scope and the row
count it implies, so state the reason in `final_fit_reason`.

**Every fitted model stops somewhere — say where, in `models[].stopping`.** It is a
required field, and it is not always an iteration count:

| Where the stopping comes from | What to write |
|---|---|
| an iterative solver | the cap and tolerance you set, and whether the cap was **reached** (`max_iter`, `tol`). A solver that hit its cap has not converged, and its score is not the converged model's score. |
| tree growth | the constraint that ended it (`max_depth`, `min_samples_leaf`, `ccp_alpha`). An unconstrained tree stops when every leaf is pure — that is memorisation, not convergence. |
| an ensemble | rounds and learning rate, plus whether early stopping was used and what it monitored. |
| nothing iterative | say so plainly, and name the constraint doing the same job. |

Do not invent a criterion you did not set. Where a model has no iterative stopping, the
honest entry is that it has none — not a plausible-looking `max_iter`. `overfitting_control`
is the companion cell: what keeps this model from fitting the training rows too closely —
the penalty, the depth cap, the bootstrap resample. Keep both to a short clause; the
report's table wraps at about 34 characters a line.

→ **`validation.*`**, **`tuning.*`**, **`models[].stopping`**,
**`models[].overfitting_control`**.

## Step 7 — Evaluate and compare

The driver scores **both models on the identical test data** once the plan is executed —
that is what makes the comparison fair, and one fixed metric definition means neither model
gets a different ruler. It writes `out/evidence/metrics.json` and
`out/figures/confusion_pair.png`, the two matrices side by side in one figure, because two
separate figures do not fit a two-page report and side by side is how a marker compares
them anyway.

**Choose the primary metric; do not inherit one — `primary_metric.name`.** Four inputs
decide it, and no single one of them settles it:

- the **task objective** — what the model is for, in the user's terms;
- the **class distribution** — how uneven the classes actually are;
- the **cost of each error type** — which mistake is worse, and by roughly how much;
- the **validation consequences** — whether the estimate can carry the claim. A metric this
  design cannot estimate stably is not a better metric, only a noisier one.

Accuracy, macro-F1, precision, recall, ROC-AUC and PR-AUC are all candidates.
[reference.md](reference.md) §6 says what each is sensitive to — read it as considerations,
not a lookup: *imbalanced* does not by itself mean macro-F1, and *rare positive* does not by
itself mean recall. Write the metric you chose in `name`, the input that decided it in
`reason`, and the alternative you rejected in `rejected` with `rejected_reason`. It must be
the metric `tuning.metric` optimised in Step 6, or `tuning.reason` must say why not.

Then discuss bias–variance and any violated assumption (Naive Bayes independence, linearity,
distance concentration) in `narrative.performance_comparison` and
`narrative.limitations`. That is where "the linear model is the steadier of the two if the
test file is drawn from a slightly different population" belongs — the driver cannot write
it, and it is the part a marker is actually reading for.

→ **`primary_metric.*`**, **`narrative.performance_comparison`**,
**`narrative.limitations`**.

## Step 8 — Run the driver and read the report

**Ask your user for four things before you write the plan.** `report.student_id`,
`report.student_name`, `report.variant` and `report.repository` are facts about the person
submitting the report. None of them is in the dataset, in the assignment brief, or in your
environment. **Ask, and wait for the answer, even when you are certain you already know** —
every wrong value this skill has produced came from a model that was certain:

- `IN6227` is the course code, and it is also a folder name and a filename stem, so it reads
  like a matric number. It is not one.
- Agent harnesses ship with a repository URL of their own. It is in your environment, it
  looks like an answer, and it is somebody else's repository.
- The `plan.json` skeleton later in this file prints `"variant": "Variant-2"`. That is an
  example, and a wrong variant costs exactly what a wrong name costs.

No automatic check can catch any of this. The validator sees a non-empty string and an
`https://` prefix; it cannot tell a real matric number from a plausible one, or this skill's
repository from your harness's. **The only defence is that you asked.**

**Then establish who you are.** The report must declare the model and the interface that
actually produced it, and both are graded. Neither is the example printed in this file or in
the assignment brief:

- **Model name & version** — `report.model_version`. The exact identifier you are being
  served as. Your interface normally exposes it: a status line, a model picker, a session
  header, an environment variable. Read it there. Do not infer it from what you would rather
  be.
- **LLM interface** — `report.harness`. The *kind* of interface plus its version: `API`,
  `ChatUI`, or `Agent Harness such as <name> <version>`.

**Never copy an example.** A report naming a model that did not run it is worse than one
naming nothing, because this is the single field whose entire purpose is to prevent exactly
that. If you cannot determine your own identity, stop and ask the user — do not guess.

**Include the real repository URL — `report.repository`.** The report's Repository section
is graded, and it must be the GitHub link to **this skill** — the folder you are reading
this file from, not the harness you are running inside. **Ask the user for it** (see the top
of this step); a URL you were never given is worse than an empty section, because nobody can
check it. The validator refuses a value that is not a URL, which is the only part of this a
machine can check.

Then write the prose. Every field in `narrative` is spliced into the report verbatim, so
write it as report prose, not as notes:

| Field | Where it lands |
|---|---|
| `candidate_models` | §1–2: the family choice and the eliminations, with this dataset's numbers |
| `feature_engineering` | §3: what was constructed, or one line saying nothing was and why |
| `cleaning_why` | §1: the cleaning table — one `{issue, action, why}` row per thing you did |
| `validation_argument` | §4: why this unit and this strategy |
| `primary_metric_argument` | §4: the four inputs, condensed |
| `performance_comparison` | §5: how the two models actually differed |
| `limitations` | §5: what this design cannot support |

`report.dataset_label` is the dataset line as a reader should see it, e.g.
`` "IN6227 A1 Variant 2 — train.csv (31,112 rows) / test.csv (13,334 rows)" ``.

Now run it:

```bash
python "<skill>/driver.py" run --experiment "/work/my-run"
```

**Give this command time — minutes to well over half an hour, never two.** `run`
cross-validates each family across its whole grid before it fits anything, so its runtime
scales with `grid × folds × rows`. Measured on ~30k rows: two families over a handful of
combinations took three minutes; an eight-combination gradient-boosting grid at 5 folds took
thirty-four. Width on an `n_estimators`-style axis is the expensive kind — Step 6 is where
that is decided, and where to cut it if the wait is not earned. Many agent harnesses cap a
single shell call at two minutes by default. When that cap hits, the command is killed
mid-tuning — the log stops at `[driver] tuning <family>`, no report appears, and it reads
like a crash. **Raise the limit for this call**: an explicit timeout of an hour is not
excessive on your shell tool, or whatever setting your harness reads. Do not narrow the grid
just to fit the default — the two are unrelated, and a grid trimmed under time pressure is a
grid you cannot defend.

`run` executes the plan and, on success, writes `out/report.md` and `out/report.pdf` and
prints the headline numbers. Before it renders anything it checks:

- **the plan** — every field, every menu, every grid key, at once. A plan repaired one error
  per run costs one run per error, so the refusal lists *all* the problems it found, each
  naming the field. Fix them together and run again.
- **the interpreter** — against the one `prepare` recorded.
- **the target, the columns, the folds** — a `drop_columns` name that is not a column, a
  `group_column` that is missing, more folds than the rarest class can fill.
- **the report's evidence** — including that `out/evidence/eda.json` came from `profile()`
  and `metrics.json` from `evaluate_pair()`. Existence alone was not enough: a run that
  reimplemented the workflow wrote its own `eda.json` by hand, missing exactly the three
  cleaning keys, and every report section it fed still rendered. The three cleaning keys are
  the whole of section 1's "issues found" line, so the check is on the contract, not the
  filename.

**If it fails, no report is written, and that is the point.** `run` deletes `report.md` and
`report.pdf` at the very start — before it even parses the plan — so a report exists if and
only if a run reached the end. A failed run leaves the evidence, `plan.json` and
`out/evidence/validation.json` sitting there with no report beside them; the message names
the step that failed (`FAILED at: tuning random_forest`). Read it, fix the plan, run again.

The renderer draws the matric / name / model / harness block as a header **on page 1** —
the assignment asks for it "at the top of the first page", not on a separate cover sheet —
so the whole PDF is ≤ 2 pages including that header. It prints a warning and exits non-zero
if you overflow, and `run` then **deletes the PDF** rather than leaving a three-page
near-miss in the folder. The fix is to trim prose, not to shrink the font.

## Hard constraints

1. **Never hardcode anything from a particular dataset** — not a column name, not a
   suspected entity key, not a timestamp, not a label value. Inference must work on a
   dataset the skill has never seen, so the *evidence* for each is computed and read here.
2. **`plan.json` carries decisions and prose, never code.** There is no field that takes an
   expression, a snippet, or a callable. A key named `code`, `script`, `formula`, `query`,
   `eval`, `lambda`, or one prefixed `custom_`/`on_`/`pre_`/`post_`, is refused anywhere in
   the tree, at any depth. If you find yourself wanting one, the decision you are trying to
   express is not in the menu — say so in the report rather than working around it.
3. Every decision carries a one-line justification, and the alternative you rejected. The
   validator refuses an empty `reason` or an empty `alternatives_considered`, mirroring
   what `DecisionLog` has always required: a blank reason still renders and still paginates.
4. Exactly two models, of two different families, evaluated on one shared protocol. Both
   come from `out/evidence/candidate_screen.json`'s shortlist, and a family that file does
   not measure — one on `excluded_without_fitting` or `errored` — is **not** an eliminated
   family. Never write one up as if it had lost: "not measured" and "measured and behind"
   are different claims, and the difference is what the screen was added to make visible.
5. **Accuracy is not the objective.** Justified choices beat high scores.
6. Report ≤ 2 pages total, identity header included.
7. All numbers come from actually running `driver.py run`. Never invent a metric, and never
   compute one yourself — a second computation is a second set of numbers nothing
   reconciles.
8. **Do not write the Reflection section.** It is written by the human and is not counted
   toward the two pages.
9. **Keep the decisions out of `scripts/`.** If you find yourself wanting to add a default
   or a threshold to a module, it belongs in this file instead.
10. **Pass `--experiment` every time.** Never `cd` into `.opencode/`, and never derive an
    output path from the current directory.
11. **Ask the user for their matric number, name, variant and repository URL.** All four,
    every time, before writing the plan — even when you are sure you know them. They are
    the only fields in the plan that exist nowhere but in the user's head, and the only
    ones no check can verify. See Step 8.

## Troubleshooting

- **`PlanError` listing several problems** → that is the design. Fix all of them; nothing
  was executed and no report was written.
- **`FAILED at: <step>`** → read the step name. `tuning <family>` means the plan reached
  fitting and the grid or the data is at fault; `checking the interpreter` means `prepare`
  ran under a different Python; `reading <file>` means the path in `plan.json` is wrong.
- **The log stops at `[driver] tuning <family>` and never finishes** → this is almost never
  the driver. `run` spends its time cross-validating, and an agent harness that caps a shell
  call at two minutes cuts it off there — the harness usually says so in its own words
  ("terminated command after exceeding timeout 120000 ms"). Run it again with a much larger
  timeout on that call — an hour is a safe ceiling, and half an hour is not enough for a wide
  grid. The evidence gets rewritten rather than corrupted, so a repeat costs only the wait.
  Do not answer it by dropping a family: the cap belongs to the harness, and the pair was
  chosen on its own merits. Whether the *grid* was wider than the dataset justified is a
  separate question, and Step 6 is where it gets asked — not here, under time pressure.
- **`no plan at <path>`** → `plan.json` goes in the experiment folder, beside `out/`, not
  inside `out/`.
- **`candidates.json` lists several plausible targets** → ask the user. Do not guess.
- **`n row(s) carry no label and cannot be screened; dropped`** → not an error, and not
  something to repair in the CSV. The target column has missing values; `prepare` left those
  rows out of the screen and recorded the count as `rows_dropped_missing_label`. Set
  `cleaning.drop_rows_with_missing_label` to `true` and cite that count in
  `narrative.cleaning_why`. Writing a pre-cleaned copy of the dataset and pointing `dataset`
  at it is the wrong move twice over: the copy is not the frame the evidence describes, and
  the label values you chose become the classes the report prints.
- **`... label code(s) are negative, so they belong to no class`** → the same fault reaching
  a different guard. Something encoded a missing label instead of dropping the row; the
  message says how many. This is the sentence that replaces `np.bincount`'s
  `'list' argument must have no negative elements`, which named neither the column nor the
  cause.
- **`candidate_screen.json` has a non-empty `errored`** → a family's fit raised, so it was
  neither measured nor eliminated, and Step 4 cannot choose it. Fix the cause and re-run
  `driver.py screen` before writing the plan. If the cause is the environment rather than
  the data — a `threadpoolctl` older than 3.0 cannot read a newer MKL's version, which
  kills sklearn's brute-force neighbour search, so KNN dies on any frame wide enough for
  `algorithm="auto"` to pick brute; `D:\Anaconda` had this until it was upgraded to 3.7.0
  — say so in `narrative.limitations` and pick from the families that *did* run. Upgrade
  `threadpoolctl` where you can (`pip install -U threadpoolctl`); the report must not
  describe the family as having lost.
- **The screen's `why` names a column you believe is a real entity key** → the inference
  refused it because `structure.json` reports `is_in_feature_set` true: it is a column the
  model predicts with, so rows sharing a value resemble each other for a reason that has
  nothing to do with shared entities. If you know better, pass `--group-column <name>` and
  the screen will use it.
- **The screen takes too long** → `--sample` looks at fewer rows (whole entities are kept
  together), or `--folds 2`. Either changes the screening numbers and both are recorded in
  the evidence; neither changes how the final models are fitted. Do not skip the screen:
  Step 4 has nothing else to go on.
- **Single-class or all-unique target** → stop and ask; it is a regression or an ID column,
  not a classification task.
- **A class with one row** → it cannot go through an ordinary stratified split, because one
  row cannot be divided across folds, and a holdout either keeps the class entirely or loses
  it from one side. That is the *mechanical* limit, not the judgement one, and `run` will
  refuse rather than quietly fitting something. It has to be dealt with before any split is
  chosen — merged into another class, dropped, or handled by a rule outside the model — and
  which of those is a decision, not a rule.
- **`validation.n_splits` refused as above the cap** → the message names the cap. A fold
  that cannot contain one row of every class scores *higher*, not lower, so the number is
  never silently produced.
- **The rarest class looks thin against the split you are considering** → read
  `rarest_class.json` **before** touching the split, then the arithmetic the driver records
  in `out/evidence/validation.json`. Read the evidence as evidence, not as a verdict. A
  placeholder name, heavy duplication and contradictions point toward a class you may not
  want to be selecting on; a low `lift_over_chance` says only that *these columns* do not
  separate it, and that can be true of a class that is entirely real. Decide, and write down
  the decision and what supported it.
- **ROC-AUC fails on multiclass** → expected; `evaluate` falls back to macro-averaged
  one-vs-rest and records which was used.
- **`pr_auc` or `roc_auc` is null** → the driver's evidence records the note that came with
  it. "Not defined for multiclass" is expected and correct to report as such; a note naming
  an exception is a real failure, and a blank cell is not an acceptable substitute for
  reading it. Never report a null without its note, and put it in
  `narrative.limitations` if it affects the claim.
- **`report.repository` refused as not a URL** → ask the user for the link; do not write a
  plausible-looking address.
- **A `report` field is non-empty and still wrong** (a course code where a matric number
  goes, your harness's repository, the example variant) → `run` cannot catch this; the
  validator only checks the shape of a string. The rule is at the top of Step 8: ask the
  user for all four, every time, before you write the plan.
- **PDF overflows two pages** → `run` deletes the PDF and says so. Cut discussion prose
  first, then drop a figure. Each image is capped at 0.20 of a page, so **two separate
  confusion matrices plus the EDA figures will not fit** — the driver already draws both
  models' matrices as one side-by-side figure rather than two.
- **`scipy` missing** → the rarest-class and structure evidence need it for the neighbour
  scan; without it, install scipy rather than deleting the call. The lift is the one piece
  of evidence that distinguishes a real rare class from scattered noise — and a real entity
  key from an ordinary categorical column.
- **`structure.json` names a column that looks like an entity key** → do not simply drop
  it. Check `rows_per_value_mean` first: if it is close to 1 the column is an identifier or
  a near-continuous measure, and the lift is withheld because the question is unanswerable,
  not because the answer was bad. If values genuinely repeat, put it in
  `cleaning.drop_columns` **and** pass it as `validation.group_column`.
- **`structure.json` reports lift `3.06` or similar for a plain categorical column** →
  expected, and not a finding. Rows sharing a category really are somewhat alike, so the
  column is a predictor; an entity key is a different claim and needs the name convention or
  an unmistakable repeat count to support it.
- **`lift_over_chance` is `null` with a `lift_withheld_because` note** → read the note and
  move on. It means the ratio would have been computed from too few events, or from values
  that are nearly all distinct. The raw share and baseline are still reported; using them
  anyway is how a 59× "lift" gets into a report.
- **A grouped or time-ordered dataset** → the usual row-based split is wrong and nothing
  will tell you. Set `unit`/`strategy` to the matching row of the table in Step 6, name the
  column, and read the overlap the driver records — it must be zero.
- **`models[i].grid` refused: "family has no such parameter"** → the message lists that
  family's real parameter names. Grid keys are bare estimator parameters
  (`{"max_depth": [6, 12]}`); the driver adds the pipeline step prefix itself.

## The plan file

One JSON object at `<experiment>/plan.json`. Every value comes from a closed menu — the
validator refuses anything else, and reports every problem it finds at once.

```json
{
  "schema_version": 1,
  "dataset": "/data/train.csv",
  "test_dataset": null,
  "seed": 42,
  "target":          {"column": "...", "reason": "...", "alternatives_considered": ["..."]},
  "cleaning":        {"drop_columns": [], "drop_rows_with_missing_label": true,
                      "rarest_class_action": "keep",
                      "outlier_policy": {"method": "keep"},
                      "reason": "...", "alternatives_considered": ["..."]},
  "models":          [{...}, {...}],
  "validation":      {"unit": "row", "strategy": "stratified_kfold",
                      "test_size": 0.2, "val_fraction": 0.2, "n_splits": 5,
                      "gap": null, "group_column": null, "time_column": null,
                      "reason": "...", "alternatives_considered": ["..."]},
  "tuning":          {"metric": "f1_macro", "final_fit_scope": "pool",
                      "final_fit_reason": "...", "reason": "...",
                      "alternatives_considered": ["..."]},
  "primary_metric":  {"name": "f1_macro", "rejected": "accuracy",
                      "reason": "...", "rejected_reason": "..."},
  "report":          {"student_id": "...", "student_name": "...", "variant": "Variant-2",
                      "model_version": "...", "harness": "...", "repository": "https://...",
                      "dataset_label": "..."},
  "narrative":       {"candidate_models": "...", "feature_engineering": "...",
                      "cleaning_why": [{"issue": "...", "action": "...", "why": "..."}],
                      "validation_argument": "...", "primary_metric_argument": "...",
                      "performance_comparison": "...", "limitations": "..."}
}
```

Menus, in full:

| Field | Allowed |
|---|---|
| `models[i].family` | any name from `models.py`'s registry — [reference.md](reference.md) §2 |
| `models[i].scale` | `true` / `false` (plus `scale_reason`) |
| `models[i].grid` | `{bare parameter name: [values]}` — keys checked against the family's own `get_params()` |
| `validation.unit` | `row`, `entity`, `time` |
| `validation.strategy` | `holdout`, `stratified_kfold` (row) · `grouped_holdout`, `stratified_group_kfold` (entity) · `temporal` (time) |
| `cleaning.rarest_class_action` | `keep`, `drop_duplicated_rows` |
| `cleaning.outlier_policy` | `{"method": "keep"}` or `{"method": "iqr", "factor": 0.1–10.0, "columns": [...]}` |
| `tuning.metric` | `f1_macro`, `accuracy` — what the scorer implements |
| `tuning.final_fit_scope` | `train`, `train+validation`, `pool` |
| `primary_metric.name` | `accuracy`, `f1_macro`, `precision_macro`, `recall_macro`, `roc_auc`, `pr_auc` |

**Two metric fields, and they are not the same decision.** `tuning.metric` is what the
search can score; `primary_metric.name` is the headline, out of a wider menu. They will
usually agree. When they do not, `tuning.reason` says which one the search used instead and
why — a search tuned on one metric and reported on another is a comparison nobody can check,
so the divergence has to be argued rather than assumed.

`schemas/plan.schema.json` ships so an editor can complete the fields, and a test asserts it
and the validator agree on the field set, so they cannot drift apart silently. When the two
disagree, `scripts/plan.py` is right.
