# Reference — model and preprocessing decision tables

Read this before Step 4. Nothing here is a rule to follow blindly; each row is a
reason to cite in the report.

## 1. Attribute types

| Type | Meaning | Example | Encoding |
|---|---|---|---|
| Nominal | unordered categories | region, species | one-hot (or target encoding) |
| Ordinal | ordered categories | low/med/high | integer preserving order |
| Interval | numeric, no true zero | temperature °C | scale |
| Ratio | numeric, true zero | duration, count | scale |

Also record: dimensionality, sparsity, and granularity (is a row a person, a session,
an event?). The same extreme value is noise under one task and the signal under another.

Granularity is not a description — it decides the split. `structure_evidence` produces the
evidence for it, in §8b.

## 2. Model families

This table is what you choose **from**, not what chooses. There is no recommended pair and
no ordering, and the last column is not a set of triggers. Each cell states what a family is
*good at* and what it *costs*; that becomes a reason only after you have put it beside
numbers you actually computed. "Random forest can capture nonlinear interactions" is a
property of random forests. "Rows ≫ columns, therefore random forest" is the misuse — it
reads as though the table had already decided.

| Family | Boundary it can express | Scaling | One-hot | Relevant considerations | Interpretability |
|---|---|---|---|---|---|
| Decision tree | non-linear — axis-aligned steps | not needed | optional | can express nonlinear relationships and interactions; the fitted tree can be drawn and read; a single tree is high-variance and will memorise noise unless depth or leaf size is limited | high |
| Random forest | non-linear — the same steps, averaged | not needed | optional | can capture nonlinear interactions; averaging decorrelated trees can reduce variance; cost grows with the number of trees and their depth | low–medium |
| Gradient boosting | non-linear — steps fitted to residuals | not needed | optional | can model nonlinear relationships and is often the strongest of these on tabular data; more sensitive to tuning and slower to fit | low–medium |
| Logistic regression | **linear in the log-odds** | generally recommended — the solver, the L2 penalty and the coefficient comparison all behave better when columns share a scale | categories must be numeric; one-hot is the usual choice | the only family here that is linear in the log-odds, which is what it is *for* in a comparison — but being the only linear family is a fact about the table, not a reason to include it in every pair; each coefficient is signed and comparable | high |
| KNN | non-linear — arbitrarily local | effectively required — the method *is* a distance | categories must be numeric; one-hot is common and imperfect | meaningful where local similarity is plausible; sensitive to dimensionality and to weakly relevant columns | low |
| Naive Bayes | non-linear, within the assumptions of the chosen variant | not needed | variant-dependent — see below | very fast, works on small samples, and a useful baseline; the conditional-independence assumption is strong and almost never exactly true | medium |
| SVM | **set by the kernel** — linear or not | generally recommended — the margin is measured as a distance | categories must be represented numerically; one-hot is a common choice | effective where a margin is plausible; fitting cost grows faster than linearly in the number of rows | low |

**Naive Bayes is a family, not one estimator.** The three `sklearn` variants make different
assumptions about what a column *is*, so "is naive Bayes suitable?" is really "which variant,
and does its assumption hold for these columns?"

| Variant | What it assumes a column is | A column that breaks it |
|---|---|---|
| `GaussianNB` | continuous, roughly normal within each class | counts, indicators, heavily skewed variables |
| `MultinomialNB` | non-negative counts | negative or fractional values |
| `BernoulliNB` | binary present/absent | anything with more than two values |

The registry wires up `GaussianNB`, the only variant that accepts a mixed numeric frame at
all — and the one whose assumption a skewed or count-like column most often violates.

### Choosing from this table

The capability table says what a family *can* do. It does not say whether it does it *here* —
and the difference used to be filled in by argument, which is how the same pair came back from
five runs on four datasets. So the ordering is now measurement first, table second:

```
candidate_screen.json        every family fitted on one protocol, ranked
        ↓
shortlist = within 1 std of the best mean      arithmetic, not judgement
        ↓
read the capability table against model_selection.json
        ↓
take two from the shortlist that differ in inductive bias
        ↓
record: what the screen measured for each, plus rejections and reasons
```

`screen` fits every family on the **same folds** with its own defaults, so the ranking is
like-for-like in the only sense that matters: no family is compared on a different split, a
different metric, or a different amount of tuning. Its exclusions are two, and neither is
about difficulty — a hard cost bound (RBF past 15,000 rows, neighbour search past six million
cells), or a fit that raised. A family on either list was **not measured**, which is not the
same as measured and behind, and the report has to say which it was.

Three questions still do the work the screen cannot:

- **How wide does the frame actually get?** Use `one_hot_width`, not the raw column count.
  That is what a linear model or a kernel actually faces, and it is the number that decides
  how much of a distance survives once the categorical columns are spread across indicators.
- **How many rows per column?** Distance-based methods can become less discriminative as
  dimensionality grows, particularly when many columns are weakly informative or irrelevant,
  and they get slower because a neighbour search costs more in higher dimensions. When they
  do degrade, they degrade *smoothly* — the accuracy just comes out lower, with no warning.
  The screen will show that as a low score with a wide spread; read it as a caution, not as
  a threshold.
- **What does this dataset's shape make of each family's own assumptions?** Skewed or
  count-like columns against `GaussianNB`'s normality, correlated one-hot columns against
  logistic regression's independence, a suspected interaction against a linear boundary. This
  is where the table earns its place: the screen tells you how they *scored*, and these are
  the reasons *why*.

**Contrast is a consequence of choosing well, not a criterion.** Choose models because they
are appropriate to the task and to this dataset. Seek complementary inductive biases, since
two families that fail in the same way tell you less than two that fail differently — but do
not pick a model in order to manufacture a convenient comparison. Selecting a model because
it makes Step 5 easier to write is picking the report over the task.

## 3. Preprocessing — ask the estimator, not a table

`MODEL_REGISTRY` is the source of truth for what each estimator needs; §2's table is only a
compact summary of it. A hand-written matrix used to sit here, and it was removed because the
table was prose you had to trust while `make_preprocessor` and `make_model` are what actually
run. Ask the code:

```
python -c "import sys; sys.path.insert(0,'scripts'); from models import describe_models; print(describe_models())"
```

That prints `scaling`, `one_hot`, `nonlinearity`, `supports_probability`, `assumes` and the
hyperparameters for all seven families. Read the full entries in `MODEL_REGISTRY`; the printed
table truncates them to their first clause. Three consequences:

- **`scaling`, `one_hot` and `nonlinearity` are descriptive properties, not automatic
  decisions.** They state what a family is like and what it costs, and matching that against
  a dataset is yours. `nonlinearity` in particular is what makes model *selection* possible
  without the registry making it.
- **There is no ranking and no `recommended` key.** One would move the choice back into the
  file, for whatever dataset its author had in mind.
- **No hyperparameter defaults either** — not for `C`, `max_depth`, `class_weight`,
  `n_neighbors` or `n_estimators`. Those decide the model for a particular dataset, so they
  are chosen in Step 4 and passed to `make_model(...)`.

A requirement belongs to the estimator, not to the dataset: the same column is a problem for
one family and invisible to another. That is why it is looked up per model rather than decided
once for the run, and why the registry's wording is a *strength of preference* —
"generally recommended" and "effectively required" are different claims, and the report is
graded on the difference.

## 4. Missing values

These are the ordinary responses, and the third column is what turns each into a *reason*
rather than a habit. Read the middle column as a starting point you have to justify against
your own data, not as an instruction.

| Situation | Usual response | Justification to write |
|---|---|---|
| A few values missing, at random | impute median/mode | costs nothing and keeps every row; the pipeline already does it |
| Non-trivial, at random | impute median/mode | keeps sample size, no leakage |
| Missingness itself is informative | add a missing indicator | absence carries signal |
| Structural (column empty for a subgroup) | drop column | cannot be imputed meaningfully |

Impute **inside** the pipeline so the statistic is learned on the training fold only —
imputing before splitting leaks test information. `make_preprocessor` does this already, so
*where* imputation happens is not your decision. *How* it happens still can be: median-or-mode
is a reasonable default, not a law. It is the wrong answer when the missingness is not random
(a value absent because it was never measured, or too extreme to be recorded), when a
categorical column has no meaningful mode, or when absence itself carries signal — the third
row above. Then the strategy is a modelling decision you make and write down.

**Dropping rows is a sample-size decision, not a missing-value one.** With imputation already
leak-safe it is rarely worth its cost — the same trade-off as `val_fraction` in §8, paid by
the minority class first. Reach for it only with a reason beyond tidiness.

## 5. Outliers and imbalance

- **Keep outliers** when they belong to the class of interest (fraud, fault, disease).
  Removing them removes the positives and inflates apparent accuracy.
- **Filter outliers** only for "typical pattern" models, and say so.
- Imbalance: class weighting (`class_weight='balanced'`) and resampling are **alternatives, not
  a sequence**. Either may suit the dataset and the validation design; neither is the automatic
  response to an uneven class, and both change what the model is actually fitted on. Whichever
  you pick applies to the **training rows only** — resample before the split and the same rows
  land on both sides of it.

## 6. Metrics

**Nothing here maps a dataset onto a metric.** The primary metric is chosen in Step 7 from the
task objective, the class distribution, the cost of each error type, and what the validation
design can actually estimate. This table only says what each candidate is sensitive to, so that
choice can be argued rather than asserted.

| Candidate | Sensitive to | Blind to, or worth caution |
|---|---|---|
| accuracy | nothing but the majority | a rare class can be missed entirely and the score barely moves |
| macro-F1 | every class equally, absent ones included | assumes all classes matter equally, which the task may not |
| weighted-F1 | class frequency | reintroduces the majority's dominance that macro-F1 removes |
| precision | the cost of a false positive | says nothing about the rows the model failed to find |
| recall | the cost of a false negative | says nothing about how many false alarms it took |
| ROC-AUC | ranking, threshold-free | flatters a rare positive class — the large negative class dominates the curve |
| PR-AUC | ranking **of the rare class** | undefined with no positives, and noisy on very few of them |
| confusion matrix | everything, per class | not a scalar; it is what makes the others checkable |

Also fixed: the same protocol and the same ruler for both models, with a confusion matrix each.
Without the matrix the reported scalars cannot be checked.

## 7. Hyperparameters vs experimental configuration

| Hyperparameter (tune it, then report the value) | Configuration (fix it, then report the value) |
|---|---|
| `max_depth`, `min_samples_leaf` | random seed |
| `k` (KNN) | train/test split ratio |
| `C`, `gamma` (SVM), `C`/`penalty` (logistic) | cross-validation fold count |
| `n_estimators`, `learning_rate` | scoring metric for tuning |
| `ccp_alpha` (pruning) | whether a validation split was held out |

Tuning protocol: candidates are scored on the **validation** split and the winner is
reported on the **test** split, which is never touched during selection. Test-set error
is never used to choose a hyperparameter.

### Stopping criteria

A third thing, and not a hyperparameter: a hyperparameter is *searched over*, a stopping
criterion is what *ends the fit*. The report asks for it explicitly.

| Where it comes from | Examples | What to report |
|---|---|---|
| an iterative solver | `max_iter`, `tol` | the cap and tolerance set, and whether the cap was **reached** |
| tree growth | `max_depth`, `min_samples_leaf`, `ccp_alpha` | which constraint ended growth |
| an ensemble | `n_estimators`, `learning_rate`, early stopping | rounds, rate, and what early stopping monitored |
| none | — | say so, and name the constraint that ends the fit instead |

Report the criterion actually in force for the model you shipped. A cap that was reached is a
failure to converge and must be reported as one — the score it produced is not the score of a
converged model. Never write a criterion you did not set.

## 8. Choosing the validation strategy

There is no threshold table here, and that is deliberate. The right split depends on the
dataset in front of you, so the decision is made per dataset from numbers the toolkit
computes (`validation_report`) — never from a constant that happened to suit one CSV. A
cut-off of "50 rare rows" is meaningless without knowing whether the pool has 500 rows or
500,000.

### 8a. Before choosing a split: is the rarest class supportable?

A rare class is not simply real or fake. `rarest_class_evidence` reports evidence on four
separate questions — is the name meaningful, are the rows duplicated, do rows contradict each
other, and do the rows separate locally — and more than one of them can point at a problem at
once. The reading is yours; the responses differ by which question the evidence answers.

| Evidence | Reading |
|---|---|
| a meaningful class name (`fault`, `churn`, a disease code) | supports a substantive class |
| a name like `other`, `unknown`, `n/a`, `?` | may be a catch-all bucket — or a legitimate "other" category the task really has |
| `effective_unique_rows` ≈ `n_rows` | the rows are distinct; nothing looks duplicated |
| `effective_unique_rows` far below `n_rows` | repeated rows may be inflating the sample |
| `contradicted_rows` = 0 | no identical feature vector carries two labels |
| `contradicted_rows` > 0 | *possible* label noise, duplicate conflict, genuine ambiguity, or a missing discriminative column |
| `lift_over_chance` well above 1 | rare rows neighbour each other: the class has local structure here |
| `lift_over_chance` ≈ 1 | these columns give it little local separation |

**Low lift is evidence about the columns, not a verdict on the class.** These are different
claims and they lead to different reports. A real rare class can simply be spread through the
feature space — if its rows are no closer to each other than to the majority, then nothing
*available here* separates it, which says something about the data you have, not about whether
the class is meaningful. Say which you believe and what would settle it.

If the class is substantively meaningful, model it, use metrics that do not average it away,
and state the limitation. If the evidence points to a data problem, work out which one:
de-duplicating, relabelling, adding a missing feature, and merging a genuine catch-all are
different responses to different findings, and only some of them delete rows. Write down what
you concluded and why.

**Cross-validation does not repair an unsupported class.** Averaging *k* estimates of a label
with no signal gives a stable estimate of nothing, and it will look reassuring in the report.

### 8b. Are the rows independent?

Before asking which splitter is better, ask what a row *is*. If several rows describe one
patient, one customer or one session, a random split puts that entity on both sides of the
line, and the model scores well by recognising the entity rather than by learning the task —
with nothing raised to warn you.

`structure_evidence` reports candidate group and time columns **with the evidence for each and
never a verdict** — the same contract as `label_candidates` and `rarest_class_evidence`:

| Evidence | Reading |
|---|---|
| a candidate's name matches an entity convention (`id`, `key`, `patient`, `session`, `account`, `device` …) | **possible** entity key — repetition is what makes it one, so check `rows_per_value_mean` before believing the name |
| `rows_per_value_mean` ≥ 2 with `lift_over_chance` well above 1 **and `is_in_feature_set` false** | the value groups rows that really are neighbours: an entity, not a category |
| `is_in_feature_set` true, whatever the lift | **not** an entity key — it is a column the model predicts with, so rows sharing a value resemble each other *because the value predicts the label*. Grouping on it cuts folds through a predictor, not through entities. spambase's `capital_run_length_total` scores a lift of 46.6 this way |
| `lift_over_chance` ≈ 1 with a solid `rows_per_value_mean` | an ordinary **category** — a row split cannot leak through it |
| withheld: *"only N rows per value on average"* | the value is nearly unique per row, so "same value" means "the same row" — an identifier or a continuous measure, and the ratio measures nothing |
| withheld: *"only about N rows have a same-value nearest neighbour"* | too few events; two near-zero numbers give a meaningless quotient, not a lift |
| any `time_candidates` entry | the split must respect order; see `temporal_folds` below |

A withheld lift is not a failure: it is the honest answer to a question the data cannot
support, and the reason is always printed with it. **The lift only means anything because the
candidate is held out of the feature space** — left in, every column makes its own rows look
close and *every* categorical column appears to be a group key. That is also why an entity key
must not appear in `numeric`/`categorical` *and* be passed as `groups=`.

**If the rows are not independent**, the unit of the split changes, and with it §8c's
arithmetic: with entities the rare class is *g* entities rather than *n* rows, and one entity
crossing the boundary moves all of its rare rows at once.

| Factory | Guarantee it enforces |
|---|---|
| `grouped_holdout(test_size, seed)` | whole entities on each side; yields **positional** indices, so `X.iloc[…]` is correct and `X.loc[…]` is not |
| `stratified_group_folds(n_splits, seed)` | both guarantees at once; refuses when `n_splits` exceeds the number of entities |
| `temporal_folds(n_splits, gap, order)` | raises if `order` is not sorted; **`gap` has no default** — how stale a model may be is a decision |

`fold_consequences(splitter, X, y, groups, classes)` realizes the folds without fitting
anything and reports the group overlap — which must be zero — plus which classes are
**absent** from each fold. Absence matters because macro-F1 averages over the classes it can
*see*: a fold missing a class scores **higher** than its siblings rather than being penalised.
`validation_report(..., groups=..., order=...)` adds the matching blocks at the new unit, and
still makes no recommendation.

### 8c. Will the estimate rank candidates reliably?

Choose the strategy from the dataset's structure — what §8b found — and from how much evidence
the rarest class offers. Four things to weigh:

1. **Start from the rarest class, not the row count.** Selection is on macro-F1, so a 12-row
   class and a 7,000-row class carry *equal* weight in the score, and a split that looks fine
   by row count can be pure noise for the class that decides the macro average. Report the
   rarest-class count, how many of its rows land in one estimate, and the share of its recall
   a single flipped row accounts for.

2. **Choose an honest split.** Holdout, k-fold, repeated k-fold, group-aware and temporal
   validation are all valid; which one fits depends on sample size, class distribution, row
   independence, ordering, and the stability selection needs. Proportional sampling keeps a
   class from being accidentally emptied, and `stratified_split` enforces it — but never break
   group or time structure merely to preserve class proportions. Nor is a cleverer split a
   substitute for §8a: if the class is unsupported, no strategy rescues it.

3. **Prefer stability when selection is noisy.** k-fold averages several validation estimates
   and validates every rare row exactly once, which is why it can win *even when its per-fold
   rare count is lower than a holdout's*; it also leaves the final model trained on the whole
   pool. It is not a √k precision gain, because folds are not independent samples of the data.
   A wider hyperparameter grid needs a more stable estimate, since taking the maximum of many
   noisy scores overfits the split.

4. **Check the arithmetic before committing.** Ordinary stratified k-fold needs `n_splits`
   within the smallest class count, or a fold cannot score that class; group-aware folds are
   constrained by the entities instead (§8b). There is no default fold count — `n_splits` is
   a field in the plan and `run` refuses a value above the cap with the cap in the message.
   The cap is a ceiling rather than a target: at the cap each fold scores a single rare row,
   the worst estimate available.

**It is fine to conclude that no split is adequate.** The validation record the driver writes
to `out/evidence/validation.json` — rare rows per estimate, and folds that lose the class
outright — supplies the arithmetic, and when it cannot carry the claim, the honest report
says so rather than choosing the option that looks better.
rather than choosing the option that looks better. There is no count at which this turns true:
a class of 30 rows in a pool of 400 and a class of 30 in a pool of 400,000 are different
problems, and only the second is thin.

Report the decision with its arithmetic: strategy, rarest-class count, rare rows per estimate,
and what it costs. This is the part of the report a marker can actually check. The only stops
enforced in code are the two that are arithmetic rather than judgement: a one-row class cannot
be stratified at all, and a fold cannot hold fewer than one row of a class.

## 9. Costs of a three-way split

A three-way split carries two costs, and they are easy to confuse:
1. **The data given up.** Validation rows are rows the model never trains on. On an
   imbalanced dataset this bites the minority class first, because that is where the
   examples are scarcest.
2. **A noisier selection signal.** One split is a less stable estimate than k-fold over the
   same rows, so the chosen hyperparameters can shift with the seed.

Do not assume which one moved a result. When both the data and the selected hyperparameters
change between two runs the comparison is confounded, so hold one factor fixed and rerun
before attributing a difference to it. `plan.json` plus a fixed `seed` is what makes that
possible: change one field, run again, and the two `evidence/metrics.json` files are
directly comparable.
