#!/usr/bin/env python
"""A registry of model families, and the facts that let you choose between them.

The point of this file is to save the *lookup*, not the *decision*.

What it may hold
----------------
  * **Facts about an algorithm** - whether it needs scaled inputs, whether it can take
    one-hot columns, what shape of boundary it can express, whether it can produce
    probabilities, what it assumes. These do not depend on the dataset; they are properties
    of the estimator, and a report is expected to cite them.

    These facts are what make model *selection* possible without this file making it:
    the registry says a tree reaches a staircase and logistic regression does not, and
    leaves it to you to notice that a dataset with a suspected interaction needs one of
    each. There is deliberately no `recommended` key - a recommendation here would be a
    choice made by whoever wrote this file, which is exactly the transfer this module
    exists to prevent.
  * **Constructor settings that are not modelling choices.** `max_iter=2000` on logistic
    regression only stops a convergence warning; it decides nothing about the fit's
    quality. `probability=True` on an SVM only makes `predict_proba` exist at all.

What it may NOT hold
--------------------
Any default for a value the agent has to choose *for this dataset*: `C`, `max_depth`,
`class_weight`, `n_neighbors`, `n_estimators`, `gamma`, `var_smoothing`. Putting those here
would mean the choice was made by whoever wrote this file, for whatever dataset they had in
mind, and inherited silently by everyone else - which is the failure this skill exists to
avoid. `make_model` takes them as keyword arguments, and SKILL.md Step 4 is where you decide
them.

`scaling` and `one_hot` are prose with a reason rather than a boolean. "Mandatory" and
"recommended" are different claims and a boolean cannot tell them apart; the report is
graded on the difference.

If you are about to add a default or a policy to this file, it belongs in SKILL.md instead.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

MODEL_REGISTRY: dict[str, dict] = {
    "decision_tree": {
        # `inductive_bias` is the coarse name of the shape of boundary a family can reach -
        # coarser than `nonlinearity`, which is the accurate sentence. It exists so that
        # "these two differ in inductive bias" (SKILL.md Step 4) is checkable rather than
        # asserted: two families differ when these strings differ. `decision_tree`,
        # `random_forest` and `gradient_boosting` share one, because a forest is trees
        # averaged and a boosted model is trees fitted to residuals - all three reach the
        # same axis-aligned staircase, so a pair of them tells the report nothing.
        "inductive_bias": "axis-aligned staircase",
        "family": "Decision tree",
        "estimator": DecisionTreeClassifier,
        "constructor": {},
        "scaling": "not required - a tree splits on order, so the units a column is "
                   "measured in cannot change where a split falls",
        "one_hot": "optional - it can also take label-encoded columns, because an "
                   "axis-aligned split can isolate a category either way",
        "nonlinearity": "non-linear - each split is a step on one column, so any "
                        "axis-aligned staircase is reachable; a boundary that runs "
                        "diagonally has to be approximated by many small steps",
        "supports_probability": "yes, as the class proportions in the reached leaf",
        "assumes": "nothing about the shape of the boundary, which is why it finds "
                   "interactions a linear model cannot - but it will happily memorise "
                   "noise unless depth or leaf size is limited",
        "main_hyperparameters": ["max_depth", "min_samples_leaf", "criterion", "ccp_alpha"],
        "interpretability": "high - the tree can be drawn and read",
    },
    "random_forest": {
        "family": "Random forest",
        "inductive_bias": "axis-aligned staircase",
        "estimator": RandomForestClassifier,
        "constructor": {},
        "scaling": "not required - an ensemble of trees, and each member still splits on "
                   "order",
        "one_hot": "optional - same reason as a single tree",
        "nonlinearity": "non-linear - the same staircase as a single tree, averaged over "
                        "many of them; adding trees removes variance, it does not smooth "
                        "the staircase away",
        "supports_probability": "yes, as the mean of the trees' leaf proportions",
        "assumes": "that averaging decorrelated trees reduces the variance a single tree "
                   "has; it does not remove bias, so a forest of stumps is still stumps",
        "main_hyperparameters": ["n_estimators", "max_depth", "max_features",
                                 "min_samples_leaf"],
        "interpretability": "low-medium - feature importances, not a readable rule set",
    },
    "gradient_boosting": {
        "family": "Gradient boosting",
        "inductive_bias": "axis-aligned staircase",
        "estimator": GradientBoostingClassifier,
        "constructor": {},
        "scaling": "not required - still trees, still splitting on order",
        "one_hot": "optional - each split isolates one category at a time, so an integer "
                   "code and a one-hot column give the same set of reachable splits",
        "nonlinearity": "non-linear - a sum of small staircases, each fitted to the "
                        "previous ones' errors, which is why it reaches smoother and "
                        "lower-order boundaries than one deep tree",
        "supports_probability": "yes, through the logistic link on the summed scores",
        "assumes": "that the errors of a weak learner can be fitted by the next one; "
                   "usually the strongest of these on tabular data, and the slowest",
        "main_hyperparameters": ["n_estimators", "learning_rate", "max_depth",
                                 "subsample"],
        "interpretability": "low-medium",
    },
    "logistic_regression": {
        "family": "Logistic regression",
        "inductive_bias": "linear in the log-odds",
        "estimator": LogisticRegression,
        # max_iter=2000 is not a modelling choice: the default of 100 frequently stops
        # before the solver has converged and emits a ConvergenceWarning, which silently
        # leaves the coefficients short of the optimum. Raising the iteration ceiling
        # changes nothing except letting the fit finish. C, penalty and class_weight are
        # deliberately absent - those decide the model and are yours to choose.
        "constructor": {"max_iter": 2000},
        "scaling": "generally recommended, and close to required in practice - a column "
                   "measured in hundreds can dominate one measured in single digits purely "
                   "because of its units, since a single penalty is applied to coefficients "
                   "that are compared against each other; the solver also converges more "
                   "reliably on a scaled matrix",
        "one_hot": "categories must be represented numerically, and one-hot is the usual "
                   "choice - the estimator multiplies a single coefficient per column, so "
                   "an integer-coded category would be read as an ordering it does not "
                   "have",
        "nonlinearity": "linear in the log-odds - a curved boundary is not reachable from "
                        "the raw columns, it has to be written in by hand as an "
                        "interaction or a squared term",
        "supports_probability": "yes, by construction",
        "assumes": "that the log-odds of the outcome are linear in the features; one-hot "
                   "columns are treated as independent, which is not literally true when "
                   "the categories are correlated",
        "main_hyperparameters": ["C", "penalty", "solver", "class_weight"],
        "interpretability": "high - one signed coefficient per feature",
    },
    "knn": {
        "family": "K-nearest neighbours",
        "inductive_bias": "local similarity",
        "estimator": KNeighborsClassifier,
        "constructor": {},
        "scaling": "effectively required - the whole method is a distance, and an unscaled "
                   "column measured in hundreds contributes hundreds of times more to it; "
                   "there is no version of the method that recovers from this",
        "one_hot": "categories must be numeric, and one-hot is common but imperfect: "
                   "one-hot columns make every category equidistant from every other and "
                   "dilute the numeric distances",
        "nonlinearity": "non-linear, and arbitrarily so - the boundary is whatever the "
                        "local vote makes it, which is as wiggly as the data and just as "
                        "wiggly in the gaps where there is no data",
        "supports_probability": "yes, as the neighbour vote share",
        "assumes": "that proximity in feature space means similarity of outcome, which "
                   "degrades as the number of columns grows - in high dimensions the "
                   "nearest and farthest neighbour are nearly the same distance away",
        "main_hyperparameters": ["n_neighbors", "weights", "metric"],
        "interpretability": "low - a prediction is a set of training rows",
    },
    "naive_bayes": {
        "family": "Naive Bayes (GaussianNB)",
        "inductive_bias": "per-column independence",
        "estimator": GaussianNB,
        "constructor": {},
        # Only GaussianNB is wired up. MultinomialNB and BernoulliNB are separate
        # estimators with different data assumptions, and choosing among them is a
        # modelling decision, so it is not made here.
        "scaling": "not required - each column is summarised by its own mean and variance",
        "one_hot": "depends on the variant - GaussianNB models continuous columns, so a "
                   "categorical column has to be encoded numerically; MultinomialNB wants "
                   "non-negative counts and BernoulliNB wants binary present/absent",
        "nonlinearity": "non-linear, within the variant's assumptions - each column is "
                        "fitted on its own and the results multiplied, so an interaction "
                        "between two columns is not reachable at all",
        "supports_probability": "yes, though the probabilities are poorly calibrated",
        "assumes": "that the columns are independent given the class - almost never true, "
                   "which is why it is a fast baseline rather than a final answer",
        "main_hyperparameters": ["var_smoothing"],
        "interpretability": "medium",
    },
    "svm": {
        "family": "Support vector machine",
        "inductive_bias": "kernel margin",
        "estimator": SVC,
        # probability=True exists so that predict_proba does at all: without it SVC
        # exposes only decision_function, and evaluate() cannot compute ROC-AUC or PR-AUC.
        # It is not free - it runs an internal cross-validation to fit the calibration -
        # so drop it with make_model("svm", probability=False) if you only need labels.
        "constructor": {"probability": True},
        "scaling": "generally recommended, and close to required with an rbf or polynomial "
                   "kernel - the margin is measured as a distance, so an unscaled column "
                   "sets the margin by itself",
        "one_hot": "categories must be represented numerically, and one-hot is a common "
                   "choice - the kernel is a dot product, and an integer code would enter "
                   "it as a magnitude, so category 3 would sit three times as far from "
                   "category 1 as category 2 does",
        "nonlinearity": "decided by the kernel, not by the family - linear with a linear "
                        "kernel, non-linear with rbf or polynomial; the kernel is a "
                        "hyperparameter, so this one is yours to set",
        "supports_probability": "only with probability=True, which costs an internal "
                                "cross-validation",
        "assumes": "that a margin exists - that a boundary separating the classes can be "
                   "drawn with room to spare; it also does not scale to large n",
        "main_hyperparameters": ["C", "kernel", "gamma"],
        "interpretability": "low",
    },
}

FORBIDDEN_KEYS = ("C", "max_depth", "min_samples_leaf", "class_weight", "n_neighbors",
                  "n_estimators", "learning_rate", "gamma", "var_smoothing", "penalty")

# What the `scaling` prose above already says, as a flag - so `screen.py` can build a
# pipeline for every family without parsing English, and so a test can assert the two
# agree. This is NOT the plan's decision: `models[].scale` is still chosen per model in
# Step 5 and recorded there. This is only the value a *screen* uses, where there is no plan
# to read one from yet.
SCALING_REQUIRED = {
    "decision_tree": False,       # splits on order, so units cannot move a split
    "random_forest": False,       # an ensemble of the same splits
    "gradient_boosting": False,   # still trees
    "logistic_regression": True,  # the solver and the shared L2 penalty both want a scale
    "knn": True,                  # the method *is* a distance
    "naive_bayes": False,         # each column is fitted on its own
    "svm": True,                  # the margin is measured as a distance
}


def model_names() -> list[str]:
    """Every name `make_model` accepts."""
    return sorted(MODEL_REGISTRY)


def make_model(name: str, **params):
    """Build an unfitted estimator by name, then apply `params` on top.

    A typo raises and lists the valid names. Falling back to a default model would be the
    same silent failure this package keeps guarding against: the run would complete, the
    report would contain a table, and the model in it would not be the one that was asked
    for.

    Nothing dataset-specific is filled in. `params` is where you supply exactly the values
    you decided on in SKILL.md Step 4 - `make_model("decision_tree", max_depth=8)`,
    `make_model("logistic_regression", C=1.0, class_weight="balanced")`. Omit one and
    scikit-learn's own default applies, which is a fact you should then have a reason for
    rather than a choice this file made for you.
    """
    key = str(name).strip().lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown model {name!r}; available: {', '.join(model_names())}"
        )
    spec = MODEL_REGISTRY[key]
    settings = dict(spec["constructor"])
    settings.update(params)
    return spec["estimator"](**settings)


def _clip(text: str, width: int) -> str:
    """First clause of a prose field, cut to the column width.

    Every one of these fields is full prose ending in ` - <the reason>`, so the summary
    column takes the clause before the dash. It is then cut to fit: a summary that runs
    past its column silently pushes the next one along and the table stops being readable,
    which is the same class of quiet wrongness these fields exist to avoid.
    """
    head = text.split(" - ")[0].split(",")[0].strip()
    return head if len(head) <= width else head[:width - 1].rstrip() + "…"


def describe_models() -> str:
    """Render the registry's capability table for a terminal.

    This is the machine-readable copy of what `reference.md` §2 and §3 say in prose. If the
    two ever disagree, this one is what `make_model` actually does.

    Column widths are **computed from the content**, not pinned to constants. A pinned width
    is a claim about how long a prose field is, and the prose is edited far more often than
    the constant - so the row outgrows the rule meant to underline it, and the table stops
    being readable without anything raising.
    """
    names = model_names()
    head = ("model family", "inductive bias", "scaling", "one-hot", "nonlinearity",
            "probability")
    fields = ("family", "inductive_bias", "scaling", "one_hot", "nonlinearity",
              "supports_probability")
    caps = {"inductive bias": 26, "scaling": 16, "one-hot": 15, "nonlinearity": 15,
            "probability": 18}
    rows = [[n] + [_clip(MODEL_REGISTRY[n][f], caps[h])
                   for f, h in zip(fields[1:], head[1:])] for n in names]
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(head)]

    rule = "-" * (sum(widths) + 2 * (len(head) - 1))
    lines = ["  ".join(h.ljust(w) for h, w in zip(head, widths)).rstrip(), rule]
    lines += ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows]

    lines += ["", "main hyperparameters (yours to choose per dataset, never defaulted here):"]
    lines += [f"  {n:<21} {', '.join(MODEL_REGISTRY[n]['main_hyperparameters'])}"
              for n in names]

    lines += [
        "",
        "Each entry is prose with its reason, not a switch - read the full text in",
        "MODEL_REGISTRY. `assumes` is what a report should cite when it names a violated",
        "assumption.",
        "",
        "`inductive bias` is the coarse grouping, and it is the column Step 4's pair rule",
        "reads: two families differ in bias when this string differs. The three tree",
        "families share one on purpose - a forest is trees averaged and a boosted model is",
        "trees fitted to residuals, so a pair of them reaches one shape of boundary twice.",
        "",
        "This table is what you choose FROM, not what chooses. It carries no ranking and no",
        "`recommended` flag on purpose. Read it against the dataset's own numbers from",
        "`model_selection_evidence(...)`, and eliminate a family only for a reason that names",
        "that dataset (SKILL.md Step 4).",
    ]
    return "\n".join(lines)
