#!/usr/bin/env python
"""Preprocessing primitives - the leak-safe half of a Pipeline.

Mechanics only. The one decision here is `scale`, and it is a required argument with no
default, because whether a model needs comparable feature magnitudes depends on the model
being used and is settled in SKILL.md Step 5.

Everything else is a guarantee rather than a choice:

  * Imputation is fitted INSIDE the transformer, so the median and mode are learned from
    the rows it is fitted on and nothing else. Impute the frame by hand one line before the
    split and the test rows have already leaked into the statistic: it runs, it does not
    warn, and the scores come out looking better than they are.
  * `handle_unknown="ignore"` encodes a category first seen in validation or test as
    all-zeros instead of raising.
  * `remainder="drop"` means nothing reaches the model that the caller did not name.

If you are about to add a default or a policy to this file, it belongs in SKILL.md instead.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    """Build a leak-safe preprocessing block.

    `scale=True` standardises the numeric columns (needed by anything that weighs inputs
    by magnitude: logistic regression, SVM, KNN, a neural net). `scale=False` leaves them
    alone, which is correct for trees - they split on order, so the units are irrelevant.

    The important part is not the flag, it is that imputation happens *here*, inside the
    transformer. When this block is fitted, the median and mode are learned from whatever
    rows it is fitted on and nothing else. Call this once and put it in a Pipeline, and
    the leak cannot happen; impute the frame by hand beforehand and it silently will.
    """
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))

    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore" so a category seen only in validation/test is encoded
        # as all-zeros instead of raising.
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), list(numeric)),
            ("cat", Pipeline(cat_steps), list(categorical)),
        ],
        remainder="drop",
    )
