#!/usr/bin/env python
"""The record of what was decided, and why.

Every choice in this workflow is made at run time by the agent and justified in a report.
Until now that justification existed only as comments in the run script - readable if you
read the whole file, and impossible to summarise, diff or check. `DecisionLog` makes a
decision a first-class object: a choice, a reason, and the alternatives that were rejected.

The two required fields are the point of the whole class:

  * **a reason** - without one, a choice is indistinguishable from a habit.
  * **the alternatives considered** - without them, there was nothing to decide. A choice
    made when no other option was on the table is a default wearing a decision's clothes,
    and `record` refuses it rather than letting it into the report.

`inputs` carries the numbers the reason cites, so a justification is grounded in what the
run actually computed rather than in prose written afterwards to fit the result. A reason
that says "the rarest class is well populated" is an opinion; one that carries
`rarest_rows_per_estimate: 1492.8` is checkable.

The log **records**; it never chooses. Nothing here has an opinion about which model is
better, and nothing here should ever gain one.

If you are about to add a default or a policy to this file, it belongs in SKILL.md instead.
"""

from __future__ import annotations

import json
from pathlib import Path

# The steps a decision can belong to, in workflow order. Recording one keeps the report's
# decision table in the order a marker reads it rather than the order the code ran.
STEPS = [
    "label",
    "cleaning",
    "features",
    "model_selection",
    "preprocessing",
    "validation",
    "metrics",
    "final_fit",
]


class DecisionLog:
    """An ordered record of (step, decision, choice, reason, alternatives, inputs).

    Usage:

        log = DecisionLog()
        log.record("validation", "validation strategy", "20% stratified holdout",
                   reason="the rarest class holds 7,464 rows, so an estimate carries "
                          "~1,493 of them and one flipped row moves its recall by 0.07%",
                   alternatives_considered=["5-fold CV - ~sqrt(5) tighter, 5x the fitting "
                                            "cost, no change in the ranking"],
                   inputs={"rarest_rows_per_estimate": 1492.8})
        log.write("out/decisions.json")
        print(log.render())
    """

    def __init__(self) -> None:
        self.entries: list[dict] = []

    def record(self, step: str, decision: str, choice: str, reason: str,
               alternatives_considered, inputs: dict | None = None) -> dict:
        """Add one decision. Raises if it is not actually a decision.

        `step` should be one of STEPS (it is not enforced to a closed set - a workflow may
        legitimately have a step this list did not anticipate - but a name outside it is
        worth a second look).

        `alternatives_considered` may be a list of strings or a single string; either way
        it must not be empty. `reason` must not be empty. Both are refused rather than
        warned about, because the failure they guard against is invisible downstream: a
        decision table with a blank reason column still renders, still paginates, and still
        reads as complete.
        """
        if not str(reason).strip():
            raise ValueError(
                f"decision {decision!r} has no reason. A choice with no reason cannot be "
                f"told apart from a default, which is the one thing this log exists to "
                f"prevent."
            )

        if alternatives_considered is None:
            alts: list[str] = []
        elif isinstance(alternatives_considered, str):
            alts = [alternatives_considered] if alternatives_considered.strip() else []
        else:
            alts = [str(a) for a in alternatives_considered if str(a).strip()]

        if not alts:
            raise ValueError(
                f"decision {decision!r} lists no alternatives considered. If nothing else "
                f"was on the table then nothing was decided - record it as a configuration "
                f"fact in the run script instead, or name the alternative you rejected and "
                f"why."
            )

        entry = {
            "step": str(step),
            "decision": str(decision),
            "choice": str(choice),
            "reason": str(reason),
            "alternatives_considered": alts,
            "inputs": dict(inputs) if inputs else {},
        }
        self.entries.append(entry)
        return entry

    def by_step(self) -> list[dict]:
        """Entries in workflow order, ties broken by the order they were recorded."""
        ranked = {s: i for i, s in enumerate(STEPS)}
        return sorted(self.entries, key=lambda e: ranked.get(e["step"], len(STEPS)))

    def to_dict(self) -> dict:
        return {
            "_evidence": {"type": "decisions", "schema_version": 1,
                          "producer": "DecisionLog"},
            "n_decisions": len(self.entries),
            "steps_covered": sorted({e["step"] for e in self.entries}),
            "decisions": self.by_step(),
        }

    def write(self, path) -> Path:
        """Write the log as JSON. Returns the path written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def render(self) -> str:
        """A markdown table of the decisions, for pasting into the report.

        The alternatives go in the same cell as the reason rather than in a column of
        their own: the report is two pages, and a five-column table of prose does not fit
        one. They are not dropped - a decision without its rejected alternatives reads as
        inevitable, which is exactly what the log is for.
        """
        rows = ["| Step | Decision | Choice | Why, and what else was considered |",
                "|---|---|---|---|"]
        for e in self.by_step():
            why = e["reason"].strip().rstrip(".")
            alts = "; ".join(a.strip().rstrip(".") for a in e["alternatives_considered"])
            rows.append(
                f"| {e['step']} | {e['decision']} | {e['choice']} | {why}. "
                f"Rejected: {alts}. |"
            )
        return "\n".join(rows)

    def render_plain(self) -> str:
        """The same content for a terminal, where a markdown table wraps badly."""
        out = []
        for e in self.by_step():
            out.append(f"[{e['step']}] {e['decision']}")
            out.append(f"    chose : {e['choice']}")
            out.append(f"    why   : {e['reason']}")
            for a in e["alternatives_considered"]:
                out.append(f"    not   : {a}")
            for k, v in e["inputs"].items():
                out.append(f"    input : {k} = {v}")
            out.append("")
        return "\n".join(out)
