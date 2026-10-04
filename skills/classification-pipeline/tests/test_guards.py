#!/usr/bin/env python
"""Checks for the guards in `paths.py` and `make_pdf.py`.

    python tests/test_guards.py

Every guard here exists because the failure it catches is **silent**. A run that writes its
output beside the machinery, or that skips the workflow and hand-rolls its own script, still
ends with a report that looks finished - so nothing raises and nobody looks. Each test pins
one refusal, the escape hatch that turns it off, and - for the evidence guard - one shape of
hand-written file that the weaker existence-only check used to let through.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from make_pdf import _evidence_problems, main, require_evidence
from paths import SKILL_ROOT, refuse_bad_output_dir


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


class env:
    """Set environment variables for the duration of a `with` block, then restore."""

    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.kw}
        os.environ.update({k: v for k, v in self.kw.items()})
        return self

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ------------------------------------------------------------- evidence fixtures


def valid_eda(**over) -> dict:
    """Exactly what `profile()` returns, trimmed to the keys the guard requires."""
    d = {
        "_evidence": {"type": "eda", "schema_version": 1, "producer": "profile"},
        "n_rows": 31112, "n_features": 15, "label_column": "label", "n_classes": 2,
        "class_counts": {"no": 23645, "yes": 7464}, "imbalance_ratio": 3.1677,
        "numeric_features": ["aptitude_score"], "categorical_features": ["region"],
        "missing_per_column": {}, "missing_total": 0, "duplicate_rows": 0,
    }
    d.update(over)
    return d


def valid_decisions() -> dict:
    return {
        "_evidence": {"type": "decisions", "schema_version": 1,
                      "producer": "DecisionLog"},
        "n_decisions": 1, "steps_covered": ["validation"],
        "decisions": [{
            "step": "validation", "decision": "validation strategy",
            "choice": "5-fold stratified CV", "reason": "the rarest class holds 7,464 rows",
            "alternatives_considered": ["20% holdout"], "inputs": None,
        }],
    }


def valid_metrics(models=("logistic_regression", "decision_tree")) -> dict:
    def one(name):
        return {
            "_evidence": {"type": "evaluation", "schema_version": 1, "producer": "evaluate"},
            "model": name, "accuracy": 0.9, "precision_macro": 0.88,
            "recall_macro": 0.87, "f1_macro": 0.87, "confusion_matrix": [[1, 2], [3, 4]],
            "per_class": {},
        }
    return {n: one(n) for n in models}


class _Skip:
    """Marker for `write_evidence`: leave this file out entirely."""


SKIP = _Skip()


def evidence_dir(out_dir: Path) -> Path:
    """Where the evidence lives: `<out_dir>/evidence/`, one level below `report.md`.

    `make_pdf.require_evidence` resolves it as `Path(md_path).parent / "evidence"`, and
    `driver.py` writes it there. A test that stages evidence anywhere else is testing a
    layout the workflow does not have - which is what the earlier version of this file did.
    """
    d = Path(out_dir) / "evidence"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_evidence(out_dir: Path, eda=None, decisions=None, metrics=None) -> Path:
    """Write valid evidence into `<out_dir>/evidence/`. Returns that directory.

    `None` means the valid default; `SKIP` omits the file entirely.
    """
    target = evidence_dir(out_dir)
    payloads = {
        "eda.json": valid_eda() if eda is None else eda,
        "decisions.json": valid_decisions() if decisions is None else decisions,
        "metrics.json": valid_metrics() if metrics is None else metrics,
    }
    for name, value in payloads.items():
        if value is SKIP:
            continue
        (target / name).write_text(json.dumps(value), encoding="utf-8")
    return target


REPORT = "# Test report\n\nOne short paragraph, enough to render a page.\n"


# --------------------------------------------------------------- write target


def test_writing_inside_the_skill_directory_is_refused():
    e = raises(lambda: refuse_bad_output_dir(SKILL_ROOT / "out" / "report.pdf"),
               SystemExit, contains="skill directory")
    assert "experiment folder" in str(e), str(e)


def test_writing_into_a_dot_opencode_tree_is_refused():
    """The case that actually happened: cwd was `.opencode`, so OUT landed beside the
    machinery rather than inside it - which the older skill-directory-only check passed."""
    with tempfile.TemporaryDirectory() as d:
        e = raises(lambda: refuse_bad_output_dir(Path(d) / ".opencode" / "out"),
                   SystemExit, contains=".opencode")
        assert "experiment folder" in str(e), str(e)


def test_writing_into_a_directory_holding_skill_md_is_refused():
    """A skill directory under any name, not just this one."""
    with tempfile.TemporaryDirectory() as d:
        other = Path(d) / "some-other-skill"
        other.mkdir()
        (other / "SKILL.md").write_text("---\nname: x\n---\n", encoding="utf-8")
        raises(lambda: refuse_bad_output_dir(other), SystemExit, contains="SKILL.md")


def test_writing_outside_the_skill_directory_is_allowed():
    with tempfile.TemporaryDirectory() as d:
        refuse_bad_output_dir(Path(d) / "out" / "report.pdf")   # must not raise


def test_the_write_guard_can_be_turned_off():
    with env(SKILL_ALLOW_LOCAL_OUT="1"):
        refuse_bad_output_dir(SKILL_ROOT / "out" / "report.pdf")   # must not raise


def test_a_writing_primitive_refuses_a_dot_opencode_target():
    """The guard is wired into the primitives, not only into the PDF renderer.

    `make_pdf` is the last step and a run that goes wrong often never reaches it. The
    primitives are where a drifting run actually writes.
    """
    import numpy as np
    import pandas as pd
    from data import save_figures
    from evaluation import save_confusion_pair

    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "label": [0, 0, 1, 1]})
    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / ".opencode" / "out"
        raises(lambda: save_figures(df, "label", ["a"], bad), SystemExit, contains=".opencode")
        raises(lambda: save_confusion_pair(np.array([0, 0, 1, 1]),
                                           {"m": np.array([0, 0, 1, 1])}, bad),
               SystemExit, contains=".opencode")
        # and a good target still writes
        good = save_figures(df, "label", ["a"], Path(d) / "out")
        assert good.name == "figures" and (good / "class_balance.png").exists()


def test_reading_a_markdown_from_the_skill_dir_is_not_a_write():
    """The guard covers the write target only, so a report living in the skill tree still
    renders.

    The whole run directory is staged inside the skill tree - markdown and evidence both -
    because that is the shape a re-render of an old run actually has.
    """
    staged = SKILL_ROOT / f"_tmp_guard_test_{os.getpid()}"
    try:
        staged.mkdir()
        write_evidence(staged)
        (staged / "report.md").write_text(REPORT, encoding="utf-8")
        with tempfile.TemporaryDirectory() as d:
            out_pdf = Path(d) / "report.pdf"
            assert main(["--md", str(staged / "report.md"),
                         "--out", str(out_pdf)]) == 0
            assert out_pdf.exists()
    finally:
        # The staged directory lives in the tree that gets submitted - never leave it.
        # `rmtree`, not a glob of `unlink`: the evidence lives in a subdirectory now, and
        # `unlink` on a directory is an error rather than a fix.
        shutil.rmtree(staged, ignore_errors=True)


# ------------------------------------------------------------------- evidence


def test_rendering_without_the_evidence_files_is_refused():
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d) / "out"
        out_dir.mkdir()
        md = out_dir / "report.md"
        md.write_text(REPORT, encoding="utf-8")
        e = raises(lambda: require_evidence(md), SystemExit, contains="eda.json")
        for name in ("eda.json", "decisions.json", "metrics.json"):
            assert name in str(e), (name, str(e))


def test_rendering_refuses_when_only_some_of_the_evidence_is_there():
    """Absent files are named; valid ones present are not complained about."""
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d) / "out"
        out_dir.mkdir()
        md = out_dir / "report.md"
        md.write_text(REPORT, encoding="utf-8")
        write_evidence(out_dir, metrics=SKIP)
        e = raises(lambda: require_evidence(md), SystemExit)
        assert "metrics.json" in str(e), str(e)
        assert "eda.json" not in str(e), str(e)


def test_a_missing_file_does_not_hide_problems_in_the_ones_that_are_there():
    """An earlier version returned as soon as one file was absent, so a directory holding a
    hand-written eda.json and no decisions.json reported only the absent file."""
    hand_written = {k: v for k, v in valid_eda().items()
                    if k not in ("_evidence", "missing_per_column", "missing_total",
                                 "duplicate_rows")}
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d) / "out"
        out_dir.mkdir()
        md = out_dir / "report.md"
        md.write_text(REPORT, encoding="utf-8")
        ev = evidence_dir(out_dir)
        (ev / "eda.json").write_text(json.dumps(hand_written), encoding="utf-8")
        (ev / "metrics.json").write_text(json.dumps(valid_metrics()), encoding="utf-8")
        problems = _evidence_problems(ev)
        assert any("decisions.json" in p for p in problems), problems
        assert any("eda.json" in p for p in problems), problems
        assert not any("metrics.json" in p for p in problems), problems


def test_a_hand_written_eda_json_is_refused():
    """The real failure: eight of fourteen keys, and exactly the cleaning ones missing.

    These three are the whole of section 1's "issues found" line.
    """
    hand_written = {
        "n_rows": 31112, "n_features": 15, "label_column": "label", "n_classes": 2,
        "class_counts": {"no": 23645, "yes": 7464}, "imbalance_ratio": 3.1677,
        "numeric_features": ["aptitude_score"], "categorical_features": ["region"],
    }
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d) / "out"
        out_dir.mkdir()
        md = out_dir / "report.md"
        md.write_text(REPORT, encoding="utf-8")
        write_evidence(out_dir, eda=hand_written)
        e = raises(lambda: require_evidence(md), SystemExit, contains="missing_total")
        for key in ("missing_per_column", "missing_total", "duplicate_rows"):
            assert key in str(e), (key, str(e))
        assert "profile()" in str(e), str(e)


def test_an_eda_json_with_the_wrong_producer_is_refused():
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d) / "out"
        out_dir.mkdir()
        md = out_dir / "report.md"
        md.write_text(REPORT, encoding="utf-8")
        write_evidence(out_dir, eda=valid_eda(
            **{"_evidence": {"type": "eda", "schema_version": 1, "producer": "hand"}}))
        raises(lambda: require_evidence(md), SystemExit, contains="producer")


def test_a_decision_log_with_no_entries_is_refused():
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d) / "out"
        out_dir.mkdir()
        md = out_dir / "report.md"
        md.write_text(REPORT, encoding="utf-8")
        write_evidence(out_dir, decisions=dict(valid_decisions(), decisions=[]))
        raises(lambda: require_evidence(md), SystemExit, contains="no decisions")


def test_a_metric_entry_missing_a_field_is_refused():
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d) / "out"
        out_dir.mkdir()
        md = out_dir / "report.md"
        md.write_text(REPORT, encoding="utf-8")
        partial = valid_metrics()
        del partial["decision_tree"]["per_class"]
        write_evidence(out_dir, metrics=partial)
        raises(lambda: require_evidence(md), SystemExit, contains="per_class")


def test_metrics_with_one_model_is_refused():
    """The run that stopped early still renders a comparison table with a blank column."""
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d) / "out"
        out_dir.mkdir()
        md = out_dir / "report.md"
        md.write_text(REPORT, encoding="utf-8")
        write_evidence(out_dir, metrics=valid_metrics(("logistic_regression",)))
        e = raises(lambda: require_evidence(md), SystemExit, contains="expected 2")
        assert "logistic_regression" in str(e), str(e)


def test_a_flattened_classification_report_is_refused():
    """A sklearn `classification_report` dumped straight to JSON was the actual shape left
    behind by the run this guard was written for. It must be reported as a shape problem,
    not dissected key by key."""
    flattened = {"0.0": {"precision": 0.9}, "1.0": {"precision": 0.5},
                 "accuracy": 0.83, "macro avg": {"precision": 0.7},
                 "weighted avg": {"precision": 0.8}}
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d) / "out"
        out_dir.mkdir()
        md = out_dir / "report.md"
        md.write_text(REPORT, encoding="utf-8")
        ev = write_evidence(out_dir, metrics=flattened)
        problems = _evidence_problems(ev)
        assert any("expected 2" in p for p in problems), problems
        shape = [p for p in problems if "per-model" in p]
        assert len(shape) == 1, f"expected one shape line, got {shape}"
        assert not any("metrics.json[0.0]" in p for p in problems), problems


def test_rendering_with_the_evidence_files_present_is_allowed():
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d) / "out"
        out_dir.mkdir()
        md = out_dir / "report.md"
        md.write_text(REPORT, encoding="utf-8")
        write_evidence(out_dir)
        require_evidence(md)                              # must not raise


def test_the_evidence_guard_can_be_turned_off():
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d) / "out"
        out_dir.mkdir()
        md = out_dir / "report.md"
        md.write_text(REPORT, encoding="utf-8")
        with env(SKILL_ALLOW_MISSING_EVIDENCE="1"):
            require_evidence(md)                          # must not raise


def test_main_refuses_end_to_end_when_the_evidence_is_missing():
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d) / "out"
        out_dir.mkdir()
        (out_dir / "report.md").write_text(REPORT, encoding="utf-8")
        raises(lambda: main(["--md", str(out_dir / "report.md"),
                             "--out", str(out_dir / "report.pdf")]),
               SystemExit, contains="missing")
        assert not (out_dir / "report.pdf").exists()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
