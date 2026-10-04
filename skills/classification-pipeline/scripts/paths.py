#!/usr/bin/env python
"""Where output may and may not go - one definition, used by every primitive that writes.

This is the rule a run can break while everything still *looks* finished. The figures
render, the JSON parses, the report builds, and the whole set is sitting in a directory
nobody will open. Nothing downstream can tell - so it is checked in code, at each place
that writes, rather than asked for in prose. SKILL.md states the rule for a reader; this
module is what makes breaking it stop the run.

Three locations are refused:

  1. **inside the skill directory** - the machinery. Artefacts written here are inherited
     by the next run as though they were its own, and they are not in the folder the
     marker opens.
  2. **anywhere under a harness directory** - beside the machinery rather than inside it.
     This is the case that actually happened: SKILL.md said to compute `OUT` from
     `Path.cwd()`, and when the harness had `cd`-ed into `.opencode` that resolved to
     `.opencode/out/` - one level up from `skills/`, so it is *outside* the skill
     directory (check 1 passes it) and outside the submitted folder (nobody notices).
     `HARNESS_DIRS` names the directories, so the rule is not a rule about opencode.
  3. **a directory that holds `SKILL.md`** - a skill directory under some other name.

A refusal raises `SystemExit`, deliberately: `SystemExit` is not caught by
`except Exception`, so error handling written around a call site cannot swallow the
tripwire and carry on with output in the wrong place.
"""

from __future__ import annotations

import os
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parent.parent   # holds SKILL.md and scripts/

ALLOW_ENV = "SKILL_ALLOW_LOCAL_OUT"

# ------------------------------------------------------------------ the layout
# Named once, here, because every module that writes or reads an artefact has to agree on
# it. A second copy of "evidence" spelled slightly differently - or worse, `metrics.json`
# in one place and `out/evidence/metrics.json` in another - is how a renderer ends up
# looking for a file the run did write, failing, and being "fixed" by writing the file by
# hand. That is the failure mode this whole design exists to remove.
OUT_NAME = "out"
EVIDENCE_SUBDIR = "evidence"
FIGURES_SUBDIR = "figures"
PLAN_NAME = "plan.json"
REPORT_NAME = "report.md"
PDF_NAME = "report.pdf"

HARNESS_DIRS = (".opencode", ".claude")


def _why_refused(path) -> str | None:
    """The reason this path may not hold output, or None if it may."""
    p = Path(path).resolve()
    if p == SKILL_ROOT or SKILL_ROOT in p.parents:
        return "inside the skill directory"
    hit = [d for d in HARNESS_DIRS if d in p.parts]
    if hit:
        return f"inside a {'/'.join(hit)} tree"
    if (p / "SKILL.md").exists():
        return "a directory holding SKILL.md"
    return None


def experiment_root(path) -> Path:
    """Resolve `--experiment`, and refuse it if output may not go there.

    The one place the experiment folder is decided. `driver.py` takes it as a required
    argument and never consults the working directory: computing it from `Path.cwd()` is
    what put a run's output in `.opencode/out/` - beside the machinery, outside the folder
    the marker opens, and nowhere anything would look. Passing it explicitly is the
    difference between an operational fix ("start the harness in the right directory") and
    one that holds whatever directory the harness happens to be in.
    """
    p = Path(path).expanduser().resolve()
    refuse_bad_output_dir(p)
    if p.exists() and not p.is_dir():
        raise SystemExit(f"refusing to write output: {p} exists and is not a directory")
    return p


def refuse_bad_output_dir(*paths) -> None:
    """Refuse to write output outside the experiment folder. Silent when it is fine.

    Set `SKILL_ALLOW_LOCAL_OUT=1` when developing the skill itself, where writing beside
    the machinery is the point.

    Refuse rather than redirect: quietly relocating a path the caller named is its own
    surprise, and the caller learns nothing about having asked for the wrong one.
    """
    if os.environ.get(ALLOW_ENV) == "1":
        return
    for p in paths:
        if p is None:
            continue
        why = _why_refused(p)
        if why:
            raise SystemExit(
                f"refusing to write output: {Path(p).resolve()}\n"
                f"  reason: {why}\n"
                f"  the experiment folder is the one that CONTAINS the harness directory "
                f"({', '.join(HARNESS_DIRS)}) -\n"
                f"  not the harness directory itself, and not the skill directory. Output "
                f"belongs in\n"
                f"  <experiment folder>/{OUT_NAME}/. Pass it explicitly with "
                f"--experiment <dir>.\n"
                f"  (set {ALLOW_ENV}=1 only when developing the skill itself)"
            )
