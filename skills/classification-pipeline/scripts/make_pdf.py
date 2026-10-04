#!/usr/bin/env python
"""Render a filled report-template.md into a PDF of at most two pages.

    python make_pdf.py --md report.md --out report.pdf \
        --student-id 123456 --student-name "Ada Lovelace" --variant "Variant-2" \
        --model-version "<the model you actually ran as>" \
        --harness "<API | ChatUI | Agent Harness, and its version>" \
        --dataset "<file>, <n> rows x <m> features"

Uses matplotlib's PDF backend, so it needs no LaTeX, pandoc or reportlab.
Supports the markdown subset used by report-template.md: headings, bullets,
pipe tables, paragraphs, images and HTML comments.

Inline `**bold**` and `` `code` `` are rendered rather than stripped. Line breaking
still runs on the marker-free text, so pagination does not move when emphasis is
added or removed - which is what keeps the two-page budget honest. See `_runs`.

The assignment asks for the matric number, name, "IN6227-Assignment-1" and the
variant at the top of the first page - not on a separate cover sheet - and caps
the whole submission at two pages. So the identity block is drawn as a header on
page 1 and the body flows underneath it. A "Cover" section inside the markdown
is skipped rather than duplicated.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.image import imread
from matplotlib.lines import Line2D

from paths import EVIDENCE_SUBDIR, SKILL_ROOT, refuse_bad_output_dir

PAGE_W, PAGE_H = 8.27, 11.69          # A4 inches
LEFT, RIGHT = 0.08, 0.92
TOP, BOTTOM = 0.93, 0.07
BODY_PT = 8.5
LINE_FACTOR = 1.42
MONO_PT = 7.3
MAX_COLS = 108                        # monospace characters that fit the text width
CODE_SCALE = 0.92                     # inline code is monospace, one notch smaller

COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
CODE_RE = re.compile(r"`(.+?)`")
INLINE_RE = re.compile(r"\*\*(.+?)\*\*|`([^`]+)`")

H1, H2, H3 = 13.0, 10.8, 9.4
SPACE_PT = {H1: 12.0, H2: 9.0, H3: 6.0}

SKILL_ROOT = SKILL_ROOT   # re-exported for callers that locate the tree from here


EVIDENCE_FILES = ("eda.json", "decisions.json", "metrics.json")

# What each file must contain, not merely that it exists. The keys are the ones the report
# reads: `missing_per_column`, `missing_total` and `duplicate_rows` are the whole of section
# 1's "issues found" line, and `decisions.json` is section-by-section justification. A run
# that hand-rolls its own EDA writes the keys it happened to think of - in the run this was
# written for, eight of fourteen, with exactly the three cleaning keys missing - and the
# resulting report reads as complete while its cleaning section has nothing behind it.
EDA_REQUIRED = (
    "n_rows", "n_features", "label_column", "n_classes", "class_counts",
    "imbalance_ratio", "numeric_features", "categorical_features",
    "missing_per_column", "missing_total", "duplicate_rows",
)
DECISION_REQUIRED = ("step", "decision", "choice", "reason", "alternatives_considered")
METRIC_REQUIRED = ("model", "accuracy", "precision_macro", "recall_macro", "f1_macro",
                   "confusion_matrix", "per_class")
EXPECTED_MODELS = 2                   # the report's comparison table has two columns


def _read_json(path: Path):
    """Parse one evidence file. Returns (value, problem-line) - never raises."""
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as e:
        return None, f"{path.name}: not valid JSON ({e.msg} at line {e.colno})"
    except OSError as e:
        return None, f"{path.name}: cannot be read ({e.strerror})"


def _evidence_problems(out_dir: Path) -> list[str]:
    """Everything wrong with the evidence beside a report, as lines a reader can act on.

    Every file is checked independently. An earlier version returned as soon as one file
    was missing, which meant a directory holding a hand-written `eda.json` and no
    `decisions.json` reported only the missing file - the fake sailed through behind it.
    """
    problems: list[str] = []
    present = set()
    for name in EVIDENCE_FILES:
        if (out_dir / name).exists():
            present.add(name)
        else:
            problems.append(f"{name}: missing")

    if "eda.json" in present:
        eda, err = _read_json(out_dir / "eda.json")
        if err:
            problems.append(err)
        elif not isinstance(eda, dict):
            problems.append(f"eda.json: expected an object, got {type(eda).__name__}")
        else:
            missing = [k for k in EDA_REQUIRED if k not in eda]
            if missing:
                problems.append(
                    f"eda.json: missing {', '.join(missing)} - these are computed by "
                    f"`profile()`; a dict written by hand usually has only the keys its "
                    f"author thought of")
            producer = (eda.get("_evidence") or {}).get("producer")
            if producer != "profile":
                problems.append(
                    f"eda.json: _evidence.producer is {producer!r}, expected 'profile' - "
                    f"the file was not written by profile()")

    if "decisions.json" in present:
        decisions, err = _read_json(out_dir / "decisions.json")
        if err:
            problems.append(err)
        elif not isinstance(decisions, dict) and not isinstance(decisions, list):
            problems.append("decisions.json: expected an object or a list of entries")
        else:
            entries = (decisions.get("decisions") if isinstance(decisions, dict)
                       else decisions)
            producer = ((decisions.get("_evidence") or {})
                        if isinstance(decisions, dict) else {}).get("producer")
            if producer != "DecisionLog":
                problems.append(
                    f"decisions.json: _evidence.producer is {producer!r}, expected "
                    f"'DecisionLog' - the file was not written by DecisionLog")
            if not entries:
                problems.append(
                    "decisions.json: no decisions recorded - every choice in the report "
                    "needs a DecisionLog entry with the alternative that was rejected")
            else:
                for i, e in enumerate(entries):
                    missing = [k for k in DECISION_REQUIRED if k not in (e or {})]
                    if missing:
                        problems.append(
                            f"decisions.json: entry {i} is missing {', '.join(missing)}")

    if "metrics.json" in present:
        metrics, err = _read_json(out_dir / "metrics.json")
        if err:
            problems.append(err)
        elif not isinstance(metrics, dict):
            problems.append("metrics.json: expected an object keyed by model name")
        else:
            scored = {k: v for k, v in metrics.items() if not k.startswith("_")}
            odd = [k for k, v in scored.items() if not isinstance(v, dict)]
            if len(scored) != EXPECTED_MODELS:
                problems.append(
                    f"metrics.json: {len(scored)} model(s) scored, expected "
                    f"{EXPECTED_MODELS} ({', '.join(scored) or 'none'}) - the report's "
                    f"comparison table needs two columns, and with one the missing "
                    f"column reads as a model that lost rather than a run that stopped "
                    f"early")
            if odd:
                # One line, not one per key: a sklearn `classification_report` dumped to
                # JSON is several top-level keys, and listing each one buries the single
                # fact that the shape is wrong.
                problems.append(
                    f"metrics.json: {len(odd)} top-level "
                    f"{'entry is' if len(odd) == 1 else 'entries are'} not "
                    f"{'a' if len(odd) == 1 else ''} per-model "
                    f"{'object' if len(odd) == 1 else 'objects'} "
                    f"({', '.join(odd[:4])}{', ...' if len(odd) > 4 else ''}) - "
                    f"it must be {{model_name: evaluate(...)}}, the dict "
                    f"`evaluate_pair` returns, not a flattened report")
            # Only dissect the per-model dicts when the shape is right. When it is not,
            # every line below would be a restatement of the shape problem.
            if len(scored) == EXPECTED_MODELS and not odd:
                for name, m in scored.items():
                    missing = [k for k in METRIC_REQUIRED if k not in m]
                    if missing:
                        problems.append(
                            f"metrics.json[{name}]: missing {', '.join(missing)} - these "
                            f"come from `evaluate()`")
                    elif (m.get("_evidence") or {}).get("producer") != "evaluate":
                        problems.append(
                            f"metrics.json[{name}]: _evidence.producer is "
                            f"{(m.get('_evidence') or {}).get('producer')!r}, expected "
                            f"'evaluate' - the numbers were not produced by evaluate()")
    return problems


def require_evidence(md_path) -> None:
    """Refuse to render a report the workflow did not actually produce.

    The evidence lives in `out/evidence/` - `driver.py` writes it there, and `report.md`
    sits one level above it in `out/`. A run that skipped the workflow and hand-rolled its
    own still ends in a `.md`, so without this check the PDF renders anyway and the bypass
    leaves no trace.

    Checking that the files *exist* turned out not to be enough: the run this was written
    for wrote its own `eda.json` by hand, with eight of the fourteen keys `profile()`
    returns and exactly the three cleaning keys missing. So each file is parsed and
    checked against the contract the report needs, and against the producer that is
    supposed to have written it.

    This is still a tripwire, not a lock. `producer` is self-declared, so a run that sets
    out to forge it can - but the observed failure is imitation rather than forgery, and
    an imitated file is what this catches. Set SKILL_ALLOW_MISSING_EVIDENCE=1 to render
    anyway.
    """
    if os.environ.get("SKILL_ALLOW_MISSING_EVIDENCE") == "1":
        return
    md_path = Path(md_path)
    out_dir = md_path.parent / EVIDENCE_SUBDIR
    problems = _evidence_problems(out_dir)
    if problems:
        raise SystemExit(
            f"refusing to render: the evidence in {out_dir} does not match what the "
            f"workflow produces\n"
            + "".join(f"  - {p}\n" for p in problems)
            + f"  these files are written by `driver.py run` before {md_path.name};\n"
            + "  evidence that is missing or hand-written usually means the run skipped\n"
            + "  the driver and reimplemented the workflow - see 'Step 0' in SKILL.md\n"
            + "  (set SKILL_ALLOW_MISSING_EVIDENCE=1 to render anyway)"
        )


# ------------------------------------------------------------------- parsing


def strip_inline(text: str) -> str:
    text = COMMENT_RE.sub("", text)
    text = BOLD_RE.sub(r"\1", text)
    text = CODE_RE.sub(r"\1", text)
    return text.strip()


def strip_markers_only(text: str) -> str:
    """Drop HTML comments and trim, but keep `**` and backticks for the renderer.

    Paragraphs and bullets go through this; headings and table cells keep using
    `strip_inline`, because a table is aligned by character count and a bold face would
    break its columns, and a heading is bold already.
    """
    return COMMENT_RE.sub("", text).strip()


def split_inline(text: str) -> tuple[str, list[tuple[int, int, str]]]:
    """Return `(plain, spans)` - the marker-free text, and where emphasis applies in it.

    `plain` comes out identical to `strip_inline`, which is the point: line breaking runs
    on it, so adding or removing `**` cannot move a line break and the two-page budget
    stays put.

    Positions are recorded as the text is walked rather than searched for afterwards.
    A search would be wrong on the ordinary case of the same word appearing twice -
    `"**yes** or yes or **yes**"` - where the second span would land on the middle, unstyled
    occurrence. Markers nested inside a code span are scanned recursively, so
    `` `a **b** c` `` styles the `b` and still yields the same plain text as `strip_inline`.
    """
    out: list[str] = []
    spans: list[tuple[int, int, str]] = []
    pos = 0
    last = 0
    text = COMMENT_RE.sub("", text).strip()
    for m in INLINE_RE.finditer(text):
        head = text[last:m.start()]
        out.append(head)
        pos += len(head)
        content = m.group(1) if m.group(1) is not None else m.group(2)
        style = "bold" if m.group(1) is not None else "code"
        inner, inner_spans = split_inline(content)
        out.append(inner)
        spans.append((pos, pos + len(inner), style))
        spans.extend((pos + s, pos + e, st) for s, e, st in inner_spans)
        pos += len(inner)
        last = m.end()
    out.append(text[last:])
    return "".join(out), spans


def style_runs(line: str, start: int, spans) -> list[tuple[str, bool, bool]]:
    """Split one wrapped line into `(text, bold, code)` runs.

    `start` is where the line begins in the plain text, so the source spans can be clipped
    to it. Grouping is by run rather than by character: the caller draws one string per
    run, and a run per character would be both slow and visibly uneven.
    """
    end = start + len(line)
    flags = [0] * len(line)                 # bit 1 = bold, bit 2 = code
    for s, e, style in spans:
        bit = 1 if style == "bold" else 2
        for k in range(max(s, start), min(e, end)):
            flags[k - start] |= bit

    runs: list[tuple[str, bool, bool]] = []
    cur, cur_flag = "", (flags[0] if flags else 0)
    for ch, f in zip(line, flags):
        if f != cur_flag:
            runs.append((cur, bool(cur_flag & 1), bool(cur_flag & 2)))
            cur, cur_flag = "", f
        cur += ch
    if cur or not runs:
        runs.append((cur, bool(cur_flag & 1), bool(cur_flag & 2)))
    return runs


def parse(md_text: str) -> list[tuple[str, object]]:
    """Markdown subset -> list of (kind, payload) blocks."""
    lines = COMMENT_RE.sub("", md_text).splitlines()
    blocks: list[tuple[str, object]] = []
    paragraph: list[str] = []
    table: list[list[str]] = []
    skip_cover = False

    def flush_paragraph():
        if paragraph:
            blocks.append(("para", strip_markers_only(" ".join(paragraph))))
            paragraph.clear()

    def flush_table():
        if table:
            blocks.append(("table", [row[:] for row in table]))
            table.clear()

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()

        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue                      # separator row
            flush_paragraph()
            table.append(cells)
            continue
        flush_table()

        if not stripped:
            flush_paragraph()
            continue

        if stripped in ("---", "***", "___"):
            flush_paragraph()
            blocks.append(("space", None))
            continue

        # An image, optionally with a label in front of it: `Confusion matrices: ![alt](path)`
        # is how `report.py` writes a captioned figure. The label may not contain `!`, which
        # keeps the match anchored on the first figure on the line. Matching the line in full
        # is what decides whether this is a figure at all; anything else containing the syntax
        # - prose that mentions it, two figures on one line - the guard below flags outright
        # rather than printing. See the guard for why the coarseness is the point.
        image = re.fullmatch(r"(?:([^!\n]*?):\s*)?!\[[^\]]*\]\(([^)]+)\)", stripped)
        if image:
            flush_paragraph()
            if image.group(1):
                blocks.append(("para", image.group(1).strip() + ":"))
            blocks.append(("image", image.group(2)))
            continue

        heading = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            text = strip_inline(heading.group(2))
            if level <= 2 and re.search(r"cover", text, re.I):
                skip_cover = True
                continue
            if skip_cover and level <= 2:
                skip_cover = False
            if skip_cover:
                continue
            blocks.append((f"h{level}", text))
            continue

        if skip_cover:
            continue

        bullet = re.match(r"^[-*+]\s+(.*)$", stripped)
        if bullet:
            flush_paragraph()
            blocks.append(("bullet", strip_markers_only(bullet.group(1))))
            continue

        numbered = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if numbered:
            flush_paragraph()
            blocks.append(("bullet", strip_markers_only(numbered.group(1))))
            continue

        if re.search(r"!\[[^\]]*\]\(", stripped):
            # Only image syntax the pattern above could not place in full reaches here - text
            # on the same line as the figure, say. Appending it as prose is what let the
            # report print `Confusion matrices: ![...](figures/confusion_pair.png)` as a
            # paragraph with no figure on the page and nothing raised. `image()` already uses
            # this bracketed shape for a figure it cannot read; this is the parser's half of
            # the same rule, and it keeps the failure visible on the page.
            flush_paragraph()
            blocks.append(("para", f"[unplaceable figure: {stripped}]"))
            continue

        paragraph.append(stripped)

    flush_paragraph()
    flush_table()
    return blocks


def format_table(rows: list[list[str]]) -> list[str]:
    """Render a pipe table as aligned monospace lines, wrapping long cells."""
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    budget = max(10, (MAX_COLS - 3 * (ncol - 1)) // ncol)
    widths = [min(max(len(r[i]) for r in rows), budget) for i in range(ncol)]

    out = []
    for idx, row in enumerate(rows):
        wrapped = [textwrap.wrap(strip_inline(c) or " ", widths[i]) or [" "]
                   for i, c in enumerate(row)]
        for k in range(max(len(w) for w in wrapped)):
            cells = [(w[k] if k < len(w) else "").ljust(widths[i])
                     for i, w in enumerate(wrapped)]
            out.append("  ".join(cells).rstrip())
        if idx == 0:
            out.append("-" * min(MAX_COLS, sum(widths) + 3 * (ncol - 1)))
    return out


# ----------------------------------------------------------------- rendering


class Renderer:
    def __init__(self, pdf: PdfPages, base_dir: Path | None = None, variant: str = ""):
        self.pdf = pdf
        self.base_dir = base_dir or Path.cwd()
        self.variant = variant
        self.page = None
        self.y = TOP
        self.lines = 0
        self.body_pages = 0

    # -- page handling ----------------------------------------------------
    def new_page(self, body: bool = True):
        if self.page is not None:
            self.pdf.savefig(self.page)
            plt.close(self.page)
        self.page = plt.figure(figsize=(PAGE_W, PAGE_H))
        footer = "IN6227-Assignment-1"
        if self.variant:
            footer += f" · {self.variant}"
        self.page.text(LEFT, 0.035, footer, fontsize=6.5, color="#888888")
        self.y = TOP
        if body:
            self.body_pages += 1

    def identity_header(self, fields: list[tuple[str, str]]):
        """Identity block at the top of page 1 - the assignment asks for the matric
        number, name, 'IN6227-Assignment-1' and the variant on the first page."""
        self.page.text(LEFT, self.y, "IN6227 Data Mining - Assignment 1",
                       fontsize=13, weight="bold", va="top", ha="left")
        if self.variant:
            self.page.text(RIGHT, self.y, self.variant, fontsize=11, color="#4c78a8",
                           va="top", ha="right")
        self.y -= 15.0 / (PAGE_H * 72)

        self.page.text(LEFT, self.y, "IN6227-Assignment-1",
                       fontsize=8, color="#777777", va="top", ha="left")
        self.y -= 11.0 / (PAGE_H * 72)

        # Two fields per line keeps the block inside a compact band.
        row: list[str] = []
        for key, value in fields:
            if value:
                row.append(f"{key}: {value}")
            if len(row) == 2:
                self._text("     ".join(row), BODY_PT - 0.5, color="#333333")
                row = []
        if row:
            self._text("     ".join(row), BODY_PT - 0.5, color="#333333")

        self.y -= 3.0 / (PAGE_H * 72)
        self.page.add_artist(Line2D([LEFT, RIGHT], [self.y, self.y],
                                    color="#bbbbbb", lw=0.8,
                                    transform=self.page.transFigure))
        self.y -= 5.0 / (PAGE_H * 72)

    def ensure(self, needed: float):
        if self.y - needed < BOTTOM:
            self.new_page()

    # -- text -------------------------------------------------------------
    def _advance(self, s: str, size: float, mono: bool, bold: bool = False) -> float:
        """Width of `s` as a fraction of the page width, measured rather than estimated.

        A measured advance is what lets bold and code sit inline without overlapping or
        drifting. The font property must match what is drawn - **weight included**: the
        bold face is wider than the regular one, so measuring a bold run with the regular
        font under-reports it by roughly the width of the following space, and the next run
        is drawn early enough to swallow that space. Falls back to an estimate if the
        backend cannot measure, which keeps a figure out of the text path.
        """
        prop = FontProperties(family="monospace" if mono else "sans-serif", size=size,
                              weight="bold" if bold else "normal")
        try:
            w_px, _, _ = self.page.canvas.get_renderer().get_text_width_height_descent(
                s, prop, False)
            return (w_px / self.page.dpi) / PAGE_W
        except Exception:
            scale = 0.60 if mono else 0.52
            return (len(s) * size * scale * (1.04 if bold else 1.0)) / (PAGE_W * 72)

    def _runs(self, runs, size: float, *, indent: float = 0.0, color: str = "#111111",
              mono_line: bool = False):
        """Draw one line as consecutive runs, each placed after the measured last one."""
        x = LEFT + indent
        for text, bold, code in runs:
            if not text:
                continue
            draw_size = size if mono_line else (size * CODE_SCALE if code else size)
            mono = mono_line or code
            self.page.text(x, self.y, text, fontsize=draw_size, va="top", ha="left",
                           family="monospace" if mono else "sans-serif", color=color,
                           weight="bold" if bold else "normal",
                           transform=self.page.transFigure)
            x += self._advance(text, draw_size, mono, bold)
        self.y -= (size * LINE_FACTOR) / (PAGE_H * 72)
        self.lines += 1

    def _text(self, s: str, size: float, *, indent: float = 0.0, mono: bool = False,
              color: str = "#111111", weight: str = "normal"):
        self._runs([(s, weight == "bold", False)], size, indent=indent, color=color,
                   mono_line=mono)

    def _wrap(self, text: str, size: float, indent: float = 0.0) -> list[str]:
        if not text:
            return [""]
        pt_per_char = size * (0.60 if size == MONO_PT else 0.52)
        avail_pt = (RIGHT - LEFT - indent) * PAGE_W * 72
        width = max(20, int(avail_pt / pt_per_char))
        return textwrap.wrap(text, width) or [""]

    def _flow(self, text: str, size: float, *, indent: float = 0.0, marker: str = ""):
        """Wrap the plain text, then find each wrapped line in it to recover its emphasis.

        The wrap runs on the marker-free text exactly as it did before emphasis was
        rendered, so line breaks are unchanged. The forward scan is what maps the emphasis
        back on; a line that cannot be found - which would take unusual whitespace in the
        source - is drawn plain instead of raising.
        """
        plain, spans = split_inline(text)
        cursor = 0
        for i, line in enumerate(self._wrap(plain, size, indent=indent)):
            self.ensure(size * LINE_FACTOR / (PAGE_H * 72))
            at = plain.find(line, cursor)
            if at < 0:
                runs = [(line, False, False)]
            else:
                cursor = at + len(line)
                runs = style_runs(line, at, spans)
            if marker:
                runs = [((marker if i == 0 else " " * len(marker)), False, False)] + runs
            self._runs(runs, size, indent=indent)

    def heading(self, level: int, text: str):
        size = {1: H1, 2: H2, 3: H3}[level]
        self.ensure((size + SPACE_PT[size]) / (PAGE_H * 72) * 2)
        self.y -= SPACE_PT[size] / (PAGE_H * 72)
        self._text(text, size, weight="bold")

    def paragraph(self, text: str):
        self._flow(text, BODY_PT)

    def bullet(self, text: str):
        self._flow(text, BODY_PT, indent=0.018, marker="• ")

    def table(self, rows: list[list[str]]):
        for i, line in enumerate(format_table(rows)):
            self.ensure(MONO_PT * LINE_FACTOR / (PAGE_H * 72))
            self._text(line, MONO_PT, mono=True,
                       color="#333333" if i == 1 else "#111111")

    def image(self, path: str, max_h: float = 0.20):
        # Figures are written relative to the markdown file, not to the shell's cwd.
        p = Path(path)
        if not p.is_absolute() and not p.exists():
            candidate = self.base_dir / path
            if candidate.exists():
                p = candidate
        if not p.exists():
            self.paragraph(f"[missing figure: {path}]")
            return
        try:
            arr = imread(p)
        except Exception:
            self.paragraph(f"[unreadable figure: {path}]")
            return
        aspect = arr.shape[1] / arr.shape[0]
        avail_w = (RIGHT - LEFT) * PAGE_W
        h_in = min(max_h * PAGE_H, avail_w / aspect)
        w_in = h_in * aspect
        h_frac = h_in / PAGE_H
        self.ensure(h_frac + 0.01)
        self.y -= h_frac
        ax = self.page.add_axes([
            LEFT + ((RIGHT - LEFT) - w_in / PAGE_W) / 2,
            self.y, w_in / PAGE_W, h_frac,
        ])
        ax.imshow(arr)
        ax.axis("off")
        self.y -= 0.008

    def space(self):
        self.y -= 6.0 / (PAGE_H * 72)

    def render(self, blocks):
        for kind, payload in blocks:
            if kind == "h1":
                self.heading(1, payload)
            elif kind == "h2":
                self.heading(2, payload)
            elif kind == "h3":
                self.heading(3, payload)
            elif kind == "para":
                self.paragraph(payload)
            elif kind == "bullet":
                self.bullet(payload)
            elif kind == "table":
                self.table(payload)
            elif kind == "image":
                self.image(payload)
            elif kind == "space":
                self.space()
        self.pdf.savefig(self.page)
        plt.close(self.page)
        self.page = None


def _pages_warning(total: int, limit: int) -> int:
    print(f"WARNING: report is {total} pages, limit is {limit}. "
          f"Trim discussion prose before shrinking any font.", file=sys.stderr)
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md", required=True, help="filled report markdown")
    ap.add_argument("--out", required=True, help="output PDF path")
    ap.add_argument("--student-id", default="")
    ap.add_argument("--student-name", default="")
    ap.add_argument("--variant", default="",
                    help='e.g. "Variant-2"; drawn in the page-1 header and the footer')
    ap.add_argument("--model-version", default="")
    ap.add_argument("--harness", default="")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--max-pages", type=int, default=2,
                    help="total page limit; the report body shares page 1 with the header")
    args = ap.parse_args(argv)

    md_path = Path(args.md)
    if not md_path.exists():
        raise SystemExit(f"markdown not found: {md_path}")

    blocks = parse(md_path.read_text(encoding="utf-8"))
    if not blocks:
        raise SystemExit("nothing to render: the markdown produced no blocks")

    out_path = Path(args.out)
    refuse_bad_output_dir(out_path)
    require_evidence(md_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        r = Renderer(pdf, base_dir=md_path.parent, variant=args.variant)
        r.new_page()
        r.identity_header([
            ("Matric number", args.student_id),
            ("Full name", args.student_name),
            ("Model name & version", args.model_version),
            ("LLM interface", args.harness),
            ("Dataset", args.dataset),
        ])
        r.render(blocks)

    print(f"wrote {out_path}  ({r.body_pages} page(s), header + body)")
    if r.body_pages > args.max_pages:
        return _pages_warning(r.body_pages, args.max_pages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
