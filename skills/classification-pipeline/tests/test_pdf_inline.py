#!/usr/bin/env python
"""Checks for the inline-emphasis renderer.

    python tests/test_pdf_inline.py

The property that matters is not "does bold look bold" - it is that **adding or removing
emphasis cannot move a line break**, because that is what keeps a two-page report from
becoming a three-page one. So most of this file compares the plain text against what the
old marker-stripping produced.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from make_pdf import (BOLD_RE, CODE_RE, COMMENT_RE, format_table, parse, split_inline,
                      style_runs)


def old_strip(text: str) -> str:
    """The stripping the renderer used before emphasis was rendered, verbatim."""
    text = COMMENT_RE.sub("", text)
    text = BOLD_RE.sub(r"\1", text)
    text = CODE_RE.sub(r"\1", text)
    return text.strip()


def test_plain_text_is_identical_to_the_old_stripping():
    for s in [
        "plain sentence with no markers at all",
        "a **bold** word and a `code` span",
        "**leading bold** then text",
        "text then **trailing bold**",
        "nested `a **b** c` inside code",
        "back-to-back **one****two**",
        "a lone * star and a lone ` backtick",
        "",
    ]:
        assert split_inline(s)[0] == old_strip(s), s


def test_spans_point_at_the_bold_word():
    plain, spans = split_inline("the **rare** class and `lift`")
    assert plain == "the rare class and lift"
    assert (plain[spans[0][0]:spans[0][1]], spans[0][2]) == ("rare", "bold")
    assert (plain[spans[1][0]:spans[1][1]], spans[1][2]) == ("lift", "code")


def test_a_word_repeated_within_a_line_keeps_its_own_span():
    """Searching for the bold text afterwards would bold the middle, unstyled 'yes'."""
    plain, spans = split_inline("**yes** or yes or **yes**")
    assert plain == "yes or yes or yes"
    assert [(s, e) for s, e, _ in spans] == [(0, 3), (14, 17)], (
        "the spans must be the first and last words, not the first two matches"
    )
    assert [plain[s:e] for s, e, _ in spans] == ["yes", "yes"]


def test_markers_nested_inside_a_code_span_are_scanned_too():
    plain, spans = split_inline("nested `a **b** c` here")
    assert plain == "nested a b c here"
    inner = [(plain[s:e], st) for s, e, st in spans]
    assert ("a b c", "code") in inner and ("b", "bold") in inner, inner


def test_style_runs_groups_consecutive_characters():
    plain, spans = split_inline("the **rare** class")
    line = plain
    runs = style_runs(line, 0, spans)
    assert runs == [("the ", False, False), ("rare", True, False), (" class", False, False)]


def test_style_runs_clips_a_span_to_the_line():
    """The line is a window onto the plain text, so the span must be clipped to it."""
    plain, spans = split_inline("aaa **bbbb** ccc")
    at = plain.find("bb")
    runs = style_runs("bb", at, spans)
    assert runs == [("bb", True, False)]


def test_style_runs_combines_bold_and_code():
    runs = style_runs("xy", 0, [(0, 2, "bold"), (0, 2, "code")])
    assert runs == [("xy", True, True)]


def test_parse_keeps_markers_for_paragraphs_and_bullets_only():
    md = ("# A **heading**\n\n"
          "A **bold** paragraph.\n\n"
          "- a **bold** bullet\n\n"
          "| a | b |\n|---|---|\n| **x** | y |\n")
    kinds = {k: v for k, v in parse(md)}
    assert kinds["h1"] == "A heading", "headings are already bold; markers are stripped"
    assert kinds["para"] == "A **bold** paragraph."
    assert kinds["bullet"] == "a **bold** bullet"
    # Table cells keep their markers through parse, and lose them at format_table - because
    # the columns are aligned by character count and a bold face would break them.
    assert kinds["table"][1] == ["**x**", "y"]
    assert "**" not in "".join(format_table(kinds["table"]))


def test_a_labelled_image_line_becomes_a_caption_and_a_figure():
    """The exact shape `report.py` writes for the confusion matrices.

    The parser matched the bare image syntax with `fullmatch`, so the label in front of it
    made the whole line fail to match and it fell through to the paragraph branch: every
    report this skill produced printed `Confusion matrices: ![confusion matrices, one panel
    per model](figures/confusion_pair.png)` as text, with no figure on the page and no test
    failing - because no test in this file parsed an image at all.
    """
    blocks = parse("Confusion matrices: ![both panels](figures/confusion_pair.png)")
    assert blocks == [("para", "Confusion matrices:"),
                      ("image", "figures/confusion_pair.png")], blocks


def test_a_bare_image_line_still_parses():
    """The form the subset documented first, which must not regress to a paragraph."""
    assert parse("![both panels](figures/confusion_pair.png)") == [
        ("image", "figures/confusion_pair.png")]


def test_image_syntax_the_parser_cannot_place_is_marked_on_the_page():
    """A paragraph is indistinguishable from a report that simply has no figure.

    So the one construct that cannot be placed is the one construct that must not be
    printed as prose - it has to look wrong on the page.
    """
    for line in ("see ![both panels](figures/confusion_pair.png) above",
                 "A: ![one](a.png) and ![two](b.png)"):
        assert parse(line) == [("para", f"[unplaceable figure: {line}]")], parse(line)


def test_prose_that_merely_mentions_image_syntax_is_also_flagged():
    """The guard keys on presence, not on shape, and that coarseness is deliberate.

    A paragraph discussing `![alt](path)` is not a figure, so this is a false positive. It
    is worth paying: the report has no legitimate prose that mentions the syntax, and the
    two ways to be wrong are not symmetric. A false positive is a visible marker on the
    page; the failure it replaces - a figure quietly rendering as its own source text - is
    invisible, and it survived several runs and a green test suite.
    """
    line = "Figures are written `![alt](path)`, relative to the markdown file."
    assert parse(line) == [("para", f"[unplaceable figure: {line}]")], parse(line)


def test_pagination_cannot_move_when_emphasis_changes():
    """The whole point: the wrap runs on the plain text, so markers do not count."""
    from make_pdf import Renderer

    class _Fake(Renderer):
        def __init__(self):
            self.page = None

    r = _Fake()
    sentence = ("This paragraph is deliberately long enough to wrap over several lines so "
                "that the break positions are actually exercised by the test rather than "
                "happening to fall in one place.")
    plain_a, _ = split_inline(sentence)
    plain_b, _ = split_inline(sentence.replace("deliberately ", "**deliberately** "))
    assert r._wrap(plain_a, 8.5) == r._wrap(plain_b, 8.5)


def test_no_marker_survives_into_a_drawn_line():
    """Phase D's gate: whatever the source markdown uses, no `**` or backtick is drawn.

    This walks the real report through the real block renderer with the page turn disabled
    and the drawing replaced by a recorder, so it checks the strings the renderer would
    actually emit - not the ones the parser happens to hold.
    """
    from make_pdf import BODY_PT, MONO_PT, Renderer

    md = ROOT / "out" / "report.md"
    if not md.exists():
        print(f"      (skipped: {md} not present)")
        return

    drawn: list[str] = []

    class Recorder(Renderer):
        def __init__(self):
            self.page = None
            self.y = 0.9
            self.lines = 0
            self.body_pages = 0
            self.base_dir = ROOT

        def ensure(self, needed):        # never turn the page; there is no canvas
            pass

        def _runs(self, runs, size, *, indent=0.0, color="#111111", mono_line=False):
            drawn.extend(t for t, _, _ in runs)
            self.y -= 0.001

    r = Recorder()
    for kind, payload in parse(md.read_text(encoding="utf-8")):
        if kind in ("h1", "h2", "h3"):
            r.heading(int(kind[1]), payload)
        elif kind == "para":
            r.paragraph(payload)
        elif kind == "bullet":
            r.bullet(payload)
        elif kind == "table":
            for i, line in enumerate(format_table(payload)):
                r._text(line, MONO_PT, mono=True)

    assert drawn, "the report produced no drawable text"
    bad = [s for s in drawn if "**" in s or "`" in s]
    assert not bad, f"markers reached the drawn lines: {bad[:5]}"
    assert any(s.strip() for s in drawn)
    _ = BODY_PT


def main() -> int:
    import traceback

    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception:
            failed.append(name)
            print(f"FAIL  {name}")
            traceback.print_exc()
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
