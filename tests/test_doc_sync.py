"""Tests alignment of documentation Python snippets with their example scripts.

Tests that documentation Python snippets appear (verbatim or near-verbatim)
in their corresponding example scripts.

The explicit ``DOC_TO_SCRIPT`` mapping connects each documentation Markdown file
to its runnable example script in ``examples/doc/``.

For each pair, every Python code block in the doc is compared against the
example script as a *snippet-to-example* check:

1. Blocks that are reference documentation, pseudo-code, or illustrative
   implementations are **skipped** (see :func:`_is_stub_block` and
   :func:`_is_reference_only_block`).

2. *Key lines* are extracted from each remaining block: non-trivial code
   statements only — blank lines, comments, imports, lines containing ``...``,
   placeholder assignments (``var = ...``), and ``print`` statements are all
   filtered out.

3. For each key line the best-matching line anywhere in the example script is
   located via ``difflib.SequenceMatcher``.  A line is *matched* when the best
   ratio ≥ ``_LINE_THRESHOLD`` (0.75).  This threshold corresponds to
   "near-verbatim": the line is essentially identical, differing at most in
   variable-name suffixes or trivial literal changes.

4. The snippet's **match score** = matched lines / total key lines.

Deviation levels (per snippet):

* Score ≥ ``_ALIGNED_THRESHOLD`` (0.85) — snippet is aligned; no action.
* ``_MINOR_THRESHOLD`` (0.60) ≤ score < ``_ALIGNED_THRESHOLD`` — minor
  deviation; a ``UserWarning`` is emitted listing the unmatched lines together
  with their best candidate in the example.
* Score < ``_MINOR_THRESHOLD`` — major deviation; the test **fails** with a
  report of every problematic snippet and its unmatched lines.

Reference-only blocks
---------------------
Some code blocks in design documents define classes or call constructors that
are intentionally absent from the example script (e.g. the Bologna worked-
example implementations in ``dd-cdt-modularity.md``, or anti-pattern
illustrations).  Such blocks are silently **skipped** rather than reported as
failures: see :func:`_is_reference_only_block`.

Standalone runner
-----------------
Run this file directly to inspect alignment without going through pytest.

**Compact summary table** (default — no arguments)::

    uv run python tests/test_doc_sync.py
    uv run python tests/test_doc_sync.py --summary

Prints one row per doc/script pair showing counts of aligned (✓ OK),
minor-warning (~ Warn), failing (✗ Fail), and skipped blocks.

**Verbose block-by-block report** for a single pair (filter by name fragment)::

    uv run python tests/test_doc_sync.py getting-started
    uv run python tests/test_doc_sync.py dd-cdt-engine

Each aligned key line is shown as ``=`` (exact) or ``~`` (near-verbatim with
the best match in the script), and each unmatched line as ``✗``.

**Verbose report for all pairs**::

    uv run python tests/test_doc_sync.py --all
    uv run python tests/test_doc_sync.py --verbose
"""

# SPDX-License-Identifier: Apache-2.0

import difflib
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Repository root  (this file lives at  tests/test_doc_sync.py)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Explicit mapping: documentation file → reference example script
# ---------------------------------------------------------------------------

#: Maps each documentation Markdown path (relative to ``_ROOT``) to the
#: corresponding runnable example script (also relative to ``_ROOT``).
#: The mapping is **explicit**: adding a new doc/example pair is a one-line
#: change here.
DOC_TO_SCRIPT: dict[str, str] = {
    "README.md": "examples/doc/doc_readme.py",
    "docs/getting-started.md": "examples/doc/doc_getting_started.py",
    "docs/design/dd-cdt-engine.md": "examples/doc/doc_engine.py",
    "docs/design/dd-cdt-model.md": "examples/doc/doc_model.py",
    "docs/design/dd-cdt-modularity.md": "examples/doc/doc_modularity.py",
    "examples/overtourism_molveno/overtourism-getting-started.md": ("examples/doc/doc_overtourism_getting_started.py"),
}

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

#: Minimum SequenceMatcher ratio for a doc line to be considered *matched*.
#: 0.75 ≈ "near-verbatim": identical or differing only in variable-name
#: suffixes / minor literal changes.
_LINE_THRESHOLD: float = 0.75

#: Per-snippet score above which the snippet is treated as aligned (no action).
_ALIGNED_THRESHOLD: float = 0.85

#: Per-snippet score below which the deviation is *major* (test fails).
#: Scores in [_MINOR_THRESHOLD, _ALIGNED_THRESHOLD) are *minor* (warning).
_MINOR_THRESHOLD: float = 0.60

# ---------------------------------------------------------------------------
# Compiled regular expressions
# ---------------------------------------------------------------------------

# Python fences (```python … ``` or ```Python … ```)
_PY_FENCE_RE = re.compile(r"```[Pp]ython\n(.*?)```", re.DOTALL)

# Inline stub method body:  ): …  or  -> ReturnType: …
_STUB_METHOD_RE = re.compile(r"(\)|(?:->\s+[\w\[\], |.]+))\s*:\s*\.\.\.\s*$")

# Illustrative-only section header:  # === … ===
_ILLUSTRATIVE_HEADER_RE = re.compile(r"^\s*#\s*===")

# Any import line
_IMPORT_LINE_RE = re.compile(r"^\s*(from\s+\w|import\s+\w)")

# Placeholder assignment:  identifier = …
_PLACEHOLDER_RE = re.compile(r"^\s*[\w.]+\s*=\s*\.\.\.\s*$")

# print(…) statement
_PRINT_STMT_RE = re.compile(r"^\s*print\s*\(")

# Class definition (captures name):  class Foo
_CLASS_DEF_RE = re.compile(r"^\s*class\s+([A-Z]\w*)")

# Class attribute access anywhere in block text:  UpperCaseName.something
# Used to detect reference-only blocks whose class name collides with a local
# example class (e.g. Bologna TrafficModel vs. local TrafficModel) or whose
# first token is not a class definition (e.g. a standalone __init__ method
# referencing InflowModel.Inputs).
_CLASS_ATTR_ACCESS_RE = re.compile(r"\b([A-Z]\w*)\.")

# Constructor call anywhere in block text:  UpperCaseName(
# Used to detect reference-only blocks that instantiate classes absent from
# the example (e.g. the TL;DR wiring block calling InflowModel(...) and
# EmissionsModel(...) which are intentionally not in the script).
_CLASS_CONSTRUCTOR_RE = re.compile(r"\b([A-Z]\w*)\s*\(")

# ---------------------------------------------------------------------------
# Python block extraction
# ---------------------------------------------------------------------------


def _extract_python_blocks(md_text: str) -> list[str]:
    """Return the raw content of every Python code fence in a Markdown file.

    Parameters
    ----------
    md_text : str
        Full text of a Markdown document.

    Returns
    -------
    list[str]
        One entry per ``python`` (or ``Python``) fenced code block, in document
        order.
    """
    return _PY_FENCE_RE.findall(md_text)


# ---------------------------------------------------------------------------
# Stub / pseudo-code detection
# ---------------------------------------------------------------------------


def _is_stub_block(code: str) -> bool:
    """Return ``True`` when a code block is reference documentation or pseudo-code.

    Parameters
    ----------
    code : str
        Raw content of a Python code fence.

    Returns
    -------
    bool
        ``True`` when the block should be skipped for alignment checking.

    Notes
    -----
    Three heuristics are applied:

    1. **Inline stub method body** — any line ending with ``):<ws>...`` or
       ``-> ReturnType:<ws>...`` (method body shown as ``...``) indicates an
       API-reference block.
    2. **Heavy pseudo-code** — more than two ``...`` tokens anywhere in the block
       signal intentionally incomplete illustrative code.
    3. **Illustrative-only header** — a line starting with ``# ===`` marks blocks
       that show a simplified internal implementation for educational purposes.
    """
    for line in code.splitlines():
        stripped = line.strip()
        if _STUB_METHOD_RE.search(stripped):
            return True
        if _ILLUSTRATIVE_HEADER_RE.match(line):
            return True
    if code.count("...") > 2:
        return True
    return False


# ---------------------------------------------------------------------------
# Reference-only block detection
# ---------------------------------------------------------------------------


def _is_reference_only_block(code: str, example_text: str) -> bool:
    """Return ``True`` when a block is a reference or worked-example block.

    Parameters
    ----------
    code : str
        Raw content of a Python code fence.
    example_text : str
        Full text of the corresponding example script.

    Returns
    -------
    bool
        ``True`` when the block should be skipped because it is a reference or
        worked-example block not expected to have a verbatim counterpart in the
        example script.

    Notes
    -----
    Three heuristics are applied in order:

    1. **Top-level class definition** — if the block opens with
       ``class Foo(…):`` and ``Foo`` is absent from the example, the block is
       reference-only.  This catches straightforward worked-example class
       definitions (``InflowModel``, ``EmissionsModel``, ``BolognaModel``, …).

    2. **Class attribute access** — scan the entire block text (including
       comments) for ``UpperCaseName.something`` patterns.  If *any* such
       class name is absent from the example the block is reference-only.
       This catches:

       * Method-only blocks (e.g. the Bologna ``InflowModel.__init__``) that
         reference ``InflowModel.Inputs`` etc. without opening with a class
         definition.
       * Blocks whose first class name coincidentally matches a local example
         class (e.g. the Bologna ``TrafficModel`` whose comment mentions
         ``InflowModel.outputs``) but whose content references absent classes.
       * Usage blocks such as ``m = BolognaModel(**BolognaModel.default_inputs())``.

    3. **Constructor call** — scan the entire block text for
       ``UpperCaseName(`` patterns.  If *any* such class name is absent from
       the example the block is reference-only.  This catches TL;DR wiring
       blocks that instantiate sub-models (e.g. ``InflowModel(...)``,
       ``EmissionsModel(...)``) that are intentionally absent from the script.
    """
    # Heuristic 1 — first top-level class definition absent from example.
    for line in code.splitlines():
        m = _CLASS_DEF_RE.match(line)
        if m:
            if m.group(1) not in example_text:
                return True
            # Only check the first top-level class definition.
            break

    # Heuristic 2 — any UpperCaseName. attribute access absent from example.
    for m in _CLASS_ATTR_ACCESS_RE.finditer(code):
        if m.group(1) not in example_text:
            return True

    # Heuristic 3 — any UpperCaseName( constructor call absent from example
    # (non-comment lines only, to avoid false positives from English phrases
    # such as "# Expected (mean) CO2" where "Expected" looks like a class name).
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for m in _CLASS_CONSTRUCTOR_RE.finditer(stripped):
            if m.group(1) not in example_text:
                return True

    return False


# ---------------------------------------------------------------------------
# Key-line extraction and normalisation
# ---------------------------------------------------------------------------


def _normalize_line(line: str) -> str:
    """Strip leading/trailing whitespace and collapse internal runs to one space.

    Parameters
    ----------
    line : str
        A single source-code line (may be indented).

    Returns
    -------
    str
        Normalised line suitable for fuzzy comparison.
    """
    return " ".join(line.split())


def _get_key_lines(block: str) -> list[str]:
    """Extract non-trivial lines from a code block for verbatim-copy checking.

    Parameters
    ----------
    block : str
        Raw content of a Python code fence.

    Returns
    -------
    list[str]
        Normalised code statements suitable for near-verbatim matching.

    Notes
    -----
    Skipped lines:

    * Blocks detected as stub / pseudo-code (see :func:`_is_stub_block`): the
      whole block is skipped and an empty list is returned.
    * Blank lines.
    * Comment-only lines (starting with ``#``).
    * Import lines (``import …`` / ``from … import …``).
    * Lines containing ``...`` anywhere — handles both standalone ``...`` and
      pseudo-code placeholders such as ``func(arg, ...)``.
    * Placeholder assignment patterns (``variable = ...``).
    * ``print(…)`` statements — display lines that legitimately differ between
      documentation and example scripts.
    """
    if _is_stub_block(block):
        return []

    result: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if _IMPORT_LINE_RE.match(stripped):
            continue
        if "..." in stripped:
            continue
        if _PLACEHOLDER_RE.match(stripped):
            continue
        if _PRINT_STMT_RE.match(stripped):
            continue
        normalised = _normalize_line(stripped)
        if normalised:
            result.append(normalised)
    return result


# ---------------------------------------------------------------------------
# Per-snippet result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SnippetResult:
    """Alignment result for a single documentation code block.

    Attributes
    ----------
    block_index : int
        Zero-based position of the block in the Markdown file.
    preview : str
        First non-trivial line of the block, truncated to 72 characters
        (used in human-readable messages).
    total_key_lines : int
        Total number of key lines extracted from the block.
    matched : list[tuple[str, str, float]]
        ``(doc_line, best_example_line, ratio)`` for each matched key line.
    missed : list[tuple[str, float]]
        ``(doc_line, best_ratio_anywhere)`` for each unmatched key line.
    """

    block_index: int
    preview: str
    total_key_lines: int
    matched: list[tuple[str, str, float]] = field(default_factory=list)
    missed: list[tuple[str, float]] = field(default_factory=list)

    @property
    def match_score(self) -> float:
        """Fraction of key lines with a near-verbatim match in the example."""
        if self.total_key_lines == 0:
            return 1.0
        return len(self.matched) / self.total_key_lines

    @property
    def is_empty(self) -> bool:
        """``True`` when the block has no key lines to check."""
        return self.total_key_lines == 0


# ---------------------------------------------------------------------------
# Per-snippet checking
# ---------------------------------------------------------------------------


def _snippet_preview(block: str, block_index: int) -> str:
    """Return a short human-readable label for a code block.

    Parameters
    ----------
    block : str
        Raw block content.
    block_index : int
        Index of the block in the document.

    Returns
    -------
    str
        First non-empty, non-comment line of the block, truncated to 72 chars.
    """
    for line in block.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:72] + ("…" if len(stripped) > 72 else "")
    return f"<block {block_index}>"


def _check_snippet(
    block_index: int,
    block: str,
    example_lines_norm: list[str],
) -> SnippetResult:
    """Compare one documentation code block against all normalised example lines.

    Parameters
    ----------
    block_index : int
        Index of this block in the Markdown file (used in the result label).
    block : str
        Raw content of the Python code fence.
    example_lines_norm : list[str]
        Every normalised non-blank line from the example script.

    Returns
    -------
    SnippetResult
        Per-snippet alignment result containing matched and unmatched lines.
    """
    preview = _snippet_preview(block, block_index)
    key_lines = _get_key_lines(block)

    result = SnippetResult(
        block_index=block_index,
        preview=preview,
        total_key_lines=len(key_lines),
    )

    for doc_line in key_lines:
        best_ratio = 0.0
        best_example_line = ""
        for ex_line in example_lines_norm:
            ratio = difflib.SequenceMatcher(None, doc_line, ex_line).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_example_line = ex_line
            if ratio >= 0.99:
                break  # exact match found; skip remaining lines
        if best_ratio >= _LINE_THRESHOLD:
            result.matched.append((doc_line, best_example_line, best_ratio))
        else:
            result.missed.append((doc_line, best_ratio))

    return result


# ---------------------------------------------------------------------------
# Human-readable report formatting
# ---------------------------------------------------------------------------


def _format_snippet_report(result: SnippetResult, doc_rel: str) -> str:
    """Format a single snippet's alignment issues as a human-readable string.

    Parameters
    ----------
    result : SnippetResult
        The alignment result to describe.
    doc_rel : str
        Relative path to the documentation file (for context).

    Returns
    -------
    str
        Multi-line text suitable for inclusion in a warning or failure message.
    """
    lines = [
        f"  Block {result.block_index} of {doc_rel}  "
        f"(score {result.match_score:.0%}: "
        f"{len(result.matched)} matched, {len(result.missed)} unmatched "
        f"out of {result.total_key_lines} key lines)",
        f"    First line: {result.preview!r}",
        "    Unmatched lines (doc line  →  best candidate in example):",
    ]
    for doc_line, best_ratio in sorted(result.missed, key=lambda t: t[1]):
        lines.append(f"      [{best_ratio:.2f}]  {doc_line!r}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compact summary  (used by __main__ with no filter arguments)
# ---------------------------------------------------------------------------


@dataclass
class PairStats:
    """Block-level counts for one doc/script pair.

    Attributes
    ----------
    doc_name : str
        Short name derived from the Markdown filename (no extension).
    total : int
        Total number of Python code blocks in the document.
    perfect : int
        Aligned blocks where every matched key line has ratio >= 0.99 (all ``=``).
    aligned : int
        Aligned blocks that have at least one near-verbatim (``~``) matched line.
    minor : int
        Blocks with ``_MINOR_THRESHOLD`` <= score < ``_ALIGNED_THRESHOLD``.
    fail : int
        Blocks with score < ``_MINOR_THRESHOLD``.
    stub_skip : int
        Blocks skipped because they are pseudo-code / API-reference stubs.
    ref_skip : int
        Blocks skipped because they are reference-only (worked examples, etc.).
    empty_skip : int
        Blocks with no key lines after filtering.
    """

    doc_name: str
    total: int = 0
    perfect: int = 0
    aligned: int = 0
    minor: int = 0
    fail: int = 0
    stub_skip: int = 0
    ref_skip: int = 0
    empty_skip: int = 0


def _collect_pair_stats(doc_rel: str, script_rel: str) -> PairStats:
    """Return block-level alignment counts for one doc/script pair.

    Parameters
    ----------
    doc_rel : str
        Documentation Markdown path relative to the repository root.
    script_rel : str
        Example script path relative to the repository root.

    Returns
    -------
    PairStats
        Populated stats object; all counts are zero when either file is missing.
    """
    doc_name = doc_rel.split("/")[-1].replace(".md", "")
    stats = PairStats(doc_name=doc_name)

    doc_path = _ROOT / doc_rel
    script_path = _ROOT / script_rel

    if not doc_path.exists() or not script_path.exists():
        return stats

    doc_text = doc_path.read_text(encoding="utf-8")
    script_text = script_path.read_text(encoding="utf-8")
    blocks = _extract_python_blocks(doc_text)
    example_lines_norm = [_normalize_line(ln) for ln in script_text.splitlines() if ln.strip()]

    for idx, block in enumerate(blocks):
        stats.total += 1
        if _is_stub_block(block):
            stats.stub_skip += 1
            continue
        if _is_reference_only_block(block, script_text):
            stats.ref_skip += 1
            continue

        result = _check_snippet(idx, block, example_lines_norm)

        if result.is_empty:
            stats.empty_skip += 1
            continue

        score = result.match_score
        if score >= _ALIGNED_THRESHOLD:
            if all(ratio >= 0.99 for _, _, ratio in result.matched):
                stats.perfect += 1
            else:
                stats.aligned += 1
        elif score >= _MINOR_THRESHOLD:
            stats.minor += 1
        else:
            stats.fail += 1

    return stats


def _print_summary_table() -> None:
    """Print a compact per-pair alignment summary table to stdout.

    Each row shows one doc/script pair with counts of perfect, aligned,
    minor-warning, failing, and skipped blocks.  A TOTAL row is appended at
    the bottom.  Pairs with no failures are prefixed ``✓``; pairs with
    failures ``✗``.

    The ``= OK`` column counts blocks where every matched key line is an exact
    or near-exact match (ratio >= 0.99).  The ``~ OK`` column counts blocks
    that are aligned overall but contain at least one near-verbatim (``~``)
    matched line.
    """
    _COL = 32  # width of the doc-name column (covers longest name + prefix)

    header = (
        f"{'Doc':<{_COL}}  {'Blks':>5}  {'= OK':>5}  {'~ OK':>5}  {'~ Warn':>6}  "
        f"{'✗ Fail':>6}  {'Stub':>5}  {'Ref':>4}  {'Empty':>5}"
    )
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    totals = PairStats(doc_name="TOTAL")

    for doc_rel, script_rel in DOC_TO_SCRIPT.items():
        s = _collect_pair_stats(doc_rel, script_rel)
        totals.total += s.total
        totals.perfect += s.perfect
        totals.aligned += s.aligned
        totals.minor += s.minor
        totals.fail += s.fail
        totals.stub_skip += s.stub_skip
        totals.ref_skip += s.ref_skip
        totals.empty_skip += s.empty_skip

        status = "✓" if s.fail == 0 else "✗"
        name_col = f"{status} {s.doc_name}"
        print(
            f"{name_col:<{_COL}}  {s.total:>5}  {s.perfect:>5}  {s.aligned:>5}  {s.minor:>6}  "
            f"{s.fail:>6}  {s.stub_skip:>5}  {s.ref_skip:>4}  {s.empty_skip:>5}"
        )

    print(sep)
    name_col = "  TOTAL"
    print(
        f"{name_col:<{_COL}}  {totals.total:>5}  {totals.perfect:>5}  {totals.aligned:>5}  {totals.minor:>6}  "
        f"{totals.fail:>6}  {totals.stub_skip:>5}  {totals.ref_skip:>4}  {totals.empty_skip:>5}"
    )
    print(sep)


# ---------------------------------------------------------------------------
# Parametrised pytest test
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Verbose alignment reporter  (used by __main__)
# ---------------------------------------------------------------------------


def _report_pair(doc_rel: str, script_rel: str) -> None:
    """Print a block-by-block alignment report for one doc/script pair.

    Parameters
    ----------
    doc_rel : str
        Documentation Markdown path relative to the repository root.
    script_rel : str
        Example script path relative to the repository root.
    """
    doc_path = _ROOT / doc_rel
    script_path = _ROOT / script_rel

    if not doc_path.exists():
        print(f"  [SKIP] doc not found: {doc_rel}")
        return
    if not script_path.exists():
        print(f"  [SKIP] script not found: {script_rel}")
        return

    doc_text = doc_path.read_text(encoding="utf-8")
    script_text = script_path.read_text(encoding="utf-8")
    blocks = _extract_python_blocks(doc_text)
    example_lines_norm = [_normalize_line(ln) for ln in script_text.splitlines() if ln.strip()]

    print(f"\n{'=' * 72}")
    print(f"  doc    : {doc_rel}")
    print(f"  script : {script_rel}")
    print(f"{'=' * 72}")

    for idx, block in enumerate(blocks):
        if _is_stub_block(block):
            print(f"\n  Block {idx:02d}  STUB-SKIP")
            continue
        if _is_reference_only_block(block, script_text):
            print(f"\n  Block {idx:02d}  REF-SKIP")
            continue

        result = _check_snippet(idx, block, example_lines_norm)

        if result.is_empty:
            print(f"\n  Block {idx:02d}  EMPTY-SKIP")
            continue

        score = result.match_score
        if score >= _ALIGNED_THRESHOLD:
            if all(ratio >= 0.99 for _, _, ratio in result.matched):
                tag = "PERFECT"
            else:
                tag = "ALIGNED"
        elif score >= _MINOR_THRESHOLD:
            tag = "minor"
        else:
            tag = "FAIL"

        print(
            f"\n  Block {idx:02d}  [{tag}]  score={score:.0%}"
            f"  ({len(result.matched)}/{result.total_key_lines} key lines)"
        )
        print(f"    First line: {result.preview!r}")

        for doc_line, ex_line, ratio in result.matched:
            doc_s = doc_line[:58] + "…" if len(doc_line) > 58 else doc_line
            if ratio > 0.99:
                print(f"    = {doc_s!r}")
            else:
                ex_s = ex_line[:58] + "…" if len(ex_line) > 58 else ex_line
                print(f"    ~ {doc_s!r}")
                print(f"        {ex_s!r}  [{ratio:.2f}]")

        for doc_line, best_ratio in result.missed:
            doc_s = doc_line[:58] + "…" if len(doc_line) > 58 else doc_line
            print(f"    ✗ [{best_ratio:.2f}] {doc_s!r}")


@pytest.mark.parametrize(
    "doc_rel,script_rel",
    list(DOC_TO_SCRIPT.items()),
    ids=[p.split("/")[-1].replace(".md", "") for p in DOC_TO_SCRIPT],
)
def test_doc_sync(doc_rel: str, script_rel: str) -> None:
    """Check that each Python code block in a doc appears near-verbatim in the example.

    Parameters
    ----------
    doc_rel : str
        Documentation Markdown path relative to the repository root.
    script_rel : str
        Example script path relative to the repository root.

    Notes
    -----
    For every Python code block in *doc_rel* that is not a stub, pseudo-code, or
    reference-only block (see :func:`_is_stub_block` and
    :func:`_is_reference_only_block`):

    * Each *key line* is matched against all lines of *script_rel* using
      ``difflib.SequenceMatcher`` with threshold ``_LINE_THRESHOLD`` (0.75).
    * The *snippet match score* = matched / total key lines.

    **Major deviation** (score < ``_MINOR_THRESHOLD`` = 0.60)
        The test **fails** and reports every problematic snippet with the
        unmatched lines and their best available similarity scores.

    **Minor deviation** (``_MINOR_THRESHOLD`` ≤ score < ``_ALIGNED_THRESHOLD`` = 0.85)
        A ``UserWarning`` is emitted listing the unmatched lines; the test passes.

    **Aligned** (score ≥ ``_ALIGNED_THRESHOLD``)
        No output; the test passes silently.

    The test is **skipped** (rather than failing with an error) when either
    file does not exist at the expected path.
    """
    doc_path = _ROOT / doc_rel
    script_path = _ROOT / script_rel

    if not doc_path.exists():
        pytest.skip(f"Documentation file not found: {doc_rel}")
    if not script_path.exists():
        pytest.skip(f"Example script not found: {script_rel}")

    doc_text = doc_path.read_text(encoding="utf-8")
    script_text = script_path.read_text(encoding="utf-8")

    blocks = _extract_python_blocks(doc_text)
    example_lines_norm = [_normalize_line(ln) for ln in script_text.splitlines() if ln.strip()]

    major_deviations: list[SnippetResult] = []
    minor_deviations: list[SnippetResult] = []

    for idx, block in enumerate(blocks):
        # Skip stub / pseudo-code blocks (API-reference signatures, TL;DR pseudo-code, …)
        if _is_stub_block(block):
            continue
        # Skip reference-only blocks (worked-example implementations, anti-patterns, …)
        if _is_reference_only_block(block, script_text):
            continue

        result = _check_snippet(idx, block, example_lines_norm)

        if result.is_empty:
            continue  # no key lines → nothing to check

        score = result.match_score
        if score < _MINOR_THRESHOLD:
            major_deviations.append(result)
        elif score < _ALIGNED_THRESHOLD:
            minor_deviations.append(result)

    # ------------------------------------------------------------------ #
    # Emit UserWarning for minor deviations (test still passes)           #
    # ------------------------------------------------------------------ #
    for result in minor_deviations:
        warnings.warn(
            f"Minor deviation in {doc_rel}:\n" + _format_snippet_report(result, doc_rel),
            UserWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------ #
    # Fail on major deviations                                            #
    # ------------------------------------------------------------------ #
    if major_deviations:
        report_lines = [
            f"Major deviation(s) detected — {len(major_deviations)} snippet(s) in",
            f"  {doc_rel}",
            "are not present as near-verbatim copies in",
            f"  {script_rel}.",
            "",
            "Each documentation Python code block should appear in the example",
            "script with at most minor adaptations (variable renames, added",
            "assertions, wrapping context).  The core logic must match.",
            "",
            "Problematic snippets:",
            "",
        ]
        for result in major_deviations:
            report_lines.append(_format_snippet_report(result, doc_rel))
            report_lines.append("")
        report_lines += [
            "To fix, either:",
            f"  • Update {script_rel} so that it contains near-verbatim copies",
            f"    of the flagged code blocks from {doc_rel}, or",
            f"  • Update {doc_rel} to match what is actually demonstrated in",
            f"    {script_rel}.",
        ]
        pytest.fail("\n".join(report_lines))


# ---------------------------------------------------------------------------
# Standalone verbose reporter
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    # No arguments or explicit --summary flag → compact table
    if not args or args == ["--summary"]:
        _print_summary_table()
        sys.exit(0)

    # --all / --verbose → full block-by-block report for every pair
    if args[0] in ("--all", "--verbose"):
        for doc_rel, script_rel in DOC_TO_SCRIPT.items():
            _report_pair(doc_rel, script_rel)
        print()
        sys.exit(0)

    # Name fragment → verbose block-by-block report for matching pairs
    query = args[0]
    pairs = [
        (doc_rel, script_rel)
        for doc_rel, script_rel in DOC_TO_SCRIPT.items()
        if query in doc_rel or query in script_rel
    ]

    if not pairs:
        print(f"No pairs match {query!r}. Available keys:")
        for k in DOC_TO_SCRIPT:
            print(f"  {k}")
        sys.exit(1)

    for doc_rel, script_rel in pairs:
        _report_pair(doc_rel, script_rel)

    print()
