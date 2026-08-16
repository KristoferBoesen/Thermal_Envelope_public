"""
Terminal presentation: one place that owns how AETHON talks to a console.

Everything the user sees on a terminal is rendered here, so the rest of the
package can compute results without deciding how they look.  The physics
modules return data; this module turns data into output.

**Why the strings here stay ASCII.**  Windows consoles fail on non-ASCII in two
ways this project has hit: characters outside cp1252 raise UnicodeEncodeError
and kill the run, and characters outside the active code page survive but
render as ``?``.  Rich draws its own box-drawing characters and downgrades them
on legacy terminals, so borders and rules are safe — but any literal text
written here is not, and must stay ASCII.  ``tests/test_console_output.py``
enforces that, and ``tests/test_cli_output.py`` renders the real output through
a cp1252 encoder to prove it survives.

Colour is disabled automatically when stdout is not a terminal, so redirecting
to a file produces clean text rather than escape sequences.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# Muted, colourblind-safe accents. Deliberately restrained: this is a technical
# report, and colour is used to separate sections rather than to decorate.
_THEME = Theme({
    "heading": "bold cyan",
    "key": "dim",
    "value": "bold",
    "good": "green",
    "warn": "yellow",
    "bad": "red",
    "hint": "dim italic",
    "path": "cyan",
})

console = Console(theme=_THEME, highlight=False, soft_wrap=False)
err_console = Console(theme=_THEME, highlight=False, stderr=True)

# Box-drawing characters, and the ASCII fallback for consoles that cannot
# encode them.
_BOX_PROBE = "┌─│"


def _unicode_safe() -> bool:
    """
    Whether the current output stream can encode box-drawing characters.

    Rich substitutes ASCII borders when it *detects a legacy Windows console*,
    which is narrower than the case that actually breaks: any stream whose
    encoding is cp1252 raises UnicodeEncodeError on a border character, and
    that includes pipes and IDE consoles Rich does not classify as legacy.
    Probing the encoding directly covers all of them.
    """
    encoding = getattr(console.file, "encoding", None)
    if not encoding:
        return True  # no declared encoding: assume a modern UTF-8 stream
    try:
        _BOX_PROBE.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def _box() -> Any:
    """Border style for panels and tables, safe for the current stream."""
    return box.SQUARE if _unicode_safe() else box.ASCII


def _rule_char() -> str:
    return "─" if _unicode_safe() else "-"


# ---------------------------------------------------------------------------
# Structural output
# ---------------------------------------------------------------------------

def rule(title: str) -> None:
    """A labelled horizontal rule separating phases of a run."""
    console.rule(f"[heading]{title}[/heading]", align="left",
                 characters=_rule_char())


def blank() -> None:
    console.print()


def hint(message: str) -> None:
    """Secondary guidance: what to do next, or what to check."""
    console.print(f"  {message}", style="hint")


def warn(message: str) -> None:
    console.print(f"Warning: {message}", style="warn")


def error(message: str) -> None:
    """Errors go to stderr so they survive a redirect of stdout."""
    err_console.print(f"Error: {message}", style="bad")


def panel(body: Any, title: str) -> None:
    console.print(Panel(
        body, title=title, title_align="left", border_style="dim", box=_box(),
    ))


def key_values(pairs: Sequence[tuple], title: Optional[str] = None) -> None:
    """
    A two-column block of labelled values.

    Used for the run header, where alignment is what makes the block scannable
    and a table border would only add noise.

    The value column folds rather than truncating.  Values here are often file
    paths, and Rich's default shortening inserts a U+2026 ellipsis - which is
    outside cp1252, so on a Windows console the user gets a corrupted character
    in the middle of a path they need to open.
    """
    table = Table.grid(padding=(0, 2))
    table.add_column(style="key", justify="right", overflow="fold")
    table.add_column(style="value", overflow="fold")
    for label, value in pairs:
        table.add_row(str(label), str(value))
    if title:
        panel(table, title)
    else:
        console.print(table)


def data_table(
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
    title: Optional[str] = None,
    right_align_from: int = 1,
    align: Optional[Sequence[str]] = None,
) -> None:
    """
    A bordered table of results.

    Numeric columns are right-aligned from *right_align_from* onward, which is
    what lets a reader compare magnitudes down a column at a glance.  Pass
    *align* — one of ``"left"``/``"right"`` per column — where that rule does
    not hold, such as a prose column sitting after the numbers: right-aligned
    text that wraps is markedly harder to read than left-aligned.
    """
    table = Table(
        title=title, title_style="heading", title_justify="left",
        border_style="dim", header_style="bold", box=_box(),
    )
    for i, name in enumerate(columns):
        if align is not None:
            justify = align[i]
        else:
            justify = "right" if i >= right_align_from else "left"
        table.add_column(name, justify=justify, overflow="fold")
    for row in rows:
        table.add_row(*[str(cell) for cell in row])
    console.print(table)


def file_list(entries: Sequence[tuple]) -> None:
    """
    What the run wrote, and what each file is for.

    Printed last because it is what the user acts on next — which means the
    paths must be complete and copy-pasteable.  Two columns would let Rich
    truncate a long path with an ellipsis, so each path gets its own full-width
    line with the description indented beneath it.
    """
    for path, description in entries:
        console.print(Text(str(path), style="path"), overflow="fold")
        console.print(f"    {description}", style="hint")


@contextmanager
def progress_bar(description: str, total: int) -> Iterator[Any]:
    """
    A progress bar with elapsed and remaining time, for the long sweep.

    The encapsulation sweep runs for minutes and previously showed a single
    rewriting line, which gave no sense of whether it was a tenth or nine
    tenths done. Yields an advance callable; a no-op when output is redirected,
    so log files do not fill with bar frames.
    """
    if not console.is_terminal:
        yield lambda: None
        return

    with Progress(
        SpinnerColumn(style="heading"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="cyan", finished_style="green"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(description, total=total)
        yield lambda: progress.advance(task)


# ---------------------------------------------------------------------------
# Run output
# ---------------------------------------------------------------------------

def print_provenance(cfg: Dict[str, Any], material: str) -> None:
    """
    State what the run is actually modelling, before doing any work.

    The decay curve is the input most easily got wrong without noticing: with
    no ``waste_source`` the solver falls back to inline ``decay_terms``, which
    in a fresh checkout are placeholders. That produces confident, plausible
    results for waste nobody has. Showing the source and the heat output at
    shutdown makes the substitution visible in one glance - 122 W/kg reads very
    differently from 1683 W/kg.
    """
    source = cfg.get("waste_source_path")
    if source:
        try:  # a short relative path reads better than an absolute one
            source = str(Path(source).relative_to(Path.cwd()))
        except ValueError:
            pass

    Q0 = float(cfg["waste_form"]["decay"](0.0))
    window = (f"{cfg['pre_encap_min_years']:g} to "
              f"{cfg['pre_encap_max_years']:g} yr from shutdown")

    pairs = [
        ("Material", material),
        ("Waste stream", source or "inline decay_terms in the config file"),
        ("Q(0)", f"{Q0:.6g} W/kg"),
        ("Campaign", f"{cfg['total_waste_mass_kg']:g} kg"),
        ("Encapsulation", window),
        ("Safety factor", f"{cfg['safety_factor']:g}"),
    ]
    key_values(pairs, title="Modelling")

    if not source:
        warn("No 'waste_source' set - check these are your numbers, "
             "not the shipped placeholders.")


def print_sweep_summary(stats: Dict[str, Any]) -> None:
    """
    Render the summary of a completed sweep.

    Takes the plain mapping from :func:`aethon.design.report.sweep_stats` so
    the numbers can be tested without going near a terminal.
    """
    if not stats:
        console.print("No designs were evaluated.", style="bad")
        hint("Check the radius and loading grids in solver_config.yaml.")
        return

    key_values([
        ("Designs", f"{stats['n_designs']}"),
        ("Radius", f"{stats['radius_min']:.3f} to {stats['radius_max']:.3f} m"),
        ("Loading", f"{stats['loading_min']:g} to {stats['loading_max']:g} wt%"),
    ], title="Swept")

    if stats["archetypes"]:
        rows = []
        for entry in stats["archetypes"]:
            earliest = ("-" if entry["earliest_encap"] is None
                        else f"{entry['earliest_encap']:.3f}")
            rows.append([
                entry["name"],
                f"{entry['n_feasible']} of {entry['n_total']} "
                f"({entry['share_pct']:.0f}%)",
                earliest,
            ])
        data_table(
            ["Cooling technology", "Can encapsulate", "Earliest t_encap (yr)"],
            rows,
            title="What each technology can handle, within the window",
        )

    rows = []
    for entry in stats["geologies"]:
        if entry["t_geo_min"] is None:
            rows.append([entry["name"], "not reachable in the search window", ""])
        else:
            rows.append([
                entry["name"],
                f"{entry['t_geo_min']:.2f}",
                f"{entry['t_geo_max']:.2f}",
            ])
    data_table(
        ["Geology", "Earliest t_geo (yr)", "Latest t_geo (yr)"],
        rows,
        title="Time to repository emplacement",
    )

    if stats["n_unexpected_binding"]:
        warn(
            f"{stats['n_unexpected_binding']} design(s) are centreline-limited "
            "at emplacement, not surface-limited. This is unusual - check the "
            "geometry and material limits for those rows."
        )


def print_milestone_glossary() -> None:
    """The three-milestone model, restated where the numbers appear."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="value")
    table.add_column(style="hint")
    for name, meaning in (
        ("t_encap", "waste sealed into a canister, active cooling begins"),
        ("t_coolers_off", "centreline passively safe; coolers off, interim store"),
        ("t_geo", "surface passively safe for the buffer; emplacement"),
        ("t_active", "= t_coolers_off - t_encap, how long the plant runs"),
    ):
        table.add_row(name, meaning)
    panel(table, "Milestones (years from reactor shutdown)")
    hint("t_coolers_off does not depend on the cooling technology: a stronger")
    hint("technology buys earlier encapsulation, not an earlier finish.")


# Geology and Archetype are deliberately absent: they become the table title
# rather than two columns repeated down every row, which is what lets the
# remaining ten fit on one line at a normal terminal width.
_CANDIDATE_COLUMNS = [
    ("Name", "Name", None),
    ("Radius_m", "R (m)", "{:.3f}"),
    ("Loading_Pct", "Load %", "{:.1f}"),
    ("N_canisters", "N", "{:.0f}"),
    ("t_encap_yr", "t_encap", "{:.3f}"),
    ("t_coolers_off_yr", "t_cool_off", "{:.3f}"),
    ("t_active_yr", "t_active", "{:.3f}"),
    ("t_geo_yr", "t_geo", "{:.2f}"),
    ("Facility_Duty_W", "Duty kW", None),
    ("Min_H_Active", "Min h", "{:.1f}"),
]


def _candidate_cell(column: str, value: Any, fmt: Optional[str]) -> str:
    """One formatted cell, with the non-finite cases spelled out."""
    if column == "Facility_Duty_W":
        return "-" if value != value else f"{value / 1000.0:.1f}"
    if fmt is None:
        return str(value)
    if value != value:  # NaN
        return "-"
    if value in (float("inf"), float("-inf")):
        return "inf"
    return fmt.format(value)


def print_candidates(candidates) -> None:
    """
    Render the named candidate designs, one table per geology and technology.

    Grouping rather than adding two more columns keeps each row on one line:
    twelve columns wrapped at any normal width, and a row split over three
    lines is unreadable however complete it is.

    Facility duty is shown in kilowatts - the watt figures run to five digits,
    which crowds the row without telling anyone anything they act on.
    """
    if candidates is None or candidates.empty:
        return

    present = [(col, head, fmt) for col, head, fmt in _CANDIDATE_COLUMNS
               if col in candidates.columns]
    headers = [head for _, head, _ in present]

    group_cols = [c for c in ("Geology", "Archetype") if c in candidates.columns]

    if not group_cols:
        groups = [(None, candidates)]
    else:
        groups = list(candidates.groupby(group_cols, sort=False))

    for key, group in groups:
        rows = [
            [_candidate_cell(col, row[col], fmt) for col, _, fmt in present]
            for _, row in group.iterrows()
        ]
        if key is None:
            title = "Candidate designs"
        else:
            parts = key if isinstance(key, tuple) else (key,)
            title = "Candidate designs: " + " / ".join(str(p) for p in parts)
        data_table(headers, rows, title=title, right_align_from=1)

    hint("Times in years from reactor shutdown. Min h is the HTC this design "
         "needs, at that technology's ambient.")


