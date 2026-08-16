"""
Tests for the terminal layer: how results are rendered to a console.

``test_console_output.py`` checks source literals are ASCII, but Rich draws
its own borders, rules and truncation marks, and those are not literals.  So
the real output is rendered here and encoded as cp1252 — the Windows default,
and the encoding that has twice killed a run in this project.
"""

import io
import re
from pathlib import Path

import pandas as pd
import pytest
import yaml

from aethon import console
from aethon.design.report import sweep_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Buffer:
    """
    A text sink that declares an encoding, the way a real console does.

    Not a StringIO subclass because ``encoding`` is read-only there, and the
    encoding is precisely what the box-style fallback keys off.
    """

    def __init__(self, encoding: str):
        self._buffer = io.StringIO()
        self.encoding = encoding

    def write(self, text: str) -> int:
        return self._buffer.write(text)

    def flush(self) -> None:
        self._buffer.flush()

    def isatty(self) -> bool:
        return False

    def getvalue(self) -> str:
        return self._buffer.getvalue()


def _capture(render, width: int = 100, encoding: str = "utf-8") -> str:
    """
    Render through a detached console and return the text it produced.

    *encoding* is what the fake stream advertises, which is what the box-style
    fallback keys off - so passing ``cp1252`` genuinely exercises the Windows
    path rather than merely checking the result afterwards.
    """
    buffer = _Buffer(encoding)
    original_file = console.console.file
    original_width = console.console.width
    console.console.file = buffer
    console.console.width = width
    try:
        render()
    finally:
        console.console.file = original_file
        console.console.width = original_width
    return buffer.getvalue()


def _unwrap(text: str) -> str:
    """
    Reassemble folded output, dropping whitespace and panel borders.

    A folded path is split across lines and may sit inside a bordered panel,
    so recovering it means removing the layout, not just the newlines.
    """
    return re.sub(r"[\s|]+", "", text)


def _make_results():
    rows = []
    for geology, offset in (("Bentonite", 4.0), ("Salt", 1.0)):
        for arch in ("NaturalAir", "WaterPool"):
            for R in (0.1, 0.2, 0.3):
                for loading in (5.0, 10.0, 15.0):
                    heat = R * loading
                    rows.append({
                        "Geology": geology, "Archetype": arch,
                        "Radius_m": R, "Loading_Pct": loading,
                        "N_canisters": max(1, int(100 / heat)),
                        "t_coolers_off_yr": heat,
                        "t_geo_yr": heat * offset,
                        "Binding_At_Geo": "surface",
                        "t_encap_yr": heat * 0.1,
                        "t_active_yr": heat * 0.9,
                        "Feasible": True,
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

class TestWindowsEncoding:
    """
    Rendered output must survive a cp1252 console.

    Rich substitutes ASCII box characters on legacy terminals, but only if we
    never hand it text it cannot downgrade. These render the real thing and
    encode it the way a Windows console would.
    """

    def test_sweep_summary_encodes_as_cp1252(self):
        stats = sweep_stats(_make_results())
        text = _capture(lambda: console.print_sweep_summary(stats),
                        encoding="cp1252")
        text.encode("cp1252")

    def test_candidates_table_encodes_as_cp1252(self):
        df = pd.DataFrame([{
            "Name": "A", "Radius_m": 0.08, "Loading_Pct": 15.0,
            "N_canisters": 412, "Geology": "Salt", "Archetype": "ForcedAir",
            "t_encap_yr": 0.083, "t_coolers_off_yr": 1.1,
            "t_active_yr": 1.017, "t_geo_yr": 1.42,
            "Facility_Duty_W": 4985.2, "Min_H_Active": 18.3,
        }])
        _capture(lambda: console.print_candidates(df),
                 encoding="cp1252").encode("cp1252")

    def test_glossary_encodes_as_cp1252(self):
        _capture(console.print_milestone_glossary,
                 encoding="cp1252").encode("cp1252")

    def test_rules_and_panels_encode_as_cp1252(self):
        def render():
            console.rule("Results")
            console.panel("body text", "Title")
            console.key_values([("Key", "value")], title="Block")
            console.warn("something to check")
            console.hint("a suggestion")

        _capture(render, encoding="cp1252").encode("cp1252")

    def test_long_paths_are_not_truncated_with_an_ellipsis(self):
        """
        Regression: a long output path in a two-column layout was shortened
        with U+2026, which is outside cp1252 and reached the user as a black
        diamond in the middle of the path they needed to open.
        """
        long_path = Path("C:/Users/someone/AppData/Local/Temp/a-very-long-"
                         "scratch-directory-name/results/run-2026-08/"
                         "explore_full_CA_Recycling_Bg-CaF2.csv")
        text = _capture(
            lambda: console.file_list([(long_path, "every design evaluated")]),
            width=70, encoding="cp1252",
        )

        text.encode("cp1252")
        assert "\u2026" not in text
        # The path must survive intact, even if folded across lines
        assert str(long_path) in _unwrap(text)

    def test_key_values_fold_long_paths_rather_than_truncating(self):
        """
        Same regression as the file list: the run header carries the waste
        stream path, and a shortened one is unusable.
        """
        long_value = ("C:/Users/someone/AppData/Local/Temp/a-very-long-scratch-"
                      "directory/results/decay/waste_source.yaml")
        text = _capture(
            lambda: console.key_values([("Waste stream", long_value)],
                                       title="Modelling"),
            width=60, encoding="cp1252",
        )

        text.encode("cp1252")
        assert "\u2026" not in text
        assert long_value in _unwrap(text)

    def test_narrow_console_still_encodes_as_cp1252(self):
        """Truncation only bites when things do not fit, so squeeze them."""
        df = pd.DataFrame([{
            "Name": "A", "Radius_m": 0.08, "Loading_Pct": 15.0,
            "N_canisters": 412, "Geology": "Bentonite",
            "Archetype": "ChilledWaterJacketWithAVeryLongName",
            "t_encap_yr": 0.083, "t_coolers_off_yr": 1.1,
            "t_active_yr": 1.017, "t_geo_yr": 1.42,
            "Facility_Duty_W": 4985.2, "Min_H_Active": 18.3,
        }])
        text = _capture(lambda: console.print_candidates(df),
                        width=40, encoding="cp1252")
        text.encode("cp1252")

    def test_utf8_streams_still_get_the_better_borders(self):
        """
        The ASCII fallback is a fallback. A modern terminal should not be
        punished for Windows' sake.
        """
        text = _capture(lambda: console.panel("body", "Title"),
                        encoding="utf-8")
        assert "─" in text


# ---------------------------------------------------------------------------
# Rendering content
# ---------------------------------------------------------------------------

class TestRendering:

    def test_summary_shows_every_technology_and_geology(self):
        stats = sweep_stats(_make_results())
        text = _capture(lambda: console.print_sweep_summary(stats))

        for name in ("NaturalAir", "WaterPool", "Bentonite", "Salt"):
            assert name in text

    def test_empty_summary_says_so(self):
        text = _capture(lambda: console.print_sweep_summary({}))
        assert "No designs were evaluated" in text

    def test_candidates_duty_is_shown_in_kilowatts(self):
        """Five-digit watt figures crowd the row without informing anyone."""
        df = pd.DataFrame([{
            "Name": "A", "Radius_m": 0.08, "Loading_Pct": 15.0,
            "N_canisters": 30, "Geology": "Salt", "Archetype": "ForcedAir",
            "t_encap_yr": 0.1, "t_coolers_off_yr": 1.0, "t_active_yr": 0.9,
            "t_geo_yr": 1.4, "Facility_Duty_W": 47729.0, "Min_H_Active": 25.1,
        }])
        text = _capture(lambda: console.print_candidates(df))
        assert "47.7" in text
        assert "47729" not in text

    def test_candidates_are_grouped_by_geology_and_technology(self):
        """
        One table per combination, so the two identifying columns become a
        title instead of repeating down every row and forcing a wrap.
        """
        rows = []
        for geology in ("Bentonite", "Salt"):
            for arch in ("NaturalAir", "ForcedAir"):
                rows.append({
                    "Name": "A", "Radius_m": 0.08, "Loading_Pct": 15.0,
                    "N_canisters": 30, "Geology": geology, "Archetype": arch,
                    "t_encap_yr": 0.1, "t_coolers_off_yr": 1.0,
                    "t_active_yr": 0.9, "t_geo_yr": 1.4,
                    "Facility_Duty_W": 1000.0, "Min_H_Active": 25.1,
                })
        text = _capture(lambda: console.print_candidates(pd.DataFrame(rows)),
                        width=110)

        assert text.count("Candidate designs") == 4
        assert "Bentonite / NaturalAir" in text
        assert "Salt / ForcedAir" in text

    def test_candidate_rows_stay_on_one_line(self):
        """A row split over three lines is unreadable however complete it is."""
        df = pd.DataFrame([{
            "Name": "A", "Radius_m": 0.08, "Loading_Pct": 15.0,
            "N_canisters": 30, "Geology": "Salt", "Archetype": "ForcedAir",
            "t_encap_yr": 0.1, "t_coolers_off_yr": 1.0, "t_active_yr": 0.9,
            "t_geo_yr": 1.4, "Facility_Duty_W": 1000.0, "Min_H_Active": 25.1,
        }])
        text = _capture(lambda: console.print_candidates(df), width=100)
        data_lines = [ln for ln in text.splitlines() if "0.080" in ln]
        assert len(data_lines) == 1

    def test_candidates_render_missing_values_as_a_dash(self):
        df = pd.DataFrame([{
            "Name": "A", "Radius_m": 0.08, "Loading_Pct": 15.0,
            "N_canisters": 30, "Geology": "Salt", "Archetype": "NaturalAir",
            "t_encap_yr": float("inf"), "t_coolers_off_yr": 1.0,
            "t_active_yr": float("nan"), "t_geo_yr": 1.4,
            "Facility_Duty_W": float("nan"), "Min_H_Active": float("nan"),
        }])
        text = _capture(lambda: console.print_candidates(df))
        assert "nan" not in text.lower().replace("inf", "")

    def test_empty_candidates_render_nothing(self):
        assert _capture(lambda: console.print_candidates(pd.DataFrame())) == ""


