"""Tests for the standalone decay chain XML parser."""

import os
import tempfile
import numpy as np
import pytest
from decay_preprocessor.chain_parser import parse_chain

# Minimal chain XML: A → B → C (C is stable)
_MINIMAL_XML = """<chain>
  <nuclide name="A" half_life="1.0">
    <decay type="beta-" target="B" branching_ratio="1.0" energy="1000.0"/>
  </nuclide>
  <nuclide name="B" half_life="2.0">
    <decay type="beta-" target="C" branching_ratio="0.9" energy="500.0"/>
    <decay type="alpha" target="C" branching_ratio="0.1" energy="200.0"/>
  </nuclide>
  <nuclide name="C"/>
</chain>"""

_LN2 = 0.6931471805599453


@pytest.fixture(scope="module")
def parsed():
    """Write the minimal XML to a temp file and parse it once."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False
    ) as f:
        f.write(_MINIMAL_XML)
        tmp_path = f.name
    try:
        result = parse_chain(tmp_path)
    finally:
        os.unlink(tmp_path)
    return result


def test_all_nuclides_present(parsed):
    """All three nuclides should appear in the index."""
    nuc_to_idx, _, _, _ = parsed
    assert "A" in nuc_to_idx
    assert "B" in nuc_to_idx
    assert "C" in nuc_to_idx


def test_decay_constants(parsed):
    """Decay constants: λ = ln2 / t½; zero for stable nuclide C."""
    nuc_to_idx, decay_constants, _, _ = parsed
    assert pytest.approx(decay_constants[nuc_to_idx["A"]], rel=1e-6) == _LN2 / 1.0
    assert pytest.approx(decay_constants[nuc_to_idx["B"]], rel=1e-6) == _LN2 / 2.0
    assert decay_constants[nuc_to_idx["C"]] == 0.0


def test_q_values(parsed):
    """
    Q-value is sum(energy * branching_ratio).
    A: 1000 * 1.0 = 1000 eV
    B: 500*0.9 + 200*0.1 = 450 + 20 = 470 eV
    C: 0 (stable)
    """
    nuc_to_idx, _, q_values, _ = parsed
    assert pytest.approx(q_values[nuc_to_idx["A"]], rel=1e-6) == 1000.0
    assert pytest.approx(q_values[nuc_to_idx["B"]], rel=1e-6) == 470.0
    assert q_values[nuc_to_idx["C"]] == 0.0


def test_matrix_diagonal(parsed):
    """Diagonal entries must be −λᵢ."""
    nuc_to_idx, decay_constants, _, matrix_A = parsed
    A_dense = matrix_A.toarray()
    for name in ("A", "B"):
        i = nuc_to_idx[name]
        assert pytest.approx(A_dense[i, i], rel=1e-6) == -decay_constants[i]
    # Stable nuclide C has no diagonal entry
    assert A_dense[nuc_to_idx["C"], nuc_to_idx["C"]] == 0.0


def test_matrix_off_diagonal(parsed):
    """Off-diagonal: B is fed by A with rate λ_A; C is fed by B with rate λ_B."""
    nuc_to_idx, decay_constants, _, matrix_A = parsed
    A_dense = matrix_A.toarray()
    lam_A = decay_constants[nuc_to_idx["A"]]
    lam_B = decay_constants[nuc_to_idx["B"]]

    # A → B  (branching ratio 1.0)
    assert pytest.approx(
        A_dense[nuc_to_idx["B"], nuc_to_idx["A"]], rel=1e-6
    ) == lam_A * 1.0

    # B → C  (branching ratio 0.9 + 0.1 = 1.0 total, both target C)
    assert pytest.approx(
        A_dense[nuc_to_idx["C"], nuc_to_idx["B"]], rel=1e-6
    ) == lam_B * 1.0
