"""
Standalone parser for OpenMC-format decay chain XML files.

Extracts nuclide data (decay constants, mean decay energies, branching ratios)
and builds the sparse Bateman transmutation matrix — without requiring OpenMC
or any non-standard dependency.

Two XML formats are supported:

**Primary format** — ``<depletion_chain>`` (OpenMC ≥ 0.13, ENDF/B-VII.1 chain files).
The total decay energy is stored as a single pre-summed attribute on the
``<nuclide>`` element::

    <depletion_chain>
      <nuclide name="H3" half_life="388789600.0" decay_energy="5690.0">
        <decay type="beta-" target="He3" branching_ratio="1.0"/>
      </nuclide>
      <nuclide name="He3"/>   <!-- stable: no half_life -->
      ...
    </depletion_chain>

Each ``<nuclide>`` element has:
- ``name``          — nuclide identifier string (e.g. ``"H3"``)
- ``half_life``     — physical half-life [s] (absent for stable nuclides)
- ``decay_energy``  — total mean decay energy per disintegration [eV] (pre-summed)

**Fallback format** — ``<chain>`` (older OpenMC convention).
The energy is stored per decay mode on each ``<decay>`` child::

    <chain>
      <nuclide name="Co60" half_life="1.6625e+08">
        <decay type="beta-" target="Ni60" branching_ratio="1.0" energy="96624.64"/>
      </nuclide>
      ...
    </chain>

Each ``<decay>`` child of a nuclide has:
- ``target``           — daughter nuclide name
- ``branching_ratio``  — fraction of decays via this mode
- ``energy``           — mean energy deposited per decay via this mode [eV]
"""

import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from typing import Dict, Tuple

_LN2 = 0.6931471805599453


def parse_chain(
    chain_path: str | Path,
) -> Tuple[Dict[str, int], np.ndarray, np.ndarray, sp.csc_matrix]:
    """
    Parse a decay chain XML file and build the Bateman transmutation matrix.

    Parameters
    ----------
    chain_path : str or Path
        Path to the chain XML file.

    Returns
    -------
    nuc_to_idx : dict
        Mapping of nuclide name → matrix row/column index.
    decay_constants : np.ndarray, shape (N,)
        Decay constant λ = ln2 / t½  [1/s].  Zero for stable nuclides.
    q_values : np.ndarray, shape (N,)
        Mean decay energy deposited per disintegration [eV].  Read from the
        ``decay_energy`` attribute on the ``<nuclide>`` element if present
        (depletion_chain format), otherwise computed as
        ``Σ energy_i · branching_ratio_i`` over all ``<decay>`` children.
    matrix_A : scipy.sparse.csc_matrix, shape (N, N)
        Bateman transmutation matrix.  Diagonal entry (i, i) is ``−λᵢ``;
        off-diagonal entry (j, i) is ``+λᵢ · branching_ratio`` for mode i → j.
    """
    tree = ET.parse(chain_path)
    root = tree.getroot()

    nuclide_elements = root.findall("nuclide")

    # First pass: build index mapping (preserves XML ordering)
    nuc_to_idx: Dict[str, int] = {
        el.attrib["name"]: i for i, el in enumerate(nuclide_elements)
    }
    N = len(nuclide_elements)

    decay_constants = np.zeros(N, dtype=float)
    q_values = np.zeros(N, dtype=float)

    data, rows, cols = [], [], []

    for i, el in enumerate(nuclide_elements):
        half_life_str = el.attrib.get("half_life")
        if half_life_str is None:
            continue  # stable nuclide — no diagonal loss or daughter production

        half_life = float(half_life_str)
        if half_life <= 0.0:
            continue

        lam = _LN2 / half_life  # [1/s]
        decay_constants[i] = lam

        # Diagonal loss term
        data.append(-lam)
        rows.append(i)
        cols.append(i)

        # Q-value: prefer pre-summed decay_energy on the nuclide element
        # (depletion_chain format); fall back to summing energy*br from
        # <decay> children (older chain format).
        de_str = el.attrib.get("decay_energy")
        if de_str is not None:
            q_values[i] = float(de_str)

        # Decay modes: daughter production (and energy if using fallback format)
        total_q_fallback = 0.0
        for mode in el.findall("decay"):
            br = float(mode.attrib.get("branching_ratio", 1.0))
            energy_ev = float(mode.attrib.get("energy", 0.0))
            total_q_fallback += energy_ev * br

            target = mode.attrib.get("target")
            if target and target in nuc_to_idx:
                j = nuc_to_idx[target]
                data.append(lam * br)
                rows.append(j)
                cols.append(i)

        if de_str is None:
            q_values[i] = total_q_fallback

    matrix_A = sp.csc_matrix((data, (rows, cols)), shape=(N, N))
    return nuc_to_idx, decay_constants, q_values, matrix_A
