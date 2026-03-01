"""
Immutable physical constants and internal solver parameters.

These are not user-configurable — they are fundamental physical or
numerical values that should not change between runs.
"""

# --- Physical Constants ------------------------------------------------------
SEC_PER_YEAR: float = 365.25 * 24 * 3600   # [s/yr]
KELVIN_OFFSET: float = 273.15              # K = °C + KELVIN_OFFSET

# --- Root-Finder Search Bounds -----------------------------------------------
# These define the feasible search space for the optimisation loops.
H_SEARCH_MIN: float = 0.1       # Minimum bracket for h search [W/(m²·K)]
H_SEARCH_MAX: float = 2000.0    # Upper bracket for h search [W/(m²·K)]
T_SEARCH_MAX_YEARS: float = 1000.0  # Upper bracket for cooling time [yr]

# --- Solver Internals --------------------------------------------------------
DEFAULT_RTOL: float = 1e-3      # Relative tolerance for solve_ivp
