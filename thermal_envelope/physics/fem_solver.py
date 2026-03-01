"""
Transient 1D cylindrical finite-difference solver using Method of Lines.

Governing PDE (cylindrical coordinates, radial symmetry):

    ρ·Cₚ(T)·∂T/∂t = (1/r)·∂/∂r[r·k(T)·∂T/∂r] + Q_vol(t)

where Q_vol(t) = Q_decay(t + t_cool) · ρ · loading.

Boundary conditions:
    r = 0:  ∂T/∂r = 0                        (symmetry)
    r = R:  −k·∂T/∂r = h·(T_s − T_∞)        (Robin / convection)

Discretisation:
    - Interior nodes: central differences for ∂²T/∂r² and (1/r)·∂T/∂r.
    - Centre node (r = 0): L'Hôpital's rule gives lim(1/r · ∂T/∂r) = ∂²T/∂r²,
      so the Laplacian becomes 2·∂²T/∂r² ≈ 4·(T₁ − T₀)/Δr² in cylindrical coords.
    - Surface node (r = R): control-volume energy balance over a half-cell.
"""

import numpy as np
from scipy.integrate import solve_ivp

from thermal_envelope.constants import SEC_PER_YEAR, DEFAULT_RTOL


class WasteForm:
    """
    Finite-difference thermal model for a cylindrical waste canister.

    Solves for the transient radial temperature distribution given:
    - Canister geometry (radius, node count)
    - Material properties (ρ, k(T), Cₚ(T), Q_decay(t))
    - Boundary conditions (ambient temperature, convective HTC)
    """

    def __init__(
        self,
        R: float,
        ambient_T: float,
        h_coeff: float,
        loading_fraction: float,
        properties: dict,
        cooling_years: float,
        effective_density: float,
        n_nodes: int = 50,
    ):
        """
        Parameters
        ----------
        R : float
            Canister outer radius [m].
        ambient_T : float
            Repository ambient temperature [K].
        h_coeff : float
            Convective heat transfer coefficient [W/(m²·K)].
        loading_fraction : float
            Waste oxide loading as a fraction (e.g. 0.05 for 5 %).
        properties : dict
            Keys: ``'decay'`` (Callable(t) → W/kg), ``'cp'`` (Callable(T) → J/(kg·K)),
            ``'k'`` (Callable(T) → W/(m·K)).
        cooling_years : float
            Pre-emplacement cooling time [years].  Shifts the decay curve.
        effective_density : float
            Effective density of the waste form [kg/m³].
        n_nodes : int
            Number of radial finite-difference nodes.
        """
        self.R = R
        self.T_inf = ambient_T
        self.h = h_coeff

        self.loading = loading_fraction
        self.t_cool = cooling_years
        self.rho = effective_density

        # Material property callables
        self.decay_f = properties['decay']
        self.cp_f = properties['cp']
        self.k_f = properties['k']

        # Radial grid
        self.N = n_nodes
        self.r = np.linspace(0, self.R, self.N)
        self.dr = self.R / (self.N - 1)

        with np.errstate(divide='ignore'):
            self.inv_r = 1.0 / self.r
            self.inv_r[0] = 0.0

    def get_source_term(self, t_sim_years: float) -> float:
        """
        Volumetric heat generation rate at simulation time *t*.

        Q_vol(t) = Q_decay(t + t_cool) · ρ · loading   [W/m³]

        Parameters
        ----------
        t_sim_years : float
            Time since emplacement [years].

        Returns
        -------
        float
            Volumetric heat generation [W/m³].
        """
        t_decay = t_sim_years + self.t_cool
        return self.decay_f(t_decay) * self.rho * self.loading

    def model_derivative(self, t_yrs: float, T: np.ndarray) -> np.ndarray:
        """
        Right-hand side of the semi-discrete ODE system  dT/dt = f(t, T).

        Implements the Method of Lines discretisation of the cylindrical heat
        equation.  Time is tracked in **years** but derivative terms are scaled
        by ``SEC_PER_YEAR`` so that the heat source (in Watts = J/s) is
        dimensionally consistent.

        Parameters
        ----------
        t_yrs : float
            Current simulation time [years].
        T : np.ndarray, shape (N,)
            Temperature at each radial node [K].

        Returns
        -------
        np.ndarray, shape (N,)
            Time derivative dT/dt [K/yr].
        """
        k = self.k_f(T)
        cp = self.cp_f(T)

        # Thermal diffusivity and source, scaled to years
        alpha = (k / (self.rho * cp)) * SEC_PER_YEAR
        source_term = (self.get_source_term(t_yrs) / (self.rho * cp)) * SEC_PER_YEAR

        dT_dt = np.zeros_like(T)

        # --- Interior nodes (central differences) ---
        T_right, T_left, T_mid = T[2:], T[:-2], T[1:-1]
        d2T_dr2 = (T_right - 2 * T_mid + T_left) / (self.dr**2)
        dT_dr = (T_right - T_left) / (2 * self.dr)
        dT_dt[1:-1] = alpha[1:-1] * (d2T_dr2 + self.inv_r[1:-1] * dT_dr) + source_term[1:-1]

        # --- Centre node (r = 0): symmetry BC via L'Hôpital ---
        # lim_{r→0} (1/r)·dT/dr = d²T/dr²  ⟹  Laplacian = 2·d²T/dr²
        # Approximated as 4·(T₁ − T₀) / Δr²
        dT_dt[0] = alpha[0] * (4 * (T[1] - T[0]) / (self.dr**2)) + source_term[0]

        # --- Surface node (r = R): Robin BC (control-volume) ---
        T_s = T[-1]
        T_inner = T[-2]

        q_cond = k[-1] * (T_inner - T_s) / self.dr       # conduction in  [W/m²]
        q_conv = self.h * (T_s - self.T_inf)              # convection out [W/m²]

        # Energy balance over the surface half-cell (width Δr/2)
        dT_dt[-1] = (
            (2.0 / (self.rho * cp[-1] * self.dr)) * (q_cond - q_conv)
            + (self.get_source_term(t_yrs) / (self.rho * cp[-1]))
        ) * SEC_PER_YEAR

        return dT_dt

    def peak_detector(self, t: float, T: np.ndarray) -> float:
        """
        Event function for ``solve_ivp``: returns dT/dt at the centre node.

        When this value crosses zero from positive to negative the centre
        temperature has peaked.  Marked as *terminal* so the solver stops.

        Parameters
        ----------
        t : float
            Current time [years].
        T : np.ndarray
            Temperature field [K].

        Returns
        -------
        float
            Approximate dT₀/dt (sign only matters for zero-crossing detection).
        """
        k = self.k_f(T[0])
        cp = self.cp_f(T[0])
        term = SEC_PER_YEAR / (self.rho * cp)
        alpha = k * term
        source = self.get_source_term(t) * term
        d2T = 4 * (T[1] - T[0]) / (self.dr**2)
        return alpha * d2T + source

    peak_detector.terminal = True
    peak_detector.direction = -1

    def solve_for_peak(self, max_years: float = 50.0) -> tuple:
        """
        Integrate the heat equation until the centre temperature peaks.

        The initial condition is a uniform temperature field at T_∞.

        Parameters
        ----------
        max_years : float
            Maximum simulation horizon [years].

        Returns
        -------
        tuple of (t_peak, T_center_K, T_surface_K)
            Peak time [years], centre temperature [K], surface temperature [K].
        """
        T0 = np.ones(self.N) * self.T_inf

        sol = solve_ivp(
            fun=self.model_derivative,
            t_span=(0, max_years),
            y0=T0,
            method='BDF',
            events=self.peak_detector,
            rtol=DEFAULT_RTOL,
        )

        if sol.status == 1:
            # Event triggered: peak detected
            return sol.t[-1], sol.y[0, -1], sol.y[-1, -1]
        else:
            # Reached max_years without peak — return max centre temp
            return sol.t[-1], np.max(sol.y[0]), sol.y[-1, -1]
