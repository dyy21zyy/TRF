from __future__ import annotations

"""
Paper-consistent computational experiments for a Bayesian-network-based robust
intervention-planning model.

Implements:
- Section 3 / 3.3 / 3.4 model structure and robust adjustment logic
- Section 4 epsilon-constraint solution method
- Section 6.1 stylized experimental design parameter generator
- Section 6.2 / 6.3 / 6.4 experiment modules
"""

import os
import math
import itertools
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Core constants and data types
# -----------------------------

STATES: Tuple[str, ...] = ("L", "M", "H")
SCENARIOS: Tuple[str, ...] = ("omega1", "omega2", "omega3")

NODES: Dict[str, str] = {
    "N1": "signal waiting time",
    "N2": "leading pedestrian behavior",
    "N3": "vehicle dynamic state",
    "N4": "visibility and context",
    "N5": "patience and urgency",
    "N6": "conformity psychology",
    "N7": "dynamic risk perception",
    "N8": "crossing intention",
    "N9": "illegal crossing behavior",
    "N10": "accident risk",
}

PARENTS: Dict[str, Tuple[str, ...]] = {
    "N1": tuple(),
    "N2": tuple(),
    "N3": tuple(),
    "N4": tuple(),
    "N5": ("N1",),
    "N6": ("N2",),
    "N7": ("N3", "N4"),
    "N8": ("N5", "N6", "N7"),
    "N9": ("N8",),
    "N10": ("N3", "N9"),
}

ROOT_NODES = ("N1", "N2", "N3", "N4")
CHILD_NODES = ("N5", "N6", "N7", "N8", "N9", "N10")
PSYCH_NODES = ("N5", "N6", "N7")
BEHAVIOR_NODES = ("N8", "N9")
TOPO_ORDER = ("N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10")

ACTIONS: Dict[str, str] = {
    "A1": "adaptive signal timing optimization",
    "A2": "dynamic speed harmonization",
    "A3": "pedestrian countdown signal",
    "A4": "smart directional audio warning",
    "A5": "intelligent active pavement markers",
}
ACTION_ORDER = ("A1", "A2", "A3", "A4", "A5")


@dataclass(frozen=True)
class ModelConfig:
    seed: int = 20260420

    # Section 6.1 baseline probability generator controls
    root_H_range: Tuple[float, float] = (0.20, 0.40)
    root_M_range: Tuple[float, float] = (0.25, 0.40)

    alpha_range: Tuple[float, float] = (0.05, 0.15)
    beta_fixed: float = 0.40
    eta_default: float = 1.0

    # Synthetic M-rule controls for CPT generation
    child_M_base_range: Tuple[float, float] = (0.15, 0.28)
    child_M_per_H_parent: float = 0.03

    # delta uncertainty settings
    delta_root_fixed: float = 0.05
    delta_psych_range: Tuple[float, float] = (0.08, 0.12)
    delta_behavior_range: Tuple[float, float] = (0.04, 0.08)
    # Option A chosen for N10 in code comments below.
    delta_target_range: Tuple[float, float] = (0.04, 0.08)

    # Gamma generation standard deviations
    gamma_sd_strong: float = 0.02
    gamma_sd_behavior: float = 0.05
    gamma_sd_a5: float = 0.04

    # Theta shock settings
    theta_psych_range: Tuple[float, float] = (0.10, 0.20)
    theta_env_range: Tuple[float, float] = (0.25, 0.40)

    # Costs (kappa)
    costs: Dict[str, float] = None

    # epsilon grid settings
    epsilon_grid_points: int = 21

    def __post_init__(self):
        if self.costs is None:
            object.__setattr__(
                self,
                "costs",
                {"A1": 10.0, "A2": 8.5, "A3": 4.0, "A4": 3.5, "A5": 5.0},
            )


@dataclass
class ParameterBundle:
    bar_pi_root: Dict[str, Dict[str, float]]
    bar_pi_child: Dict[str, Dict[Tuple[str, ...], Dict[str, float]]]
    alpha: Dict[str, float]
    beta: Dict[str, float]
    eta: Dict[str, Dict[str, float]]

    # delta applies to H state only
    delta_H_root: Dict[str, float]
    delta_H_child: Dict[str, Dict[Tuple[str, ...], float]]

    # Gamma maps action effects to targeted node / CPT entries
    gamma_root: Dict[str, float]  # only for A1->N1 and A2->N3
    gamma_child: Dict[str, Dict[Tuple[str, ...], float]]  # A3/A4/A5 by parent config

    # Theta by scenario / node / parent-config / state
    theta_root: Dict[str, Dict[str, Dict[str, float]]]
    theta_child: Dict[str, Dict[str, Dict[Tuple[str, ...], Dict[str, float]]]]

    costs: Dict[str, float]


# -----------------------------
# Utility helpers
# -----------------------------


def _all_parent_configs(node: str) -> List[Tuple[str, ...]]:
    parents = PARENTS[node]
    if not parents:
        return [tuple()]
    return list(itertools.product(STATES, repeat=len(parents)))


def _clip_and_normalize(pi_H_star: float, pi_M_star: float) -> Dict[str, float]:
    pi_H_star = float(np.clip(pi_H_star, 0.0, 1.0))
    pi_M_star = float(np.clip(pi_M_star, 0.0, 1.0))
    S = pi_H_star + pi_M_star
    if S <= 1.0:
        hat_H = pi_H_star
        hat_M = pi_M_star
        hat_L = 1.0 - S
    else:
        hat_H = pi_H_star / S
        hat_M = pi_M_star / S
        hat_L = 0.0
    return {"L": hat_L, "M": hat_M, "H": hat_H}


def _portfolio_space() -> List[Tuple[int, int, int, int, int]]:
    return list(itertools.product([0, 1], repeat=5))


def _x_tuple_to_dict(x: Tuple[int, int, int, int, int]) -> Dict[str, int]:
    return {a: int(v) for a, v in zip(ACTION_ORDER, x)}


def _portfolio_id(x: Dict[str, int]) -> str:
    return "".join(str(x[a]) for a in ACTION_ORDER)


# ---------------------------------------------
# 6.1 Experimental design: parameter generation
# ---------------------------------------------


def generate_baseline_parameters(config: ModelConfig | None = None) -> ParameterBundle:
    """
    Generate synthetic but structured baseline priors for Section 6.1.
    All values are stylized modeling inputs, not empirical ground truth.
    """
    config = config or ModelConfig()
    rng = np.random.default_rng(config.seed)

    # Root baseline marginals bar_pi[i, empty, k]
    bar_pi_root: Dict[str, Dict[str, float]] = {}
    for n in ROOT_NODES:
        pH = rng.uniform(*config.root_H_range)
        m_upper = min(config.root_M_range[1], 1.0 - pH - 0.02)
        m_lower = config.root_M_range[0]
        if m_upper < m_lower:
            m_lower = max(0.05, 1.0 - pH - 0.15)
            m_upper = max(m_lower + 1e-6, 1.0 - pH - 0.02)
        pM = rng.uniform(m_lower, m_upper)
        pL = 1.0 - pH - pM
        bar_pi_root[n] = {"L": pL, "M": pM, "H": pH}

    alpha = {n: rng.uniform(*config.alpha_range) for n in CHILD_NODES}
    beta = {n: config.beta_fixed for n in CHILD_NODES}
    eta: Dict[str, Dict[str, float]] = {
        n: {p: config.eta_default for p in PARENTS[n]} for n in CHILD_NODES
    }

    # Child baseline CPT bar_pi[i, spa, k]
    bar_pi_child: Dict[str, Dict[Tuple[str, ...], Dict[str, float]]] = {}
    for n in CHILD_NODES:
        bar_pi_child[n] = {}
        for spa in _all_parent_configs(n):
            high_count = sum(1 for s in spa if s == "H")
            weighted_sum = sum(eta[n][p] * (1.0 if s == "H" else 0.0) ** 2 for p, s in zip(PARENTS[n], spa))
            pH = min(0.98, alpha[n] + beta[n] * math.sqrt(weighted_sum))

            # Synthetic moderate-risk rule (paper-consistent baseline assumption)
            pM_base = rng.uniform(*config.child_M_base_range)
            pM = pM_base + config.child_M_per_H_parent * high_count
            pM = min(pM, 0.90 - pH)
            if pM < 0.0:
                pM = 0.0

            if pH + pM > 1.0:
                scale = 1.0 / (pH + pM)
                pH *= scale
                pM *= scale
            pL = 1.0 - pH - pM
            bar_pi_child[n][spa] = {"L": pL, "M": pM, "H": pH}

    # delta settings (H-state only)
    delta_H_root = {n: config.delta_root_fixed for n in ROOT_NODES}
    delta_H_child: Dict[str, Dict[Tuple[str, ...], float]] = {n: {} for n in CHILD_NODES}

    for n in PSYCH_NODES:
        for spa in _all_parent_configs(n):
            delta_H_child[n][spa] = rng.uniform(*config.delta_psych_range)
    for n in BEHAVIOR_NODES:
        for spa in _all_parent_configs(n):
            delta_H_child[n][spa] = rng.uniform(*config.delta_behavior_range)

    # N10 treatment choice for Section 6.1.D: Option A
    # We treat N10 as a behavioral/result-like node with moderate conservatism.
    for spa in _all_parent_configs("N10"):
        delta_H_child["N10"][spa] = rng.uniform(*config.delta_target_range)

    # Gamma effects: baseline hierarchy
    gamma_root = {
        "A1": rng.normal(-0.32, config.gamma_sd_strong),  # targets N1
        "A2": rng.normal(-0.28, config.gamma_sd_strong),  # targets N3
    }

    gamma_child = {
        "A3": {spa: rng.normal(-0.18, config.gamma_sd_behavior) for spa in _all_parent_configs("N5")},
        "A4": {spa: rng.normal(-0.15, config.gamma_sd_behavior) for spa in _all_parent_configs("N6")},
        "A5": {spa: rng.normal(+0.25, config.gamma_sd_a5) for spa in _all_parent_configs("N7")},
    }

    # Theta scenario shocks
    theta_root = {om: {n: {"H": 0.0, "M": 0.0} for n in ROOT_NODES} for om in SCENARIOS}
    theta_child = {
        om: {
            n: {spa: {"H": 0.0, "M": 0.0} for spa in _all_parent_configs(n)}
            for n in CHILD_NODES
        }
        for om in SCENARIOS
    }

    # omega2 high-stress commute -> N5,N6 primarily H-state
    for n in ("N5", "N6"):
        tH = rng.uniform(*config.theta_psych_range)
        for spa in _all_parent_configs(n):
            theta_child["omega2"][n][spa]["H"] = tH
            # mild M shift (documented baseline assumption)
            theta_child["omega2"][n][spa]["M"] = 0.10 * tH

    # omega3 adverse environment -> N3,N4 roots and N7 CPT H-state
    for n in ("N3", "N4"):
        tH = rng.uniform(*config.theta_env_range)
        theta_root["omega3"][n]["H"] = tH
        theta_root["omega3"][n]["M"] = 0.05 * tH

    tH_n7 = rng.uniform(*config.theta_env_range)
    for spa in _all_parent_configs("N7"):
        theta_child["omega3"]["N7"][spa]["H"] = tH_n7
        theta_child["omega3"]["N7"][spa]["M"] = 0.05 * tH_n7

    return ParameterBundle(
        bar_pi_root=bar_pi_root,
        bar_pi_child=bar_pi_child,
        alpha=alpha,
        beta=beta,
        eta=eta,
        delta_H_root=delta_H_root,
        delta_H_child=delta_H_child,
        gamma_root=gamma_root,
        gamma_child=gamma_child,
        theta_root=theta_root,
        theta_child=theta_child,
        costs=dict(config.costs),
    )


# --------------------------------------
# Probability adjustment (Section 3.3/3.4)
# --------------------------------------


def apply_root_probability_adjustment(
    node: str,
    x: Dict[str, int],
    omega: str,
    params: ParameterBundle,
) -> Dict[str, float]:
    """
    Root-node adjustment exactly following specified structure:
    tilde(H) = bar(H) + sum_a x_a*Gamma + Theta(H)
    tilde(M) = bar(M) + Theta(M)
    robust H: tilde_robust(H)=tilde(H)+delta
    then clipping and S-based normalization.
    """
    base = params.bar_pi_root[node]

    gamma_term = 0.0
    if node == "N1":
        gamma_term += x["A1"] * params.gamma_root["A1"]
    if node == "N3":
        gamma_term += x["A2"] * params.gamma_root["A2"]

    theta = params.theta_root[omega][node]
    tilde_H = base["H"] + gamma_term + theta["H"]
    tilde_M = base["M"] + theta["M"]

    pi_star_H = tilde_H + params.delta_H_root[node]
    pi_star_M = tilde_M
    return _clip_and_normalize(pi_star_H, pi_star_M)


def apply_child_probability_adjustment(
    node: str,
    spa: Tuple[str, ...],
    x: Dict[str, int],
    omega: str,
    params: ParameterBundle,
) -> Dict[str, float]:
    """
    Child-node adjustment with same structure as roots, where Gamma acts on CPT entries.
    """
    base = params.bar_pi_child[node][spa]

    gamma_term = 0.0
    if node == "N5":
        gamma_term += x["A3"] * params.gamma_child["A3"][spa]
    elif node == "N6":
        gamma_term += x["A4"] * params.gamma_child["A4"][spa]
    elif node == "N7":
        gamma_term += x["A5"] * params.gamma_child["A5"][spa]

    theta = params.theta_child[omega][node][spa]
    tilde_H = base["H"] + gamma_term + theta["H"]
    tilde_M = base["M"] + theta["M"]

    pi_star_H = tilde_H + params.delta_H_child[node][spa]
    pi_star_M = tilde_M
    return _clip_and_normalize(pi_star_H, pi_star_M)


# -----------------------------
# BN inference (paper-consistent)
# -----------------------------


def infer_bn_marginals(
    x: Dict[str, int],
    omega: str,
    params: ParameterBundle,
) -> Tuple[Dict[str, Dict[str, float]], float]:
    """
    Returns all node marginals mu and terminal mean-field mu_10_H.

    1) Roots: mu_i_k = hat_pi(i,empty,k)
    2) Child/intermediate: mu_i_k = sum_spa hat_pi(i,spa,k) * prod parent marginals
    3) Terminal N10 uses paper mean-field approximation:
       mu_10_H = sum_{s3,s9} hat_pi_10[(s3,s9),H] * mu_3[s3] * mu_9[s9]
    """
    mu: Dict[str, Dict[str, float]] = {}

    for n in ROOT_NODES:
        mu[n] = apply_root_probability_adjustment(n, x, omega, params)

    for n in ("N5", "N6", "N7", "N8", "N9"):
        mu[n] = {s: 0.0 for s in STATES}
        for spa in _all_parent_configs(n):
            hat = apply_child_probability_adjustment(n, spa, x, omega, params)
            prob_parents = 1.0
            for p, sp in zip(PARENTS[n], spa):
                prob_parents *= mu[p][sp]
            for s in STATES:
                mu[n][s] += hat[s] * prob_parents

    # Mean-field terminal formula (required method)
    mu10_H = 0.0
    for s3, s9 in itertools.product(STATES, STATES):
        hat = apply_child_probability_adjustment("N10", (s3, s9), x, omega, params)
        mu10_H += hat["H"] * mu["N3"][s3] * mu["N9"][s9]

    # Complete N10 marginal for bookkeeping (H from formula, L/M split proportionally)
    mu10_M_raw = 0.0
    mu10_L_raw = 0.0
    for s3, s9 in itertools.product(STATES, STATES):
        hat = apply_child_probability_adjustment("N10", (s3, s9), x, omega, params)
        mu10_M_raw += hat["M"] * mu["N3"][s3] * mu["N9"][s9]
        mu10_L_raw += hat["L"] * mu["N3"][s3] * mu["N9"][s9]
    total = mu10_H + mu10_M_raw + mu10_L_raw
    mu["N10"] = {"H": mu10_H / total, "M": mu10_M_raw / total, "L": mu10_L_raw / total}

    return mu, mu10_H


def _infer_terminal_benchmark_dependence_aware(
    x: Dict[str, int], omega: str, params: ParameterBundle
) -> float:
    """
    Dependence-aware benchmark for diagnostics (Section 6.3.1).
    Computes P(N10=H) from the exact joint P(N3,N9) induced by the BN,
    instead of mean-field factorization P(N3,N9)≈P(N3)P(N9).
    """
    # root marginals
    r = {n: apply_root_probability_adjustment(n, x, omega, params) for n in ROOT_NODES}

    joint_n3_n9 = {(s3, s9): 0.0 for s3 in STATES for s9 in STATES}

    # Enumerate all upstream states needed for exact P(N3,N9)
    for s1, s2, s3, s4 in itertools.product(STATES, STATES, STATES, STATES):
        p_roots = r["N1"][s1] * r["N2"][s2] * r["N3"][s3] * r["N4"][s4]

        p5 = apply_child_probability_adjustment("N5", (s1,), x, omega, params)
        p6 = apply_child_probability_adjustment("N6", (s2,), x, omega, params)
        p7 = apply_child_probability_adjustment("N7", (s3, s4), x, omega, params)

        for s5, s6, s7 in itertools.product(STATES, STATES, STATES):
            p_567 = p5[s5] * p6[s6] * p7[s7]
            p8 = apply_child_probability_adjustment("N8", (s5, s6, s7), x, omega, params)
            for s8 in STATES:
                p9 = apply_child_probability_adjustment("N9", (s8,), x, omega, params)
                for s9 in STATES:
                    joint_n3_n9[(s3, s9)] += p_roots * p_567 * p8[s8] * p9[s9]

    mu10H = 0.0
    for s3, s9 in itertools.product(STATES, STATES):
        p10 = apply_child_probability_adjustment("N10", (s3, s9), x, omega, params)
        mu10H += p10["H"] * joint_n3_n9[(s3, s9)]
    return mu10H


# -------------------------------------
# Portfolio evaluation + robust objective
# -------------------------------------


def evaluate_portfolio(
    x: Dict[str, int],
    params: ParameterBundle,
    use_terminal_benchmark: bool = False,
) -> Dict[str, float]:
    """Evaluate one intervention portfolio under all scenarios."""
    scenario_risks = {}
    for om in SCENARIOS:
        if use_terminal_benchmark:
            scenario_risks[om] = _infer_terminal_benchmark_dependence_aware(x, om, params)
        else:
            _, mu10H = infer_bn_marginals(x, om, params)
            scenario_risks[om] = mu10H

    Z1 = float(sum(params.costs[a] * x[a] for a in ACTION_ORDER))
    Z2 = float(max(scenario_risks.values()))
    out = {
        "portfolio": _portfolio_id(x),
        "Z1_cost": Z1,
        "Z2_worst_risk": Z2,
        **{f"risk_{om}": scenario_risks[om] for om in SCENARIOS},
        **{a: x[a] for a in ACTION_ORDER},
    }
    return out


def evaluate_all_portfolios(
    params: ParameterBundle,
    use_terminal_benchmark: bool = False,
) -> pd.DataFrame:
    rows = []
    for xt in _portfolio_space():
        x = _x_tuple_to_dict(xt)
        rows.append(evaluate_portfolio(x, params, use_terminal_benchmark=use_terminal_benchmark))
    df = pd.DataFrame(rows).sort_values(["Z1_cost", "Z2_worst_risk", "portfolio"]).reset_index(drop=True)
    return df


# --------------------------------------------------
# Section 4 solution method: epsilon-constraint grid
# --------------------------------------------------


def solve_epsilon_constraint_grid(
    all_portfolios_df: pd.DataFrame,
    epsilon_grid_points: int,
    epsilon_min: float,
    epsilon_max: float,
) -> pd.DataFrame:
    """
    This implements the paper's epsilon-constraint solution method.

    Subproblem for each epsilon:
        minimize Psi(x)=max_omega mu_10_H(x,omega)
        s.t. total cost <= epsilon

    Exhaustive enumeration is used only because the action space is small
    (2^5 = 32 portfolios), while preserving epsilon-constraint structure.
    """
    eps_values = np.linspace(epsilon_min, epsilon_max, epsilon_grid_points)
    picks = []
    for eps in eps_values:
        feasible = all_portfolios_df[all_portfolios_df["Z1_cost"] <= eps + 1e-12]
        if feasible.empty:
            continue
        best = feasible.sort_values(["Z2_worst_risk", "Z1_cost", "portfolio"]).iloc[0].to_dict()
        best["epsilon"] = float(eps)
        picks.append(best)
    return pd.DataFrame(picks)


def compute_pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Dominance filtering for bi-objective minimization (Z1,Z2)."""
    records = df[["portfolio", "Z1_cost", "Z2_worst_risk", *ACTION_ORDER]].drop_duplicates().to_dict("records")
    pareto = []
    for i, r in enumerate(records):
        dominated = False
        for j, q in enumerate(records):
            if i == j:
                continue
            weakly_better = (q["Z1_cost"] <= r["Z1_cost"] + 1e-12) and (
                q["Z2_worst_risk"] <= r["Z2_worst_risk"] + 1e-12
            )
            strictly_better = (q["Z1_cost"] < r["Z1_cost"] - 1e-12) or (
                q["Z2_worst_risk"] < r["Z2_worst_risk"] - 1e-12
            )
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    return pd.DataFrame(pareto).sort_values(["Z1_cost", "Z2_worst_risk", "portfolio"]).reset_index(drop=True)


# -----------------------
# Plotting and I/O helpers
# -----------------------


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _plot_pareto_scatter(all_df: pd.DataFrame, pareto_df: pd.DataFrame, save_path: str, title: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(all_df["Z1_cost"], all_df["Z2_worst_risk"], alpha=0.5, label="all portfolios")
    plt.plot(pareto_df["Z1_cost"], pareto_df["Z2_worst_risk"], "o-r", label="Pareto frontier")
    plt.xlabel("Z1(x): intervention cost")
    plt.ylabel("Z2(x): worst-case accident risk")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def _plot_rank_correlation(ref_df: pd.DataFrame, alt_df: pd.DataFrame, save_path: str, title: str) -> None:
    m = ref_df[["portfolio", "Z2_worst_risk"]].merge(
        alt_df[["portfolio", "Z2_worst_risk"]], on="portfolio", suffixes=("_meanfield", "_benchmark")
    )
    plt.figure(figsize=(6, 6))
    plt.scatter(m["Z2_worst_risk_meanfield"], m["Z2_worst_risk_benchmark"], alpha=0.7)
    lims = [min(m.min(numeric_only=True)), max(m.max(numeric_only=True))]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlabel("Worst-case risk (mean-field)")
    plt.ylabel("Worst-case risk (dependence-aware)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


# ----------------------------------
# Section 6.2 Baseline portfolio study
# ----------------------------------


def run_baseline_portfolio_analysis(params: ParameterBundle, out_dir: str, epsilon_grid_points: int = 21) -> Dict[str, pd.DataFrame]:
    """Section 6.2: baseline parameterization, all 32 portfolios, epsilon frontier, Pareto set."""
    section_dir = os.path.join(out_dir, "sec_6_2_baseline")
    _ensure_dir(section_dir)

    all_df = evaluate_all_portfolios(params)
    all_df.to_csv(os.path.join(section_dir, "all_portfolios.csv"), index=False)

    eps_df = solve_epsilon_constraint_grid(
        all_df,
        epsilon_grid_points=epsilon_grid_points,
        epsilon_min=0.0,
        epsilon_max=sum(params.costs.values()),
    )
    eps_df.to_csv(os.path.join(section_dir, "epsilon_grid_solutions.csv"), index=False)

    pareto_df = compute_pareto_frontier(all_df)
    pareto_df.to_csv(os.path.join(section_dir, "pareto_frontier.csv"), index=False)

    _plot_pareto_scatter(
        all_df,
        pareto_df,
        save_path=os.path.join(section_dir, "pareto_frontier.png"),
        title="Section 6.2 Baseline Pareto Frontier",
    )

    return {"all": all_df, "eps": eps_df, "pareto": pareto_df}


# -----------------------------------------------------
# Section 6.3.1 Approximation check for terminal node N10
# -----------------------------------------------------


def run_terminal_node_approximation_check(params: ParameterBundle, out_dir: str) -> pd.DataFrame:
    section_dir = os.path.join(out_dir, "sec_6_3_1_terminal_approx")
    _ensure_dir(section_dir)

    meanfield_df = evaluate_all_portfolios(params, use_terminal_benchmark=False)
    benchmark_df = evaluate_all_portfolios(params, use_terminal_benchmark=True)

    merged = meanfield_df[["portfolio", "Z2_worst_risk"]].merge(
        benchmark_df[["portfolio", "Z2_worst_risk"]], on="portfolio", suffixes=("_meanfield", "_benchmark")
    )
    merged["abs_diff"] = (merged["Z2_worst_risk_meanfield"] - merged["Z2_worst_risk_benchmark"]).abs()
    merged["rank_meanfield"] = merged["Z2_worst_risk_meanfield"].rank(method="min")
    merged["rank_benchmark"] = merged["Z2_worst_risk_benchmark"].rank(method="min")
    merged["rank_diff"] = (merged["rank_meanfield"] - merged["rank_benchmark"]).abs()

    rho = merged[["rank_meanfield", "rank_benchmark"]].corr(method="spearman").iloc[0, 1]
    summary = pd.DataFrame(
        [{"metric": "spearman_rank_corr", "value": float(rho)},
         {"metric": "max_abs_diff", "value": float(merged["abs_diff"].max())},
         {"metric": "mean_abs_diff", "value": float(merged["abs_diff"].mean())}]
    )

    merged.to_csv(os.path.join(section_dir, "terminal_approx_comparison.csv"), index=False)
    summary.to_csv(os.path.join(section_dir, "terminal_approx_summary.csv"), index=False)

    _plot_rank_correlation(
        meanfield_df,
        benchmark_df,
        save_path=os.path.join(section_dir, "terminal_approx_scatter.png"),
        title="Section 6.3.1 Mean-field vs Dependence-aware",
    )

    return summary


# -------------------------------------------------
# Section 6.3.2 Sensitivity to Gamma hierarchy assumptions
# -------------------------------------------------


def _scaled_gamma_params(params: ParameterBundle, profile: str) -> ParameterBundle:
    p = replace(params)
    p.gamma_root = dict(params.gamma_root)
    p.gamma_child = {k: dict(v) for k, v in params.gamma_child.items()}

    if profile == "baseline_hierarchy":
        return p
    if profile == "partially_equalized":
        # moderate equalization: weaken physical a bit, strengthen cognitive a bit
        p.gamma_root["A1"] *= 0.8
        p.gamma_root["A2"] *= 0.8
        p.gamma_child["A3"] = {k: 1.2 * v for k, v in p.gamma_child["A3"].items()}
        p.gamma_child["A4"] = {k: 1.2 * v for k, v in p.gamma_child["A4"].items()}
        p.gamma_child["A5"] = {k: 1.1 * v for k, v in p.gamma_child["A5"].items()}
        return p
    if profile == "favorable_to_cognitive":
        # stronger reversal toward cognitive interventions
        p.gamma_root["A1"] *= 0.6
        p.gamma_root["A2"] *= 0.6
        p.gamma_child["A3"] = {k: 1.5 * v for k, v in p.gamma_child["A3"].items()}
        p.gamma_child["A4"] = {k: 1.6 * v for k, v in p.gamma_child["A4"].items()}
        p.gamma_child["A5"] = {k: 1.3 * v for k, v in p.gamma_child["A5"].items()}
        return p
    raise ValueError(f"Unknown hierarchy profile: {profile}")


def run_parameter_hierarchy_sensitivity(params: ParameterBundle, out_dir: str) -> pd.DataFrame:
    section_dir = os.path.join(out_dir, "sec_6_3_2_hierarchy_sensitivity")
    _ensure_dir(section_dir)

    profiles = ["baseline_hierarchy", "partially_equalized", "favorable_to_cognitive"]
    rows = []
    paretos = {}

    for prof in profiles:
        pprof = _scaled_gamma_params(params, prof)
        all_df = evaluate_all_portfolios(pprof)
        all_df["profile"] = prof
        all_df.to_csv(os.path.join(section_dir, f"all_{prof}.csv"), index=False)

        pareto = compute_pareto_frontier(all_df)
        pareto["profile"] = prof
        pareto.to_csv(os.path.join(section_dir, f"pareto_{prof}.csv"), index=False)
        paretos[prof] = pareto

        best = all_df.sort_values(["Z2_worst_risk", "Z1_cost"]).iloc[0]
        rows.append({
            "profile": prof,
            "best_portfolio": best["portfolio"],
            "best_Z2": best["Z2_worst_risk"],
            "best_Z1": best["Z1_cost"],
            "pareto_size": len(pareto),
        })

    # Overlay plot
    plt.figure(figsize=(8, 5))
    markers = {"baseline_hierarchy": "o", "partially_equalized": "s", "favorable_to_cognitive": "^"}
    for prof in profiles:
        p = paretos[prof].sort_values("Z1_cost")
        plt.plot(p["Z1_cost"], p["Z2_worst_risk"], marker=markers[prof], label=prof)
    plt.xlabel("Z1(x)")
    plt.ylabel("Z2(x)")
    plt.title("Section 6.3.2 Pareto Frontiers under Hierarchy Assumptions")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(section_dir, "hierarchy_frontier_overlay.png"), dpi=180)
    plt.close()

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(section_dir, "hierarchy_summary.csv"), index=False)
    return summary


# -------------------------------
# Section 6.4 Sensitivity analyses
# -------------------------------


def run_efficacy_decay_sensitivity(params: ParameterBundle, out_dir: str) -> pd.DataFrame:
    section_dir = os.path.join(out_dir, "sec_6_4_efficacy_decay")
    _ensure_dir(section_dir)

    decay_levels = [1.0, 0.8, 0.6]
    rows = []
    for d in decay_levels:
        p = replace(params)
        p.gamma_root = {k: d * v for k, v in params.gamma_root.items()}
        p.gamma_child = {ak: {spa: d * gv for spa, gv in gmap.items()} for ak, gmap in params.gamma_child.items()}

        all_df = evaluate_all_portfolios(p)
        pareto = compute_pareto_frontier(all_df)
        all_df.to_csv(os.path.join(section_dir, f"all_decay_{d:.2f}.csv"), index=False)
        pareto.to_csv(os.path.join(section_dir, f"pareto_decay_{d:.2f}.csv"), index=False)
        best = all_df.sort_values(["Z2_worst_risk", "Z1_cost"]).iloc[0]
        rows.append({"decay": d, "best_portfolio": best["portfolio"], "best_Z2": best["Z2_worst_risk"], "pareto_size": len(pareto)})

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(section_dir, "efficacy_decay_summary.csv"), index=False)
    return summary


def run_conservatism_sensitivity(params: ParameterBundle, out_dir: str) -> pd.DataFrame:
    section_dir = os.path.join(out_dir, "sec_6_4_conservatism")
    _ensure_dir(section_dir)

    levels = [0.5, 1.0, 1.5]
    rows = []
    for c in levels:
        p = replace(params)
        p.delta_H_root = {n: c * d for n, d in params.delta_H_root.items()}
        p.delta_H_child = {n: {spa: c * d for spa, d in dm.items()} for n, dm in params.delta_H_child.items()}

        all_df = evaluate_all_portfolios(p)
        pareto = compute_pareto_frontier(all_df)
        all_df.to_csv(os.path.join(section_dir, f"all_cons_{c:.2f}.csv"), index=False)
        pareto.to_csv(os.path.join(section_dir, f"pareto_cons_{c:.2f}.csv"), index=False)
        best = all_df.sort_values(["Z2_worst_risk", "Z1_cost"]).iloc[0]
        rows.append({"conservatism_multiplier": c, "best_portfolio": best["portfolio"], "best_Z2": best["Z2_worst_risk"], "pareto_size": len(pareto)})

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(section_dir, "conservatism_summary.csv"), index=False)
    return summary


def run_stress_multiplier_sensitivity(params: ParameterBundle, out_dir: str) -> pd.DataFrame:
    section_dir = os.path.join(out_dir, "sec_6_4_stress_multiplier")
    _ensure_dir(section_dir)

    levels = [0.8, 1.0, 1.2]
    rows = []
    for m in levels:
        p = replace(params)
        p.theta_root = {om: {n: dict(v) for n, v in nd.items()} for om, nd in params.theta_root.items()}
        p.theta_child = {
            om: {n: {spa: dict(sd) for spa, sd in nd.items()} for n, nd in omd.items()}
            for om, omd in params.theta_child.items()
        }

        # multiply non-zero shock terms only
        for om in SCENARIOS:
            for n in ROOT_NODES:
                for s in ("H", "M"):
                    if abs(p.theta_root[om][n][s]) > 0:
                        p.theta_root[om][n][s] *= m
            for n in CHILD_NODES:
                for spa in p.theta_child[om][n]:
                    for s in ("H", "M"):
                        if abs(p.theta_child[om][n][spa][s]) > 0:
                            p.theta_child[om][n][spa][s] *= m

        all_df = evaluate_all_portfolios(p)
        pareto = compute_pareto_frontier(all_df)
        all_df.to_csv(os.path.join(section_dir, f"all_stress_{m:.2f}.csv"), index=False)
        pareto.to_csv(os.path.join(section_dir, f"pareto_stress_{m:.2f}.csv"), index=False)
        best = all_df.sort_values(["Z2_worst_risk", "Z1_cost"]).iloc[0]
        rows.append({"stress_multiplier": m, "best_portfolio": best["portfolio"], "best_Z2": best["Z2_worst_risk"], "pareto_size": len(pareto)})

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(section_dir, "stress_multiplier_summary.csv"), index=False)
    return summary


# ----
# Main
# ----


def main(output_dir: str = "results_paper") -> None:
    """Run all experiments in one executable workflow."""
    _ensure_dir(output_dir)
    cfg = ModelConfig()
    params = generate_baseline_parameters(cfg)

    # Section 6.2
    baseline_outputs = run_baseline_portfolio_analysis(params, out_dir=output_dir, epsilon_grid_points=cfg.epsilon_grid_points)

    # Section 6.3
    approx_summary = run_terminal_node_approximation_check(params, out_dir=output_dir)
    hierarchy_summary = run_parameter_hierarchy_sensitivity(params, out_dir=output_dir)

    # Section 6.4
    efficacy_summary = run_efficacy_decay_sensitivity(params, out_dir=output_dir)
    conserv_summary = run_conservatism_sensitivity(params, out_dir=output_dir)
    stress_summary = run_stress_multiplier_sensitivity(params, out_dir=output_dir)

    # consolidated overview
    overview = {
        "baseline_pareto_size": [len(baseline_outputs["pareto"])],
        "approx_spearman": [float(approx_summary.loc[approx_summary["metric"] == "spearman_rank_corr", "value"].iloc[0])],
        "hierarchy_profiles": [len(hierarchy_summary)],
        "efficacy_cases": [len(efficacy_summary)],
        "conservatism_cases": [len(conserv_summary)],
        "stress_cases": [len(stress_summary)],
    }
    pd.DataFrame(overview).to_csv(os.path.join(output_dir, "experiment_overview.csv"), index=False)

    print("Completed all experiments.")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
