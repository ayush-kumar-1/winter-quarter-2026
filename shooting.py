from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import jit, lax
from scipy import optimize

# =============================================================================
# 1. Visualization Setup
# =============================================================================


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "legend.frameon": False,
    }
)

# =============================================================================
# 2. Model Configuration & Functions
# =============================================================================


class ModelParams(NamedTuple):
    A: float = 1.0
    alpha: float = 0.3
    sigma: float = 2.0
    rho: float = 0.05
    delta: float = 0.05
    dt: float = 0.1
    N: int = 700


params = ModelParams()


@jit
def production(k: float, p: ModelParams) -> float:
    return p.A * (k**p.alpha)


@jit
def derivatives(state: Tuple[float, float], t: int, G: float, p: ModelParams):
    k, c = state
    k = jnp.maximum(k, 1e-6)
    mpk = p.alpha * p.A * (k ** (p.alpha - 1.0))
    k_dot = production(k, p) - c - G - p.delta * k
    c_dot = (1.0 / p.sigma) * (mpk - p.delta - p.rho) * c
    return k_dot, c_dot


@jit
def get_steady_state(G_level: float, p: ModelParams):
    rhs = (p.delta + p.rho) / (p.alpha * p.A)
    k_ss = rhs ** (1.0 / (p.alpha - 1.0))
    c_ss = production(k_ss, p) - G_level - p.delta * k_ss
    return k_ss, c_ss


@jit
def euler_step(state, G_curr, p: ModelParams):
    k_dot, c_dot = derivatives(state, 0, G_curr, p)
    k_next = state[0] + p.dt * k_dot
    c_next = state[1] + p.dt * c_dot
    return (k_next, c_next)


def simulate_trajectory(k0, c0, G_series, p: ModelParams):
    def scan_body(carry, G_val):
        state = carry
        next_state = euler_step(state, G_val, p)
        return next_state, state

    final_state, trajectory = lax.scan(scan_body, (k0, c0), G_series)
    return trajectory  # (ks, cs)


# =============================================================================
# 3. Shooting Algorithm
# =============================================================================


def solve_shooting(k0, G_series, p, target_k_ss):
    """Finds c0 to hit target_k_ss at T=N."""

    def objective(c0_guess):
        ks, _ = simulate_trajectory(k0, c0_guess, G_series, p)
        return ks[-1] - target_k_ss

    # Establish bounds
    bracket_max = production(k0, p) * 2.0
    try:
        res = optimize.root_scalar(
            objective, bracket=[0.01, bracket_max], method="brentq", xtol=1e-5
        )
        c0_opt = res.root
    except:
        c0_opt = get_steady_state(G_series[0], p)[1]

    k_path, c_path = simulate_trajectory(k0, c0_opt, G_series, p)
    return k_path, c_path


# =============================================================================
# 4. Plotting Helpers
# =============================================================================


def plot_nullclines(ax, p, k_max, G_val, label_suffix="", color="gray", style="--"):
    k_range = np.linspace(0.1, k_max, 300)
    c_k_null = p.A * (k_range**p.alpha) - G_val - p.delta * k_range
    valid_idx = c_k_null >= 0

    label_k = rf"$\dot{{k}}=0$ {label_suffix}" if label_suffix else r"$\dot{k}=0$"
    ax.plot(
        k_range[valid_idx],
        c_k_null[valid_idx],
        color=color,
        linestyle=style,
        alpha=0.6,
        label=label_k,
    )

    k_c_null_val, _ = get_steady_state(G_val, p)
    return k_c_null_val


def add_arrow(ax, x, y, dx, dy, color="blue"):
    ax.arrow(
        x,
        y,
        dx,
        dy,
        shape="full",
        lw=0,
        length_includes_head=True,
        head_width=0.04,
        head_length=0.08,
        color=color,
        zorder=10,
    )


def clean_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =============================================================================
# 5. Execution
# =============================================================================


def run_analysis():
    # --- Data Generation ---
    k_ss, c_ss = get_steady_state(0.0, params)
    G_zeros = jnp.zeros(params.N)

    # Case 1: Compute Saddle Path from LEFT and RIGHT
    # Left arm: Start from k=1.0 -> k_ss
    k_path_L, c_path_L = solve_shooting(1.0, G_zeros, params, k_ss)
    # Right arm: Start from k=7.0 (approx) -> k_ss
    k_path_R, c_path_R = solve_shooting(k_ss * 1.5, G_zeros, params, k_ss)

    # Case 2: Permanent G
    G_new = 0.2 * c_ss
    k_ss_new, c_ss_new = get_steady_state(G_new, params)
    G_perm = jnp.full(params.N, G_new)

    # --- FIX START ---
    # Previously caused ValueError. We just need the path.
    k_path_2, c_path_2 = solve_shooting(k_ss, G_perm, params, k_ss_new)
    # The jump point is implicitly the first element of the path
    # --- FIX END ---

    # Case 3: Temporary G
    G_temp_list = [0.0]
    curr = G_new
    G_temp_list.append(curr)
    for _ in range(params.N - 2):
        curr = 0.95 * curr
        G_temp_list.append(curr)
    G_temp = jnp.array(G_temp_list)
    k_path_3, c_path_3 = solve_shooting(k_ss, G_temp, params, k_ss)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    k_lim = k_ss * 1.6
    c_lim = c_ss * 1.6

    # -----------------------
    # Plot Case 1: Baseline
    # -----------------------
    ax = axes[0]
    ax.set_title(r"Case 1: Baseline ($G=0$)")
    k_c_loc = plot_nullclines(ax, params, k_lim, 0.0, color="green", style="--")
    ax.axvline(k_c_loc, color="red", linestyle=":", alpha=0.6, label=r"$\dot{c}=0$")

    # Plot both arms of the Saddle Path
    ax.plot(k_path_L, c_path_L, color="blue", linewidth=2, label="Saddle Path")
    ax.plot(k_path_R, c_path_R, color="blue", linewidth=2)

    # Add arrows to Case 1
    # Arrow on left arm
    idx_L = len(k_path_L) // 2
    add_arrow(
        ax,
        k_path_L[idx_L],
        c_path_L[idx_L],
        k_path_L[idx_L + 1] - k_path_L[idx_L],
        c_path_L[idx_L + 1] - c_path_L[idx_L],
    )
    # Arrow on right arm (flowing left)
    idx_R = len(k_path_R) // 4
    add_arrow(
        ax,
        k_path_R[idx_R],
        c_path_R[idx_R],
        k_path_R[idx_R + 1] - k_path_R[idx_R],
        c_path_R[idx_R + 1] - c_path_R[idx_R],
    )

    ax.plot(k_ss, c_ss, "ko", zorder=10, label="Steady State")

    clean_spines(ax)
    ax.set_xlim(0, k_lim)
    ax.set_ylim(0, c_lim)
    ax.set_xlabel(r"Capital $k$")
    ax.set_ylabel(r"Consumption $c$")
    ax.legend(loc="lower right")

    # -----------------------
    # Plot Case 2: Permanent
    # -----------------------
    ax = axes[1]
    ax.set_title(r"Case 2: Permanent Increase ($G=\bar{G}$)")

    # Nullclines
    plot_nullclines(
        ax, params, k_lim, 0.0, label_suffix="(Old)", color="gray", style=":"
    )
    k_c_loc_new = plot_nullclines(
        ax, params, k_lim, G_new, label_suffix="(New)", color="green", style="--"
    )
    ax.axvline(k_c_loc_new, color="red", linestyle=":", alpha=0.6, label=r"$\dot{c}=0$")

    # Start Point: (Old k_ss, New calculated c0)
    # Using the first point of the simulation path
    ax.scatter(
        [k_path_2[0]],
        [c_path_2[0]],
        color="gray",
        s=50,
        zorder=10,
        label=r"Start (Old SS $k$)",
    )

    # End Point: New SS
    ax.scatter([k_ss_new], [c_ss_new], color="red", s=50, zorder=10, label="New SS")

    clean_spines(ax)
    ax.set_xlim(0, k_lim)
    ax.set_ylim(0, c_lim)
    ax.set_xlabel(r"Capital $k$")
    ax.legend(loc="lower right")

    # -----------------------
    # Plot Case 3: Temporary
    # -----------------------
    ax = axes[2]
    ax.set_title(r"Case 3: Temporary Shock ($G_t \to 0$)")

    plot_nullclines(
        ax, params, k_lim, 0.0, label_suffix="(Target)", color="green", style="--"
    )
    plot_nullclines(
        ax,
        params,
        k_lim,
        G_new,
        label_suffix="(Initial Shock)",
        color="gray",
        style=":",
    )
    ax.axvline(k_ss, color="red", linestyle=":", alpha=0.6, label=r"$\dot{c}=0$")

    ax.plot(k_path_3, c_path_3, color="purple", linewidth=2, label="Trajectory")

    ax.scatter([k_path_3[0]], [c_path_3[0]], color="gray", zorder=10, label="Start")
    ax.scatter(
        [k_path_3[-1]],
        [c_path_3[-1]],
        marker="x",
        color="black",
        zorder=10,
        label="End (Approx)",
    )

    clean_spines(ax)
    ax.set_xlim(0, k_lim)
    ax.set_ylim(0, c_lim)
    ax.set_xlabel(r"Capital $k$")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("plots/phase_diagram.png")


if __name__ == "__main__":
    run_analysis()
