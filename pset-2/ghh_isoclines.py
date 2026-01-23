import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "pset-1"))

from shooting import ModelParams  # noqa: E402

p = ModelParams()


def steady_state(p: ModelParams, beta: float):
    alpha = p.alpha
    delta = p.delta
    r_ss = 1.0 / beta - 1.0 + delta
    k_ss = (
        alpha * (1.0 - alpha) ** ((1.0 - alpha) / (1.0 + alpha)) / r_ss
    ) ** ((1.0 + alpha) / (1.0 - alpha))
    n_ss = (1.0 - alpha) ** (1.0 / (1.0 + alpha)) * k_ss ** (
        alpha / (1.0 + alpha)
    )
    y_ss = k_ss**alpha * n_ss ** (1.0 - alpha)
    c_ss = y_ss - delta * k_ss
    x_ss = c_ss - 0.5 * n_ss**2
    if x_ss <= 0:
        raise ValueError("Steady state C - N^2/2 must be positive.")
    return {
        "k_ss": k_ss,
        "n_ss": n_ss,
        "y_ss": y_ss,
        "c_ss": c_ss,
        "x_ss": x_ss,
        "r_ss": r_ss,
    }


def compute_coeffs(p: ModelParams, beta: float):
    alpha = p.alpha
    delta = p.delta
    ss = steady_state(p, beta)

    k_ss = ss["k_ss"]
    n_ss = ss["n_ss"]
    y_ss = ss["y_ss"]
    c_ss = ss["c_ss"]
    x_ss = ss["x_ss"]
    r_ss = ss["r_ss"]

    eta_a = 1.0 / (1.0 + alpha)
    eta_k = alpha / (1.0 + alpha)

    y_a = 2.0 / (1.0 + alpha)
    y_k = 2.0 * alpha / (1.0 + alpha)

    r_a = y_a
    r_k = -(1.0 - alpha) / (1.0 + alpha)

    s_y = y_ss / k_ss
    s_c = c_ss / k_ss

    phi_kk = (1.0 - delta) + s_y * y_k
    phi_kc = -s_c
    phi_ka = s_y * y_a

    chi_c = c_ss / x_ss
    chi_n = -(n_ss**2) / x_ss

    r_total = r_ss + 1.0 - delta
    phi_r = r_ss / r_total

    return {
        "ss": ss,
        "eta_a": eta_a,
        "eta_k": eta_k,
        "y_a": y_a,
        "y_k": y_k,
        "r_a": r_a,
        "r_k": r_k,
        "s_y": s_y,
        "s_c": s_c,
        "phi_kk": phi_kk,
        "phi_kc": phi_kc,
        "phi_ka": phi_ka,
        "chi_c": chi_c,
        "chi_n": chi_n,
        "phi_r": phi_r,
    }


def isoclines(k_hat, a_hat, rho_a, coeffs):
    s_c = coeffs["s_c"]
    phi_kk = coeffs["phi_kk"]
    phi_kc = coeffs["phi_kc"]
    phi_ka = coeffs["phi_ka"]

    eta_a = coeffs["eta_a"]
    eta_k = coeffs["eta_k"]

    r_a = coeffs["r_a"]
    r_k = coeffs["r_k"]

    chi_n = coeffs["chi_n"]
    phi_r = coeffs["phi_r"]

    # Capital isocline: k_hat_t = k_hat_{t-1}
    c_hat_k = ((phi_kk - 1.0) / s_c) * k_hat + (phi_ka / s_c) * a_hat

    # Euler isocline: c_hat_t = E_t c_hat_{t+1}
    denom = (chi_n * eta_k + phi_r * r_k) * phi_kc
    num_k = (
        chi_n * eta_k - (chi_n * eta_k + phi_r * r_k) * phi_kk
    ) * k_hat
    num_a = (
        chi_n * eta_a * (1.0 - rho_a)
        + phi_r * r_a * rho_a
        - (chi_n * eta_k + phi_r * r_k) * phi_ka
    ) * a_hat
    c_hat_e = (num_k + num_a) / denom

    return c_hat_e, c_hat_k


def plot_isoclines(k_grid, a_hat, rho_a, coeffs, title, ax):
    c_e_base, c_k_base = isoclines(k_grid, 0.0, rho_a, coeffs)
    c_e_shock, c_k_shock = isoclines(k_grid, a_hat, rho_a, coeffs)

    ax.plot(k_grid, c_e_base, color="black", linestyle="--", label="Euler (A=1)")
    ax.plot(k_grid, c_k_base, color="gray", linestyle="--", label="Capital (A=1)")

    ax.plot(k_grid, c_e_shock, color="tab:blue", label="Euler (shock)")
    ax.plot(k_grid, c_k_shock, color="tab:orange", label="Capital (shock)")

    ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.5)
    ax.axvline(0.0, color="gray", linewidth=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(r"$\hat{k}_{t-1}$")
    ax.set_ylabel(r"$\hat{c}_t$")
    ax.legend(loc="best")


def run():
    beta = 1.0 / (1.0 + p.rho)
    coeffs = compute_coeffs(p, beta)

    k_grid = np.linspace(-0.2, 0.2, 300)
    a_hat = 0.02

    rho_a_temp = 0.9
    rho_a_perm = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_isoclines(
        k_grid,
        a_hat,
        rho_a_temp,
        coeffs,
        title=r"Temporary TFP: $\hat{a}_{t+1}=\rho\hat{a}_t$",
        ax=axes[0],
    )
    plot_isoclines(
        k_grid,
        a_hat,
        rho_a_perm,
        coeffs,
        title=r"Permanent TFP: $\hat{a}_{t+1}=\hat{a}_t$",
        ax=axes[1],
    )

    fig.tight_layout()
    out_path = ROOT / "pset-2" / "plots" / "ghh_isoclines.png"
    fig.savefig(out_path, dpi=200)


if __name__ == "__main__":
    run()
