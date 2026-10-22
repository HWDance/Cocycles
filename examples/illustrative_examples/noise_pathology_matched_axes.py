"""Create a noise-pathology figure whose three panels share an x range.

Unlike the original figure, the first panel evaluates the density along the
quantile map q(u). Its troughs therefore use the same Gaussian coordinate u
as the fixed-base structural map and can be compared directly with f_1'(u).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import norm, t


OUTPUT = Path(__file__).with_name("illustrative_noise_pathology_matched_axes.pdf")
BETA, A, SCALE, DF = 1.0, 4.0, 0.65, 1.0
EPS, OMEGA, U_MAX = 0.08, 5.0, 3.5
NORMALIZER = EPS + 0.5 * (1.0 - np.exp(-2.0 * OMEGA**2))


def baseline_density(z):
    z = np.asarray(z, dtype=float)
    return (
        0.5 * t.pdf((z - A) / SCALE, df=DF) / SCALE
        + 0.5 * t.pdf((z + A) / SCALE, df=DF) / SCALE
    )


def baseline_cdf(z):
    z = np.asarray(z, dtype=float)
    return (
        0.5 * t.cdf((z - A) / SCALE, df=DF)
        + 0.5 * t.cdf((z + A) / SCALE, df=DF)
    )


def density(z):
    probability = np.clip(
        baseline_cdf(z),
        np.nextafter(0.0, 1.0),
        np.nextafter(1.0, 0.0),
    )
    gaussian_coordinate = norm.ppf(probability)
    oscillation = EPS + np.sin(OMEGA * gaussian_coordinate) ** 2
    return baseline_density(z) * oscillation / NORMALIZER


def quantile_map():
    u_positive = np.linspace(0.0, U_MAX, 15_001)

    def ode(u_value, q_value):
        return [norm.pdf(u_value) / max(float(density(q_value[0])), 1e-300)]

    solution = solve_ivp(
        ode,
        (0.0, U_MAX),
        [0.0],
        t_eval=u_positive,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        max_step=0.002,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    q_positive = solution.y[0]
    u = np.concatenate((-u_positive[:0:-1], u_positive))
    q = np.concatenate((-q_positive[:0:-1], q_positive))
    return u, q


def make_figure():
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "legend.fontsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )
    u, q = quantile_map()
    density_on_q = density(q)
    structural_map = BETA + q
    structural_derivative = norm.pdf(u) / density_on_q
    valid = (
        np.isfinite(structural_map)
        & np.isfinite(structural_derivative)
        & (density_on_q > 1e-14)
        & (structural_derivative > 0.0)
    )

    y = np.linspace(-U_MAX, U_MAX, 1000)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), sharex=True)
    gray, alpha, width = "0.55", 0.55, 2.25

    axes[0].plot(u[valid], density_on_q[valid], color="black", lw=width)
    axes[0].set_title(r"Noise density along $q$")
    axes[0].set_xlabel(r"$u\sim N(0,1)$")
    axes[0].set_ylabel(r"$p_\xi(q(u))$")

    derivative_axis = axes[1].twinx()
    map_line = axes[1].plot(
        u[valid], structural_map[valid], color=gray, alpha=alpha,
        lw=width, label=r"$f_1$",
    )[0]
    derivative_line = derivative_axis.plot(
        u[valid], structural_derivative[valid], color="tab:red",
        lw=width, ls="--", label=r"$f_1'$",
    )[0]
    axes[1].set_title("Fixed-base SCM")
    axes[1].set_xlabel(r"$u\sim N(0,1)$")
    axes[1].set_ylabel(r"$f_1$")
    axes[1].set_ylim(-20, 20)
    derivative_axis.set_ylabel(r"$f_1'$")
    derivative_axis.set_yscale("log")
    axes[1].legend(
        handles=[map_line, derivative_line], frameon=False, loc="upper left"
    )

    counterfactual_derivative_axis = axes[2].twinx()
    counterfactual_line = axes[2].plot(
        y, y + BETA, color=gray, alpha=alpha, lw=width, label=r"$T_{1,0}$"
    )[0]
    counterfactual_derivative_line = counterfactual_derivative_axis.plot(
        y, np.ones_like(y), color="tab:blue", lw=width, ls="--",
        label=r"$T_{1,0}'$",
    )[0]
    axes[2].set_title("Counterfactual transport")
    axes[2].set_xlabel(r"$Y(0)$")
    axes[2].set_ylabel(r"$T_{1,0}$")
    counterfactual_derivative_axis.set_ylabel(r"$T_{1,0}'$")
    counterfactual_derivative_axis.set_ylim(0.8, 1.2)
    axes[2].legend(
        handles=[counterfactual_line, counterfactual_derivative_line],
        frameon=False, loc="upper left",
    )

    for axis in axes:
        axis.set_xlim(-U_MAX, U_MAX)
        axis.set_xticks(np.arange(-3.0, 3.1, 1.0))
    fig.tight_layout(w_pad=1.25)
    return fig


if __name__ == "__main__":
    figure = make_figure()
    figure.savefig(OUTPUT, bbox_inches="tight")
    print(OUTPUT)
