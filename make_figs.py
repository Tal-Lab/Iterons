"""Figures for 'Titration by a co-replicating reservoir'.

Reads only saved tables; no solver calls, so re-rendering is fast.
  fig1: decoupling_check.csv, fig1c_collapse.csv
  fig2: no_autorep_control.csv, handcuff_convergence.csv, fig2c_pmf.csv
  fig3: loss_rate_sweep.csv, optimum_invariant.csv, optimum_measured_kd.csv
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import os

# Resolve paths against this file, so the script runs from any working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")
_FIGS = os.path.join(_HERE, "figures")
os.makedirs(_FIGS, exist_ok=True)
# Journal of Theoretical Biology asks for at least 300 dpi and, for a full-page
# figure, at least 2244 px of width. At these figure widths 300 dpi falls short, so
# the raster density is raised; the drawing itself is unchanged.
_DPI = int(os.environ.get("FIG_DPI", "400"))
# These are line drawings, so each figure is written twice: a vector PDF, which is the
# form used in the paper, and a raster PNG at _DPI for viewers that cannot display
# vector art. The drawing code is identical for both. fonttype 42 embeds TrueType outlines rather than Type 3, which is what
# "embed the font" means in the journal's artwork table.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
_read_csv = pd.read_csv


def _csv(name):
    """Read a result table from data/ whether given a bare name or a path."""
    return _read_csv(name if os.path.isabs(name) or os.path.dirname(name)
                     else os.path.join(_DATA, name))


pd.read_csv = _csv

# apply_figure_style / panel_letter / META_GREY come from the figure-style skill's
# kernel plugin. When this file is run standalone they are absent, so fall back to
# plain-but-equivalent definitions so the repository reproduces the figures offline.
try:
    apply_figure_style, panel_letter, META_GREY
except NameError:
    META_GREY = "0.45"

    def apply_figure_style(sizes=(8, 7, 6)):
        b, m, s_ = sizes
        mpl.rcParams.update({
            "figure.dpi": 150, "savefig.dpi": 300,
            "font.size": b, "axes.titlesize": b, "axes.labelsize": b,
            "legend.fontsize": m, "xtick.labelsize": s_, "ytick.labelsize": s_,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
            "legend.frameon": True, "figure.facecolor": "white"})

    def panel_letter(ax, letter, dx=-0.13, dy=1.06):
        ax.text(dx, dy, letter, transform=ax.transAxes, fontweight="bold",
                fontsize=10, ha="right", va="top")

ACC = "#b2182b"        # focal: co-scaling architecture
ACC2 = "#2166ac"       # comparator: fixed reservoir
KDC = {3.0: "#08306b", 10.0: "#2171b5", 30.0: "#6baed6", 100.0: "#c6dbef"}


def opaque_legend(ax, **kw):
    kw.setdefault("framealpha", 1.0)
    kw.setdefault("fancybox", False)
    leg = ax.legend(**kw)
    if leg:
        leg.get_frame().set_edgecolor("0.75")
        leg.get_frame().set_linewidth(0.6)
        leg.set_zorder(20)
    return leg


# ---------------------------------------------------------------- figure 1
def fig1():
    apply_figure_style(sizes=(8, 7, 6))
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(7.2, 2.6))
    fig.subplots_adjust(wspace=0.40)

    # (a) the decoupling itself
    dc = pd.read_csv("decoupling_check.csv")
    for m_, col, lab in [(0, "0.40", "no sites, $m=0$"),
                         (5, ACC2, "$m=5$"), (14, ACC, "$m=14$")]:
        g = dc[dc.m == m_].sort_values("n")
        ax_a.plot(g.n, g.x_f, "-o", ms=3.2, lw=1.5, color=col, label=lab, zorder=4)
    ax_a.set(xscale="log", yscale="log", xlabel="copy number $n$",
             ylabel="free initiator $\\bar x_f$")
    ax_a.set_title("Sites decouple the free pool", loc="left")
    opaque_legend(ax_a, loc="upper left", handlelength=1.6)

    # (b) same sites, same affinity, opposite sign
    sr = pd.read_csv("fig1b_signreversal.csv")
    ax_b.axhline(1.0, ls="--", lw=1.0, color=META_GREY, zorder=2,
                 label="fluctuation passed through")
    for kd_, col in zip(sorted(sr.kd.unique()), (KDC[3.0], KDC[10.0], KDC[30.0])):
        g = sr[sr.kd == kd_].sort_values("m")
        ax_b.plot(g.m, g.g_fix, "--^", ms=3.0, lw=1.3, color=col, zorder=4)
        ax_b.plot(g.m, g.g_co, "-o", ms=3.0, lw=1.5, color=col, zorder=5,
                  label=None)
    ax_b.plot([], [], "--^", ms=3.0, lw=1.3, color="0.3", label="sites supplied separately")
    ax_b.plot([], [], "-o", ms=3.0, lw=1.5, color="0.3", label="sites on the replicon")
    ax_b.set(xscale="log", yscale="log", xlabel="iterons per plasmid $m$",
             ylabel="gain $\\mathrm{d}\\ln\\bar x_f/\\mathrm{d}\\ln n$")
    ax_b.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2])
    ax_b.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2"])
    ax_b.set_title("The architecture sets the sign", loc="left")
    opaque_legend(ax_b, loc="lower left", handlelength=1.6, labelspacing=0.25)

    # (c) collapse onto the capacity ratio
    f1c = pd.read_csv("fig1c_collapse.csv")
    xx = np.logspace(-2.2, 2.4, 300)
    ax_c.plot(xx, 1 / (1 + xx), "-", lw=2.2, color="0.55", zorder=2,
              label="$1/(1+\\beta\\chi/\\bar x_f)$")
    for kd_ in sorted(f1c.kd.unique()):
        g = f1c[f1c.kd == kd_].sort_values("cap_ratio")
        ax_c.plot(g.cap_ratio, g.g_co, "o", ms=3.2, color=KDC[kd_], zorder=4,
                  label=f"$k_d={kd_:g}$")
    ax_c.set(xscale="log", yscale="log",
             xlabel="capacity ratio $\\beta\\chi/\\bar x_f$",
             ylabel="gain $\\mathrm{d}\\ln\\bar x_f/\\mathrm{d}\\ln n$")
    ax_c.set_title("One ratio sets the magnitude", loc="left")
    opaque_legend(ax_c, loc="lower left", handlelength=1.2, labelspacing=0.22)

    for ax, L in zip((ax_a, ax_b, ax_c), "abc"):
        panel_letter(ax, L)
    fig.canvas.draw()
    fig.savefig(os.path.join(_FIGS, "fig1_decoupling.png"), dpi=_DPI, bbox_inches="tight")
    fig.savefig(os.path.join(_FIGS, "fig1_decoupling.pdf"), bbox_inches="tight")
    return fig


# ---------------------------------------------------------------- figure 2
def fig2():
    apply_figure_style(sizes=(8, 7, 6))
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0))
    (ax_a, ax_b), (ax_c, ax_d) = axs
    fig.subplots_adjust(wspace=0.34, hspace=0.50)

    # (a) no stationary state without autorepression: the mean tracks the ceiling
    na = pd.read_csv("no_autorep_control.csv")
    ax_a.plot(na.nmax, na.En / na.nmax, "-o", ms=4.0, lw=1.6, color=ACC, zorder=5,
              label="no autorepression")
    ax_a.axhline(1.0, ls="--", lw=1.0, color=META_GREY, zorder=2,
                 label="all mass at the ceiling")
    ax_a.set(xlabel="state-space ceiling $n_{\\max}$",
             ylabel="mean copy number / ceiling", ylim=(0.0, 1.15))
    ax_a.set_title("Without feedback, no stationary state", loc="left")
    opaque_legend(ax_a, loc="lower left", handlelength=1.8)

    # (b) variance converges only once handcuffing is present
    bc = pd.read_csv("boundary_check.csv")
    g0 = bc[bc.m == 14].sort_values("nmax")
    hc = pd.read_csv("handcuff_convergence.csv").sort_values("nmax")
    ax_b.plot(g0.nmax, g0.CV2n, "-o", ms=3.6, lw=1.6, color=ACC, zorder=5,
              label="autorepression only")
    ax_b.plot(hc.nmax, hc.CV2n, "-s", ms=3.6, lw=1.6, color=ACC2, zorder=5,
              label="plus handcuffing")
    ax_b.set(xlabel="state-space ceiling $n_{\\max}$", yscale="log",
             ylabel="$\\mathrm{CV}^2$ of copy number")
    ax_b.set_title("Variance converges only with handcuffing", loc="left")
    opaque_legend(ax_b, loc="center right", handlelength=1.6)

    # (c) the distribution narrows with array size, at matched mean
    pm = pd.read_csv("fig2c_pmf.csv")
    for m_, col in [(1, "0.40"), (5, ACC2), (20, ACC)]:
        g = pm[pm.m == m_].sort_values("n")
        ax_c.plot(g.n, g.p, "-o", ms=3.0, lw=1.3, color=col, zorder=4, label=f"$m={m_}$")
    ax_c.set(xlabel="copy number $n$", ylabel="stationary probability", xlim=(0, 15))
    ax_c.set_title("Larger arrays narrow the distribution", loc="left")
    opaque_legend(ax_c, loc="upper right", handlelength=1.6, title="mean held at 5")

    # (d) consequence: variance and random-segregation loss, normalised at m = 1
    ls = pd.read_csv("loss_rate_sweep.csv").sort_values("m")
    ax_d.plot(ls.m, ls.CV2n / ls.CV2n.iloc[0], "-o", ms=3.6, lw=1.6, color=ACC2,
              zorder=5, label="$\\mathrm{CV}^2$ of copy number")
    ax_d.plot(ls.m, ls.L / ls.L.iloc[0], "-^", ms=3.6, lw=1.6, color=ACC,
              zorder=5, label="segregational loss $\\langle L\\rangle$")
    ax_d.axhline(1.0, ls=":", lw=0.9, color=META_GREY, zorder=2, label="value at $m=1$")
    ax_d.set(xscale="log", xlabel="iterons per plasmid $m$",
             ylabel="relative to $m=1$", ylim=(0.60, 1.06))
    ax_d.set_title("Loss falls, but less than variance", loc="left")
    opaque_legend(ax_d, loc="lower left", handlelength=1.6)

    for ax, L in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        panel_letter(ax, L)
    fig.canvas.draw()
    fig.savefig(os.path.join(_FIGS, "fig2_exact.png"), dpi=_DPI, bbox_inches="tight")
    fig.savefig(os.path.join(_FIGS, "fig2_exact.pdf"), bbox_inches="tight")
    return fig


# ---------------------------------------------------------------- figure 3
def fig3():
    apply_figure_style(sizes=(8, 7, 6))
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0))
    (ax_a, ax_b), (ax_c, ax_d) = axs
    fig.subplots_adjust(wspace=0.36, hspace=0.50)

    ls = pd.read_csv("loss_rate_sweep.csv").sort_values("m")
    j = ls.CV2xf.idxmin()
    ax_a.plot(ls.m, ls.CV2xf, "-o", ms=3.6, lw=1.6, color=ACC, zorder=4,
              label="free initiator")
    ax_a.plot([ls.m[j]], [ls.CV2xf[j]], "o", ms=8, mfc="none", mec="k", mew=1.3,
              zorder=6, label=f"optimum, $m={int(ls.m[j])}$")
    ax_a.set(xscale="log", xlabel="iterons per plasmid $m$",
             ylabel="$\\mathrm{CV}^2$ of free initiator")
    ax_a.set_title("Initiator noise has an optimum", loc="left")
    opaque_legend(ax_a, loc="upper left", handlelength=1.6)

    oi = pd.read_csv("optimum_invariant.csv")
    h2 = oi[oi.nH == 2].sort_values("kd")
    sl = np.polyfit(np.log(h2.kd), np.log(h2.m), 1)
    r2 = np.corrcoef(np.log(h2.kd), np.log(h2.m))[0, 1] ** 2
    kk = np.logspace(np.log10(h2.kd.min()), np.log10(h2.kd.max()), 50)
    ax_b.plot(kk, np.exp(sl[1]) * kk ** sl[0], "--", lw=1.3, color=META_GREY, zorder=3,
              label=f"effective slope {sl[0]:.2f} ($R^2={r2:.2f}$)")
    ax_b.plot(h2.kd, h2.m, "o", ms=4.2, color=ACC2, zorder=5,
              label="optimal array size $m^*$")
    ax_b.set(xscale="log", yscale="log", xlabel="dissociation constant $k_d$",
             ylabel="optimal array size $m^*$")
    ax_b.set_title("Its location crosses over with affinity", loc="left")
    opaque_legend(ax_b, loc="upper left", handlelength=1.6)

    # (c) what is conserved at the optimum, and what is not
    QS = [("site number $m^*$", h2.m, ACC2), ("capacity $\\chi$", h2.chi, "0.55"),
          ("bound initiator $x_b$", h2.x_b, ACC), ("free initiator $\\bar x_f$", h2.Exf, "#1b7837")]
    for i, (lab, v, col) in enumerate(QS):
        gm = np.exp(np.log(v).mean())
        ax_c.plot([v.min() / gm, v.max() / gm], [i, i], "-", lw=3.0, color=col,
                  solid_capstyle="round", zorder=4)
        ax_c.plot([1.0], [i], "|", ms=9, color="k", mew=1.2, zorder=5)
        ax_c.text(v.max() / gm * 1.25, i, f"{v.max()/v.min():.1f}$\\times$",
                  va="center", ha="left", fontsize=6, color=col)
    ax_c.set_yticks(range(len(QS)))
    ax_c.set_yticklabels([q[0] for q in QS])
    ax_c.set(xscale="log", xlim=(0.12, 13),
             xlabel="range across the parameter grid,\nrelative to geometric mean")
    ax_c.set_title("Initiator amount is conserved, sites are not", loc="left")

    mk = pd.read_csv("optimum_measured_kd.csv")
    OBS = {"pSC101": 3, "pPS10": 4, "F": 4, "RK2": 5, "R6K": 7}
    lo, hi = min(OBS.values()), max(OBS.values())
    ax_d.axvspan(lo, hi, color="0.88", zorder=1,
                 label=f"observed arrays, {lo} to {hi}")
    for kd_, col in zip(sorted(mk.kd.unique()), (ACC, ACC2)):
        g = mk[mk.kd == kd_].sort_values("m")
        ax_d.plot(g.m, g.CV2xf, "-o", ms=3.4, lw=1.5, color=col, zorder=4,
                  label=f"$k_d={kd_:g}$ molecules")
        b = g.loc[g.CV2xf.idxmin()]
        ax_d.plot([b.m], [b.CV2xf], "o", ms=8, mfc="none", mec="k", mew=1.2, zorder=6)
    ax_d.set(xlabel="iterons per plasmid $m$",
             ylabel="$\\mathrm{CV}^2$ of free initiator")
    ax_d.set_title("At measured affinity, optima land in range", loc="left")
    opaque_legend(ax_d, loc="upper center", handlelength=1.6, labelspacing=0.3)

    for ax, L in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        panel_letter(ax, L)
    fig.canvas.draw()
    fig.savefig(os.path.join(_FIGS, "fig3_optimum.png"), dpi=_DPI, bbox_inches="tight")
    fig.savefig(os.path.join(_FIGS, "fig3_optimum.pdf"), bbox_inches="tight")
    return fig


if __name__ == "__main__":
    for name, fn in (("fig1_decoupling", fig1), ("fig2_exact", fig2),
                     ("fig3_optimum", fig3)):
        fn()
        print(f"wrote figures/{name}.pdf and figures/{name}.png")
