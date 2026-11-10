"""All data figures of the paper, drawn from the tables in data/.

    python make_figures.py            Figs. 2-5 and S1-S4 into figures/

Fig. 1 is a schematic drawing with no data behind it (figures/Fig1_schematic.*).
No solver is called, so redrawing takes seconds. Each figure is written as a vector PDF
(the form used in the paper) and a 600-dpi PNG.
"""
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "figures")
W = 6.7                         # inches; APS double column is 17.2 cm
SIZES = (8, 7.5, 7)             # label, legend, tick (pt) at final print size
META_GREY = "0.45"
mpl.rcParams["pdf.fonttype"] = 42   # embed TrueType outlines
mpl.rcParams["ps.fonttype"] = 42


def _csv(name):
    return pd.read_csv(os.path.join(DATA, name + ".csv"))


def apply_figure_style(sizes=SIZES):
    b, m, s_ = sizes
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": b, "axes.titlesize": b, "axes.labelsize": b,
        "legend.fontsize": m, "xtick.labelsize": s_, "ytick.labelsize": s_,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": True, "figure.facecolor": "white"})


def _save(fig, name):
    fig.canvas.draw()
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


ACC = "#b2182b"        # focal: co-scaling architecture
ACC2 = "#2166ac"       # comparator: fixed reservoir
KDC = {3.0: "#08306b", 10.0: "#2171b5", 30.0: "#6baed6", 100.0: "#c6dbef"}


def opaque_legend(ax, **kw):
    kw.setdefault("framealpha", 1.0)
    kw.setdefault("fancybox", False)
    _LEG[id(ax)] = dict(kw)
    leg = ax.legend(**kw)
    if leg:
        leg.get_frame().set_edgecolor("0.75")
        leg.get_frame().set_linewidth(0.6)
        leg.set_zorder(20)
    return leg


_LEG = {}


def _artist_points(ax, guides):
    """Sampled display-space points of plotted lines. guides=True returns only the
    reference lines drawn in axes coordinates (axhline, axvline)."""
    out = []
    for ln in ax.lines:
        if not ln.get_visible():
            continue
        x, y = ln.get_xdata(orig=False), ln.get_ydata(orig=False)
        if len(x) == 0:
            continue
        tx, ty = ln.get_transform().contains_branch_seperately(ax.transData)
        is_guide = not (tx and ty)       # axhline/axvline use one axes-coordinate
        if is_guide != guides:
            continue
        xy = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
        xy = xy[np.isfinite(xy).all(axis=1)]
        if not len(xy):
            continue
        d = ln.get_transform().transform(xy)
        if len(d) > 1:
            d = np.vstack([np.linspace(d[i], d[i + 1], 25, endpoint=False)
                           for i in range(len(d) - 1)] + [d[-1:]])
        out.append(d)
    return np.vstack(out) if out else np.empty((0, 2))


def _hits(pts, x0, y0, x1, y1, pad):
    if not len(pts):
        return 0
    return int(((pts[:, 0] >= x0 - pad) & (pts[:, 0] <= x1 + pad) &
                (pts[:, 1] >= y0 - pad) & (pts[:, 1] <= y1 + pad)).sum())


def settle_legends(fig):
    """Keep every legend off the plotted data: leave it where it is if it is clear,
    otherwise move it to the nearest clear corner region, and if no clear region
    exists add headroom above the data and try again."""
    pad = 0.10 * fig.dpi
    for ax in fig.axes:
        if ax.get_legend() is None:
            continue
        for _ in range(10):
            fig.canvas.draw()
            leg = ax.get_legend()
            kw = {k: v for k, v in _LEG[id(ax)].items()
                  if k not in ("loc", "bbox_to_anchor", "borderaxespad")}
            data, guide = _artist_points(ax, False), _artist_points(ax, True)
            bb, axbb = leg.get_window_extent(), ax.get_window_extent()
            if _hits(data, bb.x0, bb.y0, bb.x1, bb.y1, pad) == 0:
                break
            w, h = bb.width, bb.height
            best = None
            for x0 in np.linspace(axbb.x0 + 0.02 * axbb.width, axbb.x1 - 0.02 * axbb.width - w, 25):
                for y0 in np.linspace(axbb.y0 + 0.02 * axbb.height, axbb.y1 - 0.02 * axbb.height - h, 25):
                    if _hits(data, x0, y0, x0 + w, y0 + h, pad):
                        continue
                    g = _hits(guide, x0, y0, x0 + w, y0 + h, pad)
                    c = (min(abs(x0 - axbb.x0), abs(x0 + w - axbb.x1)) +
                         min(abs(y0 - axbb.y0), abs(y0 + h - axbb.y1)))
                    if best is None or (g, c) < best[0]:
                        best = ((g, c), x0, y0)
            if best is not None:
                _, x0, y0 = best
                leg.remove()
                opaque_legend(ax, loc="lower left", borderaxespad=0,
                              bbox_to_anchor=((x0 - axbb.x0) / axbb.width,
                                              (y0 - axbb.y0) / axbb.height), **kw)
                break
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi * (hi / lo) ** 0.12 if ax.get_yscale() == "log"
                        else hi + 0.12 * (hi - lo))
    fig.canvas.draw()


def place_letters(fig, axes, letters):
    """Panel letters above the top-left corner, left of the y-label and tick labels,
    so they cannot sit on any axis text. Letters in one column share an x position."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    Wp, Hp = fig.canvas.get_width_height()
    px = 0.03 * fig.dpi
    left = []
    for ax in axes:
        x0 = ax.get_window_extent(r).x0
        for art in [ax.yaxis.label, *ax.get_yticklabels()]:
            if art.get_visible() and art.get_text().strip():
                x0 = min(x0, art.get_window_extent(r).x0)
        left.append(x0)
    for i, (ax, L) in enumerate(zip(axes, letters)):
        col = [left[j] for j, a in enumerate(axes)
               if abs(a.get_window_extent(r).x0 - ax.get_window_extent(r).x0) < 5]
        fig.text((min(col) - px) / Wp, (ax.get_window_extent(r).y1 + px) / Hp, L,
                 fontweight="bold", fontsize=10, ha="right", va="bottom")


def trim_axis_above(ax, top):
    """Stop the y axis line and its ticks at `top`. Used where headroom was added only
    to hold a legend, so the empty band does not read as plotted range."""
    lo, hi = ax.get_ylim()
    keep = [t for t in ax.yaxis.get_minorticklocs() if lo <= t <= top]
    ax.yaxis.set_minor_locator(mticker.FixedLocator(keep))
    ax.spines["left"].set_bounds(lo, top)
    ax.set_ylim(lo, hi)



def fig2_decoupling():
    """Fig. 2: the decoupling (a), the sign reversal (b), the collapse onto Eq. (9) (c)."""
    apply_figure_style()
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(7.2, 2.6))
    fig.subplots_adjust(wspace=0.40)

    # (a) the decoupling itself
    dc = _csv("decoupling_check")
    for m_, col, lab in [(0, "0.40", "no sites, $m=0$"),
                         (5, ACC2, "$m=5$"), (14, ACC, "$m=14$")]:
        g = dc[dc.m == m_].sort_values("n")
        ax_a.plot(g.n, g.x_f, "-o", ms=3.2, lw=1.5, color=col, label=lab, zorder=4)
    ax_a.set(xscale="log", yscale="log", xlabel="copy number $n$",
             ylabel="free initiator $\\bar x_f$")
    opaque_legend(ax_a, loc="upper left", handlelength=1.6)

    # (b) same sites, same affinity, opposite sign
    sr = _csv("sign_reversal")
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
    opaque_legend(ax_b, loc="lower left", handlelength=1.6, labelspacing=0.25)

    # (c) collapse onto the capacity ratio
    f1c = _csv("gain_collapse")
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
    opaque_legend(ax_c, loc="lower left", handlelength=1.2, labelspacing=0.22)

    settle_legends(fig)
    trim_axis_above(ax_b, 2.6)      # headroom holds the legend only
    place_letters(fig, (ax_a, ax_b, ax_c), "abc")
    _save(fig, "Fig2_decoupling")


def fig3_exact():
    apply_figure_style()
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(W, 2.35))
    fig.subplots_adjust(wspace=0.42)
    bc = _csv("boundary_check")
    g0 = bc[bc.m == 14].sort_values("nmax")
    hc = _csv("handcuff_convergence").sort_values("nmax")
    ax_a.plot(g0.nmax, g0.CV2n, "-o", ms=3.4, lw=1.5, color=ACC, zorder=5,
              label="autorepression only")
    ax_a.plot(hc.nmax, hc.CV2n, "-s", ms=3.4, lw=1.5, color=ACC2, zorder=5,
              label="plus handcuffing")
    ax_a.set(xlabel="state-space ceiling $n_{\\max}$", yscale="log",
             ylabel="$\\mathrm{CV}^2$ of copy number")
    ax_a.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax_a.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax_a.set_yticks([0.2, 0.5, 1, 2]); ax_a.set_ylim(0.17, 2.6)
    opaque_legend(ax_a, loc="center right", bbox_to_anchor=(1.0, 0.40), handlelength=1.3, fontsize=6.5)

    pm = _csv("copy_number_pmf")
    for m_, col in [(1, "0.40"), (5, ACC2), (20, ACC)]:
        g = pm[pm.m == m_].sort_values("n")
        ax_b.plot(g.n, g.p, "-o", ms=2.8, lw=1.2, color=col, zorder=4, label=f"$m={m_}$")
    ax_b.set(xlabel="copy number $n$", ylabel="stationary probability", xlim=(0, 15), ylim=(0, 0.20))
    opaque_legend(ax_b, loc="upper right", handlelength=1.3, fontsize=6.5, title="mean 5", title_fontsize=6.5)

    ls = _csv("loss_rate_sweep").sort_values("m")
    ax_c.axvspan(3, 7, color="0.88", zorder=0, lw=0, label="observed $m$")
    ax_c.plot(ls.m, ls.CV2n / ls.CV2n.iloc[0], "-o", ms=3.2, lw=1.5, color=ACC2,
              zorder=5, label="$\\mathrm{CV}^2_n$")
    ax_c.plot(ls.m, ls.L / ls.L.iloc[0], "-^", ms=3.2, lw=1.5, color=ACC,
              zorder=5, label="loss $\\langle L\\rangle$")
    ax_c.set(xscale="log", xlabel="iterons per plasmid $m$",
             ylabel="relative to $m=1$", ylim=(0.45, 1.04))
    opaque_legend(ax_c, loc="lower left", handlelength=1.3, fontsize=6.5)
    place_letters(fig, (ax_a, ax_b, ax_c), "abc")
    _save(fig, "Fig3_exact")


def fig4_lna():
    apply_figure_style()
    lc = _csv("lna_comparison").sort_values("m")
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(W, 2.35))
    fig.subplots_adjust(wspace=0.42)
    EX = dict(color="k", lw=1.6, ms=3.4, zorder=6)
    ax_a.plot(lc.m, lc.exact_CV2n, "-o", label="exact", **EX)
    ax_a.plot(lc.m, lc.lna_CV2n, "-s", ms=3.0, lw=1.2, mfc="white", color=ACC2,
              label="LNA, same $k_x$")
    ax_a.plot(lc.m, lc.lnaM_CV2n, "--", lw=1.3, color=ACC2, label="LNA, mean retuned")
    ax_a.set(xscale="log", xlabel="iterons per plasmid $m$", ylabel="$\\mathrm{CV}^2$ of copy number", ylim=(0.22, 0.42))
    opaque_legend(ax_a, loc="upper right", handlelength=1.4, fontsize=6.5)

    ax_b.plot(lc.m, lc.exact_L, "-o", label="exact", **EX)
    ax_b.plot(lc.m, lc.lnaM_L_closed, "--", lw=1.3, color=ACC, label="Gaussian, closed")
    ax_b.plot(lc.m, lc.lnaM_L_discrete, "--", lw=1.3, color=ACC2, label="Gaussian on $n\\geq1$")
    ax_b.set(xscale="log", xlabel="iterons per plasmid $m$",
             ylabel="loss per division $\\langle L\\rangle$", ylim=(0, 0.8))
    opaque_legend(ax_b, loc="upper right", handlelength=1.4, fontsize=6.5)

    ax_c.plot(lc.m, lc.exact_CV2xf, "-o", label="exact", **EX)
    ax_c.plot(lc.m, lc.lnaM_CV2xf_tot, "--", lw=1.3, color=ACC2, label="LNA, mean retuned")
    j = lc.exact_CV2xf.idxmin(); k = lc.lnaM_CV2xf_tot.idxmin()
    ax_c.plot([lc.m[j]], [lc.exact_CV2xf[j]], "o", ms=8, mfc="none", mec="k", mew=1.1, zorder=7)
    ax_c.plot([lc.m[k]], [lc.lnaM_CV2xf_tot[k]], "o", ms=8, mfc="none", mec=ACC2, mew=1.1, zorder=7)
    ax_c.set(xscale="log", xlabel="iterons per plasmid $m$", ylabel="$\\mathrm{CV}^2$ of free initiator")
    ax_c.set_ylim(0.035, 0.14)
    opaque_legend(ax_c, loc="upper left", handlelength=1.4, fontsize=6.5)
    place_letters(fig, (ax_a, ax_b, ax_c), "abc")
    _save(fig, "Fig4_lna")


def figS1_noautorep():
    apply_figure_style()
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    na = _csv("no_autorep_control")
    ax.plot(na.nmax, na.En / na.nmax, "-o", ms=3.6, lw=1.5, color=ACC, zorder=5,
            label="titration only")
    ax.axhline(1.0, ls="--", lw=1.0, color=META_GREY, zorder=2, label="all mass at the ceiling")
    ax.set(xlabel="state-space ceiling $n_{\\max}$", ylabel="mean copy number / ceiling",
           ylim=(0.0, 1.15))
    opaque_legend(ax, loc="lower left", handlelength=1.6)
    _save(fig, "FigS1_no_autorepression")


def figS2_binding():
    """Validation of the fast-binding reduction against explicit binding kinetics."""
    apply_figure_style()
    bv = _csv("binding_validation").sort_values(["m", "separation"])
    Q = [("En", "mean copy number $\\langle n\\rangle$"), ("CV2n", "copy-number noise $\\mathrm{CV}^2_n$"),
         ("Exf", "mean free initiator $\\langle x_f\\rangle$"), ("CV2xf", "free-initiator noise $\\mathrm{CV}^2_{x_f}$"),
         ("L", "loss rate $\\langle L\\rangle$")]
    cols = {1: "#6baed6", 3: "#2171b5", 5: "#08306b"}; mk = {1: "o", 3: "s", 5: "^"}
    fig, axs = plt.subplots(2, 3, figsize=(W, 4.4), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.12, hspace=0.32)
    axs = axs.ravel()
    lo, hi = 5.76, 4320.0
    for ax, (q, title) in zip(axs[:5], Q):
        for m, g in bv.groupby("m"):
            ax.plot(g.separation, g["relerr_" + q].abs(), "-" + mk[m], ms=2.6, lw=1.2, color=cols[m],
                    label=f"$m={m}$")
            ax.plot(g.separation, g["relerr_fastlimit_" + q].abs(), "--", lw=0.9, color=cols[m])
        ax.set_title(title, fontsize=7.5)
    worst = bv.assign(w=bv[["relerr_" + q for q, _ in Q]].abs().max(axis=1))
    for m, g in worst.groupby("m"):
        axs[5].plot(g.separation, g.w, "-" + mk[m], ms=2.6, lw=1.2, color=cols[m])
    s = np.logspace(np.log10(bv.separation.min()), np.log10(bv.separation.max()), 50)
    axs[5].plot(s, 0.01 * 5.76 / s, ":", lw=1.0, color=META_GREY, label="$\\propto$ 1/separation")
    axs[5].set_title("largest of the five", fontsize=7.5)
    for ax in axs:
        ax.axvspan(lo, hi, color="0.9", zorder=0, lw=0)
        ax.axhline(0.01, lw=0.7, color="0.5", zorder=1)
        ax.set(xscale="log", yscale="log", ylim=(1e-9, 0.2))
    for ax in axs[3:]:
        ax.set_xlabel("separation $k_{off}T_{gen}$")
    for ax in axs[::3]:
        ax.set_ylabel("|relative error|")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=cols[m], marker=mk[m], ms=3, lw=1.2, label=f"$m={m}$") for m in (1, 3, 5)]
    handles += [Line2D([], [], color="k", lw=1.2, label="vs reduced model"),
                Line2D([], [], color="k", lw=0.9, ls="--", label="vs exact fast limit")]
    opaque_legend(axs[1], handles=handles, loc="lower left", handlelength=1.6, fontsize=6.0)
    place_letters(fig, list(axs), "abcdef")
    _save(fig, "FigS2_binding_validation")


def figS3_khc():
    """Sensitivity of the matched-mean results to the handcuffing constant K_hc."""
    apply_figure_style()
    S = _csv("khc_sensitivity").sort_values(["K_hc", "m"])
    cols = {0.25: "#6baed6", 0.5: "#2171b5", 1.0: "#08306b"}
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(W, 2.35))
    fig.subplots_adjust(wspace=0.45)
    for k, g in S.groupby("K_hc"):
        g = g.reset_index(drop=True)
        lab = f"$K_{{hc}}={k:g}$" + (" (baseline)" if k == 0.5 else "")
        ax_a.plot(g.m, g.CV2n / g.CV2n.iloc[0], "-o", ms=3.0, lw=1.4, color=cols[k], label=lab)
        ax_b.plot(g.m, g.L / g.L.iloc[0], "-^", ms=3.0, lw=1.4, color=cols[k], label=lab)
        ax_c.plot(g.m, g.CV2xf, "-o", ms=3.0, lw=1.4, color=cols[k], label=lab)
        j = g.CV2xf.idxmin()
        ax_c.plot([g.m[j]], [g.CV2xf[j]], "o", ms=7, mfc="none", mec="k", mew=1.0, zorder=7)
    for ax in (ax_a, ax_b):
        ax.axvspan(3, 7, color="0.88", zorder=0, lw=0)
        ax.set(xscale="log", xlabel="iterons per plasmid $m$", ylim=(0.55, 1.03))
    ax_a.set_ylabel("$\\mathrm{CV}^2_n$ relative to $m=1$")
    ax_b.set_ylabel("loss $\\langle L\\rangle$ relative to $m=1$")
    ax_c.set(xscale="log", yscale="log", xlabel="iterons per plasmid $m$", ylabel="$\\mathrm{CV}^2$ of free initiator")
    ax_c.set_yticks([0.03, 0.05, 0.1, 0.15])
    ax_c.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax_c.yaxis.set_minor_formatter(mticker.NullFormatter())
    opaque_legend(ax_a, loc="lower left", handlelength=1.4, fontsize=6.5)
    place_letters(fig, (ax_a, ax_b, ax_c), "abc")
    _save(fig, "FigS3_khc_sensitivity")



def fig5_optimum():
    """Fig. 5: the free-initiator noise minimum and its dependence on affinity."""
    apply_figure_style()
    fig, axs = plt.subplots(2, 2, figsize=(7.2 * 0.975, 5.0 * 0.975))
    (ax_a, ax_b), (ax_c, ax_d) = axs
    fig.subplots_adjust(wspace=0.36, hspace=0.50)

    ls = _csv("loss_rate_sweep").sort_values("m").reset_index(drop=True)
    j = ls.CV2xf.idxmin()
    ax_a.plot(ls.m, ls.CV2xf, "-o", ms=3.6, lw=1.6, color=ACC, zorder=4, label="free initiator")
    ax_a.plot([ls.m[j]], [ls.CV2xf[j]], "o", ms=8, mfc="none", mec="k", mew=1.3, zorder=6,
              label=f"optimum, $m={int(ls.m[j])}$")
    ax_a.set(xscale="log", xlabel="iterons per plasmid $m$", ylabel="$\\mathrm{CV}^2$ of free initiator")
    opaque_legend(ax_a, loc="upper left", handlelength=1.6)

    os_ = _csv("optimum_summary")
    sl, ic = np.polyfit(np.log(os_.kd), np.log(os_.m), 1)
    kk = np.logspace(np.log10(os_.kd.min()), np.log10(os_.kd.max()), 50)
    ax_b.plot(kk, np.exp(ic) * kk ** sl, "--", lw=1.3, color=META_GREY, zorder=3,
              label=f"fit, effective slope {sl:.2f}")
    KC = {2.0: "#6baed6", 4.0: "#2171b5", 8.0: "#08306b"}
    MK = {2.0: "v", 4.0: "o", 8.0: "^"}
    OFF = {2.0: 0.88, 4.0: 1.0, 8.0: 1.13}           # small horizontal offsets so K values do not overlap
    for K, g in os_.groupby("K"):
        ax_b.plot(g.kd * OFF[K], g.m, MK[K], ms=4.4, color=KC[K], zorder=5,
                  label=f"repression threshold $K={K:g}$")
    ax_b.set(xscale="log", yscale="log", xlabel="dissociation constant $k_d$",
             ylabel="optimal array size $m^*$", ylim=(2, 20))
    ax_b.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax_b.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax_b.set_yticks([2, 3, 5, 10, 20])
    opaque_legend(ax_b, loc="upper left", handlelength=1.4, labelspacing=0.25)

    QS = [("site number $m^*$", os_.m, ACC2), ("capacity $\\chi$", os_.chi, "0.55"),
          ("bound initiator $x_b$", os_.x_b, ACC), ("free initiator $\\bar x_f$", os_.Exf, "#1b7837")]
    for i, (lab, v, col) in enumerate(QS):
        gm = np.exp(np.log(v).mean())
        ax_c.plot([v.min() / gm, v.max() / gm], [i, i], "-", lw=3.0, color=col,
                  solid_capstyle="round", zorder=4)
        ax_c.plot([1.0], [i], "|", ms=9, color="k", mew=1.2, zorder=5)
        ax_c.text(v.max() / gm * 1.25, i, f"{v.max() / v.min():.1f}$\\times$",
                  va="center", ha="left", fontsize=6, color=col)
    ax_c.set_yticks(range(len(QS)))
    ax_c.set_yticklabels([q[0] for q in QS])
    ax_c.set(xscale="log", xlim=(0.05, 30),
             xlabel="range across the parameter grid,\nrelative to geometric mean")

    p1 = _csv("optimum_p1")
    lo, hi = 3, 7                                    # pSC101 3, pPS10 4, F 4, RK2 5, R6K 7
    ax_d.axvspan(lo, hi, color="0.88", zorder=1, label=f"observed arrays, {lo} to {hi}")
    for kd_, col in zip(sorted(p1.kd.unique()), (ACC, ACC2)):
        g = p1[p1.kd == kd_].sort_values("m")
        ax_d.plot(g.m, g.CV2xf, "-o", ms=3.4, lw=1.5, color=col, zorder=4,
                  label=f"$k_d={kd_:g}$ molecules")
        b = g.loc[g.CV2xf.idxmin()]
        ax_d.plot([b.m], [b.CV2xf], "o", ms=8, mfc="none", mec="k", mew=1.2, zorder=6)
    ax_d.set(xlabel="iterons per plasmid $m$", ylabel="$\\mathrm{CV}^2$ of free initiator")
    ax_d.set_xticks([1, 3, 5, 7, 10, 14])
    opaque_legend(ax_d, loc="upper center", handlelength=1.6, labelspacing=0.3)

    settle_legends(fig)
    place_letters(fig, (ax_a, ax_b, ax_c, ax_d), "abcd")
    _save(fig, "Fig5_optimum")



def figS4_division():
    """Continuous dilution (main model) against discrete division with binomial partitioning."""
    apply_figure_style()
    ls = _csv("loss_rate_sweep").sort_values("m").reset_index(drop=True)
    cc = _csv("cell_cycle_check").sort_values("m").reset_index(drop=True)
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(W, 2.35))
    fig.subplots_adjust(wspace=0.45)
    CON = dict(color=ACC2, lw=1.5, ms=3.2, zorder=5)
    DIS = dict(color=ACC, lw=1.5, ms=3.2, zorder=6)
    ax_a.plot(ls.m, ls.CV2n / ls.CV2n[0], "-o", label="continuous dilution", **CON)
    ax_a.plot(cc.m, cc.CV2n / cc.CV2n[0], "-^", label="discrete division", **DIS)
    ax_a.plot(cc.m, cc.CV2n_birth / cc.CV2n_birth[0], "--", color=ACC, lw=1.1, zorder=4,
              label="discrete, at birth")
    ax_b.plot(ls.m, ls.L / ls.L[0], "-o", label="continuous dilution", **CON)
    ax_b.plot(cc.m, cc.L / cc.L[0], "-^", label="discrete division", **DIS)
    for ax in (ax_a, ax_b):
        ax.axvspan(3, 7, color="0.88", zorder=0, lw=0)
        ax.set(xscale="log", xlabel="iterons per plasmid $m$", ylim=(0.55, 1.03))
    ax_a.set_ylabel("$\\mathrm{CV}^2_n$ relative to $m=1$")
    ax_b.set_ylabel("loss $\\langle L\\rangle$ relative to $m=1$")
    for d, st, kw in ((ls, "-o", CON), (cc, "-^", DIS)):
        ax_c.plot(d.m, d.CV2xf, st, **kw)
        j = d.CV2xf.idxmin()
        ax_c.plot([d.m[j]], [d.CV2xf[j]], "o", ms=7, mfc="none", mec="k", mew=1.0, zorder=7)
    ax_c.set(xscale="log", xlabel="iterons per plasmid $m$", ylabel="$\\mathrm{CV}^2$ of free initiator")
    opaque_legend(ax_a, loc="lower left", handlelength=1.4, fontsize=6.5)
    place_letters(fig, (ax_a, ax_b, ax_c), "abc")
    _save(fig, "FigS4_discrete_division")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for fn in (fig2_decoupling, fig3_exact, fig4_lna, fig5_optimum,
               figS1_noautorep, figS2_binding, figS3_khc, figS4_division):
        fn()
        print("drew", fn.__name__)
