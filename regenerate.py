"""Regenerate the result tables behind the figures and the quoted numbers.

    python regenerate.py --list          show every target and its cost
    python regenerate.py fast            the deterministic/root-finding tables (seconds)
    python regenerate.py <name> [...]    named targets
    python regenerate.py all             everything, including the slow CME sweeps

Targets marked SLOW solve a sparse stationary master equation on a two-dimensional
state space and take minutes each; the whole set is a few hours on one core. The
shipped data/ directory holds the outputs used in the paper, and verify.py compares a
regenerated file against it.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iteron_cme  # noqa: E402

LN2 = np.log(2.0)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Parameters used throughout the paper unless a target overrides them.
BASE = dict(kd=3.0, B=2.0, gf=LN2, beta=1.0, krep=0.05, gd=LN2, K=4.0, nH=2.0,
            nmin=1, K_hc=0.5)
NBAR = 5.0          # mean copy number held fixed in every matched-mean sweep
SSUP = 5.0          # per-plasmid initiator supply in the deterministic balance


# --------------------------------------------------------------- deterministic helpers
def _xf(n, m, kd, beta=1.0, s=SSUP, M_fixed=None):
    """Free initiator from the deterministic balance n s = x_f + beta M p(x_f)."""
    M = (n * m) if M_fixed is None else M_fixed
    return brentq(lambda x: n * s - (x + beta * M * x / (kd + x)),
                  1e-14, 1e10, xtol=1e-14, rtol=1e-15)


def _loggain(n, m, kd, beta=1.0, s=SSUP, fixed=False, eps=1e-6):
    """d ln x_f / d ln n, by central difference in log n."""
    M_fixed = (n * m) if fixed else None
    hi = _xf(n * (1 + eps), m, kd, beta, s, M_fixed)
    lo = _xf(n * (1 - eps), m, kd, beta, s, M_fixed)
    return (np.log(hi) - np.log(lo)) / np.log((1 + eps) / (1 - eps))


# --------------------------------------------------------------- fast targets
def decoupling_check():
    """Figure 1a: free initiator against copy number, three array sizes."""
    kd = BASE["kd"]
    rows = []
    for m in (0, 5, 14):
        for n in (1, 2, 5, 10, 20, 50, 100):
            x = _xf(n, m, kd)
            rows.append(dict(m=m, n=n, x_f=x, x_f_per_plasmid=x / n,
                             p=x / (kd + x),
                             g_co=_loggain(n, m, kd) if m else 1.0))
    return pd.DataFrame(rows)


def fig1b_signreversal():
    """Figure 1b: co-scaling versus fixed reservoir of the same mean size."""
    rows = []
    for kd in (3.0, 10.0, 30.0):
        for m in range(1, 21):
            M0 = NBAR * m
            xco = _xf(NBAR, m, kd)
            xfx = _xf(NBAR, m, kd, M_fixed=M0)
            pc, pf = xco / (kd + xco), xfx / (kd + xfx)
            rows.append(dict(
                kd=kd, m=m, M=M0,
                g_co=_loggain(NBAR, m, kd),
                g_co_pred=xco / (xco + NBAR * m * pc * (1 - pc)),
                g_fix=_loggain(NBAR, m, kd, fixed=True),
                g_fix_pred=(xfx + M0 * pf) / (xfx + M0 * pf * (1 - pf)),
                p_co=pc, p_fix=pf, sat=1 / (1 - pf)))
    return pd.DataFrame(rows)


def fig1c_collapse():
    """Figure 1c: the gain collapses onto 1/(1 + beta chi / x_f)."""
    rows = []
    for kd in (3.0, 10.0, 30.0, 100.0):
        for m in np.unique(np.round(np.logspace(0, 2.2, 22)).astype(int)):
            x = _xf(NBAR, int(m), kd)
            p = x / (kd + x)
            M = NBAR * int(m)
            chi = M * p * (1 - p)
            rows.append(dict(kd=kd, m=int(m), M=M, p=p, chi=chi, x_f=x,
                             cap_ratio=chi / x, g_co=_loggain(NBAR, int(m), kd),
                             g_pred=1.0 / (1.0 + chi / x)))
    return pd.DataFrame(rows)


def gain_with_sn():
    """Gain when per-plasmid synthesis itself depends on copy number, s(n) = s0 n^sigma."""
    def run(m, kd, beta, s0, nbar, sigma, K=None, nH=0.0, eps=1e-6):
        def h(x):
            return 1.0 if not K else 1.0 / (1.0 + (x / K) ** nH)

        def solve(nn):
            return brentq(lambda x: nn * (s0 * nn ** sigma) * h(x)
                          - (x + beta * nn * m * x / (kd + x)),
                          1e-14, 1e10, xtol=1e-14, rtol=1e-15)
        x0 = solve(nbar)
        num = (np.log(solve(nbar * (1 + eps))) - np.log(solve(nbar * (1 - eps)))) \
            / np.log((1 + eps) / (1 - eps))
        p = x0 / (kd + x0)
        Xb = beta * nbar * m * p
        chi = Xb * (1 - p)
        a = 0.0 if not K else nH * (x0 / K) ** nH / (1.0 + (x0 / K) ** nH)
        an = (x0 + sigma * (x0 + Xb)) / (x0 + chi + a * (x0 + Xb))
        return x0, num, an, abs(num / an - 1)

    rows = []
    for sigma in (0.0, -0.1, -0.25, -0.5):
        for m in (3, 5, 9):
            for K, nH in ((None, 0.0), (4.0, 2.0)):
                x0, num, an, rel = run(m, 3.0, 1.0, SSUP, NBAR, sigma, K, nH)
                rows.append(dict(sigma=sigma, m=m, autorep=bool(K), x_f=x0,
                                 g_numeric=num, g_analytic=an, rel=rel))
    return pd.DataFrame(rows)


def timescale_separation():
    """Binding residence time against generation time, over plausible on-rates."""
    rows = []
    for kon in (1e6, 1e7, 1e8):
        for kd_nM in (3.2, 12.0):
            koff = kon * kd_nM * 1e-9
            tau = 1.0 / koff
            for tgen in (30, 60):
                rows.append(dict(k_on=kon, Kd_nM=kd_nM, k_off_s=koff, residence_s=tau,
                                 Tgen_min=tgen, separation=tgen * 60 / tau))
    return pd.DataFrame(rows)


# --------------------------------------------------------------- slow targets (CME)
def _match_kx(m, lo=1.0, hi=4000.0, nmax=25, xTmax=None, target=NBAR, **kw):
    """Synthesis rate that puts the mean copy number at `target`."""
    def f(kx):
        xt = xTmax if xTmax else int(60 + 4 * kx)
        return iteron_cme.solve(m, kx=kx, nmax=nmax, xTmax=xt, **BASE, **kw)["En"] - target
    return brentq(f, lo, hi, xtol=1e-3)


def loss_rate_sweep():
    """SLOW. Variance and random-segregation loss against array size, matched mean."""
    rows = []
    for m in (1, 2, 3, 4, 5, 7, 9, 12, 15, 20):
        kx = _match_kx(m)
        r = iteron_cme.solve(m, kx=kx, nmax=25, xTmax=int(60 + 4 * kx), **BASE)
        L = float(np.sum(r["pn"] * 2.0 ** (1 - np.asarray(r["nvals"]))))
        rows.append(dict(m=m, kx=kx, En=r["En"], CV2n=r["CV2n"], L=L,
                         b_eff=(L / 2.0) ** (1.0 / r["En"]), CV2xf=r["CV2xf"],
                         edge_n=r["edge_n"]))
    return pd.DataFrame(rows)


def clustering_test():
    """SLOW. Does the restoring force survive spatial restriction of pairing partners?"""
    models = {"well-mixed (paper)": dict(),
              "clustered, 3 foci": dict(hc_foci=3.0),
              "hard cap, 4 partners": dict(hc_cap=4.0),
              "hard cap, 2 partners": dict(hc_cap=2.0)}
    rows = []
    for lab, kw in models.items():
        # the foci model is carried to higher ceilings because the manuscript
        # quotes its convergence at 60 and 80
        ceilings = (25, 40, 60, 80) if lab == "clustered, 3 foci" else (25, 40)
        for nmax in ceilings:
            r = iteron_cme.solve(5, kx=400.0, nmax=nmax, xTmax=1700, **BASE, **kw)
            rows.append(dict(model=lab, nmax=nmax, En=r["En"], CV2n=r["CV2n"],
                             edge=r["edge_n"], edge_T=r["edge_T"]))
    return pd.DataFrame(rows)


def no_autorep_control():
    """SLOW. Without autorepression the mean sits at whatever ceiling is imposed."""
    rows = []
    for nmax in (20, 30, 40):
        r = iteron_cme.solve(5, kx=400.0, nmax=nmax, xTmax=1700,
                             **{**BASE, "K": None, "nH": 1.0})
        rows.append(dict(nmax=nmax, En=r["En"], CV2n=r["CV2n"], edge_n=r["edge_n"]))
    return pd.DataFrame(rows)


def handcuff_convergence():
    """SLOW. With pairing present the variance stops moving as the ceiling recedes."""
    rows = []
    for nmax in (25, 40, 60):
        kx = _match_kx(14, nmax=nmax)
        r = iteron_cme.solve(14, kx=kx, nmax=nmax, xTmax=int(60 + 4 * kx), **BASE)
        rows.append(dict(nmax=nmax, kx=kx, En=r["En"], CV2n=r["CV2n"],
                         edge_n=r["edge_n"], edge_T=r["edge_T"]))
    return pd.DataFrame(rows)


FAST = {"decoupling_check": decoupling_check, "fig1b_signreversal": fig1b_signreversal,
        "fig1c_collapse": fig1c_collapse, "gain_with_sn": gain_with_sn,
        "timescale_separation": timescale_separation}
SLOW = {"loss_rate_sweep": loss_rate_sweep, "clustering_test": clustering_test,
        "no_autorep_control": no_autorep_control,
        "handcuff_convergence": handcuff_convergence}
TARGETS = {**FAST, **SLOW}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", default=["fast"])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=DATA)
    args = ap.parse_args()
    if args.list:
        for k in FAST:
            print(f"  fast  {k}")
        for k in SLOW:
            print(f"  SLOW  {k}")
        return
    names = []
    for tgt in args.targets:
        if tgt == "fast":
            names += list(FAST)
        elif tgt == "all":
            names += list(TARGETS)
        elif tgt in TARGETS:
            names.append(tgt)
        else:
            raise SystemExit(f"unknown target {tgt!r}; --list to see them")
    os.makedirs(args.out, exist_ok=True)
    for name in dict.fromkeys(names):
        t0 = time.time()
        df = TARGETS[name]()
        path = os.path.join(args.out, name + ".csv")
        df.to_csv(path, index=False)
        print(f"{name:24s} {len(df):5d} rows  {time.time() - t0:7.1f}s  -> {path}")


if __name__ == "__main__":
    main()
