"""Every result table of the paper, computed from the model.

    python compute_tables.py --list            every target, and which are slow
    python compute_tables.py fast              the deterministic tables (seconds)
    python compute_tables.py <name> [...]      named targets
    python compute_tables.py all               everything (several hours on 12 cores)
    python compute_tables.py <name> --jobs 4   worker processes for the parallel sweeps

Targets marked SLOW solve the stationary chemical master equation (iteron_cme.py) and
take minutes to hours. Tables are written to data/ (or --out). Each table is listed in
README.md with the figure or section of the paper that uses it.
"""
import argparse
import inspect
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
NPROC = max(1, min(12, (os.cpu_count() or 2) - 2))

# Baseline parameters (Table I of the paper); a target overrides only what it varies.
BASE = dict(kd=3.0, B=2.0, gf=LN2, beta=1.0, krep=0.05, gd=LN2, K=4.0, nH=2.0,
            nmin=1, K_hc=0.5)
NBAR = 5.0          # mean copy number held fixed in every matched-mean sweep
SSUP = 5.0          # per-plasmid initiator supply in the deterministic balance (Sec. III A)
M_SWEEP = (1, 2, 3, 4, 5, 7, 9, 12, 14, 20)


def _csv(name):
    return pd.read_csv(os.path.join(DATA, name + ".csv"))


def _loss(r):
    """Random-segregation loss per division, Eq. (13)."""
    return float(np.sum(r["pn"] * 2.0 ** (1 - np.asarray(r["nvals"]))))


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
    """Fig. 2(a): free initiator against copy number, three array sizes."""
    kd = BASE["kd"]
    rows = []
    for m in (0, 5, 14):
        for n in (1, 2, 5, 10, 20, 50, 100):
            x = _xf(n, m, kd)
            rows.append(dict(m=m, n=n, x_f=x, x_f_per_plasmid=x / n,
                             p=x / (kd + x),
                             g_co=_loggain(n, m, kd) if m else 1.0))
    return pd.DataFrame(rows)


def sign_reversal():
    """Fig. 2(b): co-scaling versus a fixed reservoir of the same mean size."""
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


def gain_collapse():
    """Fig. 2(c): the gain collapses onto 1/(1 + beta chi / x_f), Eq. (9)."""
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
    """Sec. III A, Eq. (12): gain when per-plasmid synthesis depends on copy number, s(n) = s0 n^sigma."""
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
    """Appendix B: binding residence time against generation time, over plausible on-rates."""
    rows = []
    for kon in (1e6, 1e7, 1e8):
        for kd_nM in (3.2, 12.0):
            koff = kon * kd_nM * 1e-9
            tau = 1.0 / koff
            for tgen in (30, 60):
                rows.append(dict(k_on=kon, Kd_nM=kd_nM, k_off_s=koff, residence_s=tau,
                                 Tgen_min=tgen, separation=tgen * 60 / tau))
    return pd.DataFrame(rows)




def closed_loop_gain():
    """Sec. III A, Eq. (11): reservoir plus autorepression, against numerical derivatives.
    Deterministic balance n s h(x_f) = x_f + beta n m p(x_f) at n = 10, s = 50, K = 5
    (time in units of 1/gamma_f, beta = 1); 3 array sizes x 3 affinities x 2 Hill exponents."""
    n0, s, K, beta, eps = 10.0, 50.0, 5.0, 1.0, 1e-6
    rows = []
    for m in (5, 14, 50):
        for kd in (0.5, 3.0, 30.0):
            for nH in (1, 2):
                h = lambda x: 1.0 / (1.0 + (x / K) ** nH)
                xf = lambda n: brentq(lambda x: n * s * h(x) - (x + beta * n * m * x / (kd + x)),
                                      1e-14, 1e10, xtol=1e-14, rtol=1e-15)
                x = xf(n0)
                g_num = (np.log(xf(n0 * (1 + eps))) - np.log(xf(n0 * (1 - eps)))) \
                    / np.log((1 + eps) / (1 - eps))
                p = x / (kd + x)
                xb = n0 * m * p
                chi = xb * (1 - p)
                a = nH * (x / K) ** nH / (1 + (x / K) ** nH)
                g_both = x / (x + beta * chi + a * (x + beta * xb))
                g_auto = 1 / (1 + a)
                g_res = x / (x + beta * chi)
                rows.append(dict(m=m, kd=kd, nH=nH, x_f=x, p=p, a=a, g_num=g_num, g_both=g_both,
                                 rel=abs(g_num / g_both - 1), g_auto=g_auto, g_res=g_res,
                                 res_adds=g_auto / g_both, auto_adds=g_res / g_both))
    return pd.DataFrame(rows)


# --------------------------------------------------------------- matched-mean machinery
def _xT_default(m, nmax, M_fix=0):
    return int(150 + nmax * m + M_fix)


def _solve_at(m, kx, nmax, xTmax=None, **par):
    par = {**BASE, **par}
    xt = xTmax or _xT_default(m, nmax, par.get("M_fix", 0))
    return iteron_cme.solve(m, kx=kx, nmax=nmax, xTmax=xt, **par)


def _match_secant(m, x0, nmax, xTmax=None, tol=1e-7, **par):
    """Secant iteration in ln k_x for <n> = NBAR, started at k_x = exp(x0)."""
    r = _solve_at(m, np.exp(x0), nmax, xTmax, **par); f0 = r["En"] - NBAR
    if abs(f0) < tol:
        return float(np.exp(x0)), r
    x1 = x0 + (0.1 if f0 < 0 else -0.1)
    r = _solve_at(m, np.exp(x1), nmax, xTmax, **par); f1 = r["En"] - NBAR
    for _ in range(60):
        if abs(f1) < tol:
            return float(np.exp(x1)), r
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x2 = min(max(x2, x1 - 1.0), x1 + 1.0)
        x0, f0 = x1, f1
        x1 = x2
        r = _solve_at(m, np.exp(x1), nmax, xTmax, **par); f1 = r["En"] - NBAR
    raise RuntimeError(f"k_x matching did not converge at m={m}, {par}")


def _match_bracket(m, nmax, xTmax=None, **par):
    """Matched mean with no prior guess: bracket <n> = NBAR in ln k_x, then refine."""
    f = lambda x: _solve_at(m, np.exp(x), nmax, xTmax, **par)["En"] - NBAR
    x = np.log(10.0); fx = f(x); step = 1.0 if fx < 0 else -1.0
    while True:
        y = x + step; fy = f(y)
        if fx * fy < 0:
            break
        x, fx = y, fy
    lo, hi = sorted((x, y))
    return _match_secant(m, brentq(f, lo, hi, xtol=1e-3), nmax, xTmax, tol=1e-7, **par)


def _row(m, kx, r, nmax, par):
    par = {**BASE, **par}
    kd = par["kd"]; Exf = float(r["Exf"]); p = Exf / (kd + Exf); M = float(r["En"]) * m
    return dict(kd=kd, K=par["K"], K_hc=par["K_hc"], m=m, kx=kx, En=float(r["En"]),
                CV2n=float(r["CV2n"]), Exf=Exf, CV2xf=float(r["CV2xf"]), p=p, x_b=M * p,
                chi=M * p * (1 - p), L=_loss(r), edge_n=float(r["edge_n"]),
                edge_T=float(r["edge_T"]), nmax=nmax, xTmax=int(r["xTmax"]))


def _chain(job):
    """Matched-mean solutions along a list of (m, n_max) at fixed parameters. Each solution
    seeds the next: by linear extrapolation in (ln m, ln k_x) along increasing m, or directly
    when the same m is re-solved on a larger ceiling. job = (par, [(m, nmax), ...])."""
    par, mlist = job
    rows, hist = [], []
    for m, nmax in mlist:
        if not hist:
            kx, r = _match_bracket(m, nmax, **par)
        else:
            if len(hist) == 1 or m == mlist[len(hist) - 1][0]:
                x0 = hist[-1][1]          # first step, or same m on a larger ceiling
            else:
                (a0, b0), (a1, b1) = hist[-2], hist[-1]
                x0 = b1 + (b1 - b0) / (a1 - a0) * (np.log(m) - a1)
            kx, r = _match_secant(m, x0, nmax, **par)
        hist.append((np.log(max(m, 1)), np.log(kx)))
        rows.append(_row(m, kx, r, nmax, par))
    return rows


def _run_chains(jobs, nproc):
    """Run independent chains in parallel, each in its own Python process (plain
    subprocesses rather than multiprocessing, which needs named pipes)."""
    import json
    import subprocess
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    here = os.path.abspath(__file__)
    jobs = sorted(jobs, key=lambda j: -sum(m * n for m, n in j[1]))
    with tempfile.TemporaryDirectory() as tmp:
        def go(k):
            out = os.path.join(tmp, f"chain{k}.json")
            subprocess.run([sys.executable, here, "--chain", json.dumps(jobs[k]),
                            "--chain-out", out], check=True)
            with open(out) as fh:
                return json.load(fh)
        with ThreadPoolExecutor(max_workers=max(1, min(nproc, len(jobs)))) as ex:
            out = list(ex.map(go, range(len(jobs))))
    return pd.DataFrame([r for rows in out for r in rows])


# --------------------------------------------------------------- slow targets (CME)
def loss_rate_sweep(nproc=NPROC):
    """SLOW. Fig. 3(c), Fig. 5(a), Sec. III C: CV^2, loss and free-initiator noise against
    array size at matched mean five (also the input of the LNA comparison)."""
    jobs = [({}, [(m, 25)]) for m in M_SWEEP]
    d = _run_chains(jobs, nproc).sort_values("m").reset_index(drop=True)
    d["L_at_mean"] = 2.0 ** (1 - NBAR)
    d["jensen"] = d.L / d.L_at_mean
    d["b_eff"] = (d.L / 2.0) ** (1.0 / d.En)
    return d


def handcuff_convergence(nproc=NPROC):
    """SLOW. Fig. 3(a): with handcuffing, CV^2 of copy number is independent of the ceiling."""
    jobs = [({}, [(14, 25), (14, 40), (14, 60)])]
    return _run_chains(jobs, nproc).sort_values("nmax")


def boundary_check(nproc=NPROC):
    """SLOW. Fig. 3(a), Sec. III B: autorepression without handcuffing (K_hc = 0); the
    matched-mean CV^2 of copy number grows with the ceiling (m = 0 and m = 14)."""
    jobs = [(dict(K_hc=0.0), [(m, 25), (m, 40), (m, 60)]) for m in (0, 14)]
    return _run_chains(jobs, nproc).sort_values(["m", "nmax"])


def no_autorep_control():
    """SLOW. Fig. S1: titration alone (no autorepression, no handcuffing); the mean sits at
    the ceiling."""
    rows = []
    for nmax in (20, 30, 40):
        r = iteron_cme.solve(5, kx=400.0, nmax=nmax, xTmax=1700,
                             **{**BASE, "K": None, "nH": 1.0, "K_hc": 0.0})
        rows.append(dict(nmax=nmax, xTmax=1700, En=r["En"], edge_n=r["edge_n"],
                         edge_T=r["edge_T"], En_over_nmax=r["En"] / nmax))
    return pd.DataFrame(rows)


def clustering_test():
    """SLOW. Sec. II C: pairing partners restricted to a fixed number of clusters, or capped."""
    models = {"well-mixed (paper)": dict(),
              "clustered, 3 foci": dict(hc_foci=3.0),
              "hard cap, 4 partners": dict(hc_cap=4.0),
              "hard cap, 2 partners": dict(hc_cap=2.0)}
    rows = []
    for lab, kw in models.items():
        ceilings = (25, 40, 60, 80) if lab == "clustered, 3 foci" else (25, 40)
        for nmax in ceilings:
            r = iteron_cme.solve(5, kx=400.0, nmax=nmax, xTmax=1700, **BASE, **kw)
            rows.append(dict(model=lab, nmax=nmax, En=r["En"], CV2n=r["CV2n"],
                             edge=r["edge_n"], edge_T=r["edge_T"]))
    return pd.DataFrame(rows)


def copy_number_pmf():
    """SLOW. Fig. 3(b): stationary copy-number distributions at m = 1, 5, 20 (mean five),
    at the synthesis rates of loss_rate_sweep, on a ceiling of 30."""
    ls = _csv("loss_rate_sweep").set_index("m")
    rows = []
    for m in (1, 5, 20):
        r = _solve_at(m, float(ls.loc[m, "kx"]), 30)
        rows += [dict(m=m, n=int(n), p=float(p)) for n, p in zip(r["nvals"], r["pn"])]
    return pd.DataFrame(rows)


def _pairing_strength(m, kx, nmax=25):
    """Stationary pairing strength <E[p^2|x_T] (n-1)>/<n-1>, the factor that multiplies
    K_hc (n-1) in Eq. (5)."""
    r = _solve_at(m, kx, nmax)
    n = np.asarray(r["nvals"]); P = np.asarray(r["P"]).reshape(len(n), -1)
    par = BASE
    Ep2 = np.array([iteron_cme.cond_moments_cached(int(k) * m, par["kd"], int(r["xTmax"]),
                                                   par["K"], par["nH"])[4] for k in n])
    w = P * (n[:, None] - 1)
    return float(np.sum(w * Ep2) / np.sum(w))


def control_pairing_strength():
    """SLOW. Sec. III C, first control: the change in effective pairing strength between
    m = 1 and m = 20, imposed alone at m = 1 (mean re-matched)."""
    ls = _csv("loss_rate_sweep").set_index("m")
    S1 = _pairing_strength(1, float(ls.loc[1, "kx"]))
    S20 = _pairing_strength(20, float(ls.loc[20, "kx"]))
    rows = []
    for f in (1.0, S20 / S1):
        kx, r = _match_secant(1, np.log(float(ls.loc[1, "kx"])), 25, K_hc=BASE["K_hc"] * f)
        rows.append(dict(factor=f, K_hc=BASE["K_hc"] * f, kx=kx, En=r["En"], CV2n=r["CV2n"],
                         L=_loss(r), S_m1=S1, S_m20=S20))
    d = pd.DataFrame(rows)
    full = float(ls.loc[1, "CV2n"] - ls.loc[20, "CV2n"])
    d["share_of_array_effect"] = (d.CV2n.iloc[0] - d.CV2n) / full
    return d


def control_hc_loaded():
    """SLOW. Sec. III C, second control: pairing propensity proportional to the square of the
    number of loaded iterons, normalised to coincide with Eq. (5) at m = 1."""
    rows = []
    for m in (1, 2, 3):
        xt = int(150 + 25 * m + 100 * m * m)
        kx, r = _match_bracket(m, 25, xTmax=xt, hc_loaded=True, hc_mref=1.0)
        rows.append(dict(m=m, kx=kx, En=r["En"], CV2n=r["CV2n"], CV2xf=r["CV2xf"], L=_loss(r),
                         edge_n=r["edge_n"], edge_T=r["edge_T"], xTmax=xt))
    return pd.DataFrame(rows)


def khc_sensitivity(nproc=NPROC):
    """SLOW. Supplemental Sec. S3, Fig. S3: the matched-mean sweep at K_hc = 0.25, 0.5, 1
    (ceiling 35 for K_hc = 0.25 at m <= 4, where the distribution is broadest)."""
    jobs = []
    for k in (0.25, 0.5, 1.0):
        jobs.append((dict(K_hc=k), [(m, 35 if (k == 0.25 and m <= 4) else 25)
                                    for m in M_SWEEP if m <= 4]))
        jobs.append((dict(K_hc=k), [(m, 25) for m in M_SWEEP if m > 4]))
    return _run_chains(jobs, nproc).sort_values(["K_hc", "m"])


def khc_sensitivity_summary():
    """Supplemental Sec. S3: the summary numbers quoted in the text, from khc_sensitivity."""
    rows = []
    for k, g in _csv("khc_sensitivity").groupby("K_hc"):
        g = g.set_index("m")
        j = int(g.CV2xf.idxmin())
        rows.append(dict(K_hc=k, CV2n_m1=g.CV2n[1], CV2n_m20=g.CV2n[20],
                         CV2n_ratio_1_20=g.CV2n[1] / g.CV2n[20],
                         CV2n_drop_3_7=1 - g.CV2n[7] / g.CV2n[3], L_m1=g.L[1], L_m20=g.L[20],
                         L_ratio_1_20=g.L[1] / g.L[20], L_drop_3_7=1 - g.L[7] / g.L[3],
                         argmin_CV2xf=j, CV2xf_depth=g.CV2xf[20] / g.CV2xf[j],
                         CV2xf_m3_over_m4=g.CV2xf[3] / g.CV2xf[4]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------ free-initiator optimum (Fig. 5)
OPT_KD = (0.3, 1.0, 3.0, 10.0, 30.0)
OPT_K = (2.0, 4.0, 8.0)
P1_KD = (1.9, 7.2)            # 3.2 and 12 nM in a 1-fL cell [Das et al. 2005]
_M_LO = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12)
OPT_M = {0.3: _M_LO, 1.0: _M_LO, 1.9: _M_LO, 3.0: _M_LO,
         7.2: (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14),
         10.0: (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20),
         30.0: (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 20, 25)}


def optimum_grid(nproc=NPROC):
    """SLOW. Fig. 5(b,c), Sec. III E: free-initiator noise against m over the (k_d, K) grid,
    full model (handcuffing included), matched mean five, ceiling 25."""
    jobs = [(dict(kd=kd, K=K), [(m, 25) for m in OPT_M[kd]]) for kd in OPT_KD for K in OPT_K]
    return _run_chains(jobs, nproc).sort_values(["kd", "K", "m"])


def optimum_p1(nproc=NPROC):
    """SLOW. Fig. 5(d): the same at the two ends of the measured P1 affinity."""
    jobs = [(dict(kd=kd), [(m, 25) for m in OPT_M[kd]]) for kd in P1_KD]
    return _run_chains(jobs, nproc).sort_values(["kd", "m"])


def optimum_ceiling_check(nproc=NPROC):
    """SLOW. The optimum and its two neighbours re-solved at ceiling 40 (k_x re-matched),
    for every curve of optimum_grid and optimum_p1."""
    g = pd.concat([_csv("optimum_grid"), _csv("optimum_p1")])
    jobs = []
    for (kd, K), s in g.groupby(["kd", "K"]):
        s = s.sort_values("m").reset_index(drop=True)
        j = int(s.CV2xf.idxmin())
        jobs.append((dict(kd=kd, K=K), [(int(m), 40) for m in s.m.iloc[max(j - 1, 0):j + 2]]))
    new = _run_chains(jobs, nproc)
    old = g.set_index(["kd", "K", "m"])
    for q in ("CV2xf", "CV2n", "Exf", "L"):
        new[q + "_n25"] = [old.loc[(a, b, c), q] for a, b, c in zip(new.kd, new.K, new.m)]
        new[q + "_reldev"] = new[q] / new[q + "_n25"] - 1
    return new.sort_values(["kd", "K", "m"])


def optimum_summary():
    """Fig. 5(b,c), Sec. III E: the optimum of each curve of optimum_grid and the quantities
    at the optimum."""
    g = _csv("optimum_grid")
    rows = []
    for (kd, K), s in g.groupby(["kd", "K"]):
        rows.append(s.loc[s.CV2xf.idxmin()].to_dict())
    return pd.DataFrame(rows)


# ------------------------------------------------------------ incompatibility (Appendix D)
ABELES_M = 14                         # iterons of the resident mini-P1 (incA + oriR)
ABELES_CARRIERS = (("low (c~5)", 5), ("high (c~30)", 30))
ABELES_ITERONS = (0, 1, 3, 5, 6, 9)
ABELES_MEAS = ((5, 59.0, "Table 2 pALA731, 5-mer normal spacer"),
               (6, 40.0, "Table 2 pALA2435, 6-mer"),
               (9, 2.0, "Fig 5 / pALA749 wild-type incA (<2%)"))
ABELES_CONTROL = 98.0                 # five non-binding iterons, % retention


def abeles_retention():
    """SLOW. Appendix D: the resident plasmid (full model, mean five without a trans
    reservoir) with M_fix iterons supplied in trans on a carrier of copy number c. The trans
    sites titrate initiator but do not pair (trans_hc = 0), so this isolates titration."""
    kx = float(_csv("loss_rate_sweep").set_index("m").loc[ABELES_M, "kx"])
    cache, rows = {}, []
    for lab, c in ABELES_CARRIERS:
        for it in ABELES_ITERONS:
            M = c * it
            if M not in cache:
                cache[M] = _solve_at(ABELES_M, kx, 25, M_fix=M)
            r = cache[M]
            L = _loss(r)
            rows.append(dict(carrier=lab, c=c, iterons=it, M_fix=M, kx=kx, L=L, En=r["En"],
                             Exf=r["Exf"], CV2n=r["CV2n"], edge_n=r["edge_n"], edge_T=r["edge_T"]))
    return pd.DataFrame(rows)


def abeles_comparison():
    """Appendix D: fold increase in loss required by the measured retentions, against the
    fold increase produced by titration (from abeles_retention, low-copy carrier)."""
    ar = _csv("abeles_retention")
    low = ar[ar.c == 5].set_index("iterons")
    L0 = float(low.L[0])
    rows = []
    for it, ret, src in ABELES_MEAS:
        req = np.log(ret / 100) / np.log(ABELES_CONTROL / 100)
        pred = float(low.L[it]) / L0
        rows.append(dict(iterons=it, M_fix=5 * it, ret_meas=ret, src=src, fold_loss_predicted=pred,
                         fold_loss_required=req, titration_share=pred / req))
    return pd.DataFrame(rows)


# ------------------------------------------------------------ LNA (Sec. III D, Appendix C)
def lna_comparison():
    """Fig. 4, Sec. III D, Appendix C: the linear-noise approximation of the same model
    against the exact matched-mean sweep (seconds)."""
    import lna
    return lna.build_table(DATA)


# ------------------------------------------------------------ explicit binding (Sec. III F, S2)
BIND_XT = {1: 170, 3: 210, 5: 240}
# k_off in units of gamma_d; separation (T_gen / residence time) = (k_off/gamma_d) ln 2
BIND_KG = [1, 2, 5, 5.76 / LN2, 10, 20, 50, 50 / LN2, 100, 200, 500, 1000, 4320 / LN2]
BIND_REALISTIC = (5.76, 4320.0)       # range from timescale_separation
BIND_Q = ("En", "CV2n", "Exf", "CV2xf", "L")


def binding_validation():
    """SLOW (1-2 h; requires pypardiso). Supplemental Sec. S2, Fig. S2: the master equation
    with explicit binding kinetics on (n, x_T, bound), m = 1, 3, 5, 13 unbinding rates,
    against the fast-binding reduction on the same grids."""
    import binding_cme3d as bc
    P0 = {k: v for k, v in BASE.items() if k != "nmin"}
    ls = _csv("loss_rate_sweep").set_index("m")
    raw = []
    for m in (1, 3, 5):
        kx = float(ls.loc[m, "kx"]); g = bc.Grid(m, 1, 25, BIND_XT[m])
        for kg in BIND_KG:
            t0 = time.time()
            r = bc.build_and_solve(m, kx=kx, koff=kg * LN2, grid=g, solver=bc.pardiso_solver, **P0)
            raw.append(dict(m=m, kx=kx, nmax=25, xTmax=BIND_XT[m], n_states=g.N, koff_over_gd=kg,
                            separation=kg * LN2, eps=1.0 / kg,
                            **{k: r[k] for k in BIND_Q + ("Eb", "edge_n", "edge_T", "neg_mass", "resid")},
                            wall_s=time.time() - t0))
            print(f"  binding m={m} separation={kg * LN2:8.2f}  {time.time() - t0:6.1f}s", flush=True)
    return binding_assemble(pd.DataFrame(raw))


def binding_assemble(raw=None):
    """Relative errors of the explicit-binding solutions against the reduced model (ratio
    closure, Eq. (5)) and against the exact fast-binding limit. With no argument it re-uses
    the explicit-binding columns of data/binding_validation.csv (seconds)."""
    import binding_cme3d as bc
    P0 = {k: v for k, v in BASE.items() if k != "nmin"}
    if raw is None:
        raw = _csv("binding_validation")
        raw = raw[[c for c in raw.columns if not (c.endswith("_reduced") or c.startswith("relerr")
                                                   or c == "realistic")]]
    ref = []
    for m in sorted(raw.m.unique()):
        kx = float(raw[raw.m == m].kx.iloc[0])
        for cl in ("ratio", "exact"):
            r = bc.reduced(m, kx=kx, nmax=25, xTmax=BIND_XT[m], closure=cl, **P0)
            ref.append(dict(m=m, closure=cl, **{k: r[k] for k in BIND_Q + ("edge_n", "edge_T")}))
    ref = pd.DataFrame(ref)
    ref.to_csv(os.path.join(DATA, "binding_reduced_ref.csv"), index=False)
    out = raw.copy()
    out.insert(out.columns.get_loc("eps") + 1, "realistic",
               (out.separation >= BIND_REALISTIC[0] - 1e-9) & (out.separation <= BIND_REALISTIC[1] + 1e-9))
    rr = ref[ref.closure == "ratio"].set_index("m"); rx = ref[ref.closure == "exact"].set_index("m")
    for k in BIND_Q:
        out[k + "_reduced"] = out.m.map(rr[k])
    for k in BIND_Q:
        out["relerr_" + k] = out[k] / out[k + "_reduced"] - 1
    for k in BIND_Q:
        out["relerr_fastlimit_" + k] = out[k] / out.m.map(rx[k]) - 1
    tail = ["edge_n", "edge_T", "neg_mass", "resid", "wall_s"]
    return out[[c for c in out.columns if c not in tail] + tail]



# ------------------------------------------------------------ discrete division (Supplemental S4)
CC_NMAX = 40


def _cc_one(m):
    import cell_cycle
    par = {k: BASE[k] for k in ("kd", "B", "krep", "K", "nH", "K_hc")}
    kx0 = 0.85 * float(_csv("loss_rate_sweep").set_index("m").loc[m, "kx"])
    return cell_cycle.matched(m, kx0, CC_NMAX, int(150 + 30 * m), **par)


def cell_cycle_check(nproc=NPROC):
    """SLOW. Supplemental Sec. S4, Fig. S4: the matched-mean sweep with discrete division
    (binomial partitioning of plasmids and initiator at the end of each generation) instead
    of continuous dilution; mean copy number five averaged over the cycle."""
    import json
    import subprocess
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    here = os.path.abspath(__file__)
    ms = sorted(M_SWEEP, reverse=True)
    with tempfile.TemporaryDirectory() as tmp:
        def go(m):
            out = os.path.join(tmp, f"cc{m}.json")
            subprocess.run([sys.executable, here, "--cc", str(m), "--chain-out", out], check=True)
            with open(out) as fh:
                return json.load(fh)
        with ThreadPoolExecutor(max_workers=max(1, min(nproc, len(ms)))) as ex:
            rows = list(ex.map(go, ms))
    return pd.DataFrame(rows).sort_values("m")

# ------------------------------------------------------------ driver
FAST = {"decoupling_check": decoupling_check, "sign_reversal": sign_reversal,
        "gain_collapse": gain_collapse, "gain_with_sn": gain_with_sn,
        "closed_loop_gain": closed_loop_gain, "timescale_separation": timescale_separation}
# order matters: later targets read tables written by earlier ones
SLOW = {"loss_rate_sweep": loss_rate_sweep, "lna_comparison": lna_comparison,
        "handcuff_convergence": handcuff_convergence, "boundary_check": boundary_check,
        "no_autorep_control": no_autorep_control, "clustering_test": clustering_test,
        "copy_number_pmf": copy_number_pmf, "control_pairing_strength": control_pairing_strength,
        "control_hc_loaded": control_hc_loaded, "khc_sensitivity": khc_sensitivity,
        "khc_sensitivity_summary": khc_sensitivity_summary, "optimum_grid": optimum_grid,
        "optimum_p1": optimum_p1, "optimum_summary": optimum_summary,
        "optimum_ceiling_check": optimum_ceiling_check, "abeles_retention": abeles_retention,
        "abeles_comparison": abeles_comparison, "binding_validation": binding_validation,
        "binding_assemble": binding_assemble, "cell_cycle_check": cell_cycle_check}
QUICK = {"lna_comparison", "khc_sensitivity_summary", "optimum_summary", "abeles_comparison",
         "binding_assemble"}
TARGETS = {**FAST, **SLOW}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", default=["fast"])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=None, help="output directory (default data/)")
    ap.add_argument("--jobs", type=int, default=NPROC, help="worker processes for the parallel sweeps")
    ap.add_argument("--chain", help=argparse.SUPPRESS)
    ap.add_argument("--chain-out", help=argparse.SUPPRESS)
    ap.add_argument("--cc", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.cc is not None:             # worker mode for cell_cycle_check
        import json
        with open(args.chain_out, "w") as fh:
            json.dump(_cc_one(args.cc), fh)
        return
    if args.chain:                      # worker mode for the parallel sweeps
        import json
        par, mlist = json.loads(args.chain)
        with open(args.chain_out, "w") as fh:
            json.dump(_chain((par, [tuple(x) for x in mlist])), fh)
        return
    if args.list:
        for k in TARGETS:
            cost = "fast" if (k in FAST or k in QUICK) else "SLOW"
            print(f"  {cost:4s}  {k:26s} {(TARGETS[k].__doc__ or '').split(chr(10))[0].strip()}")
        return
    names = []
    for tgt in args.targets:
        if tgt == "fast":
            names += list(FAST)
        elif tgt == "all":
            names += [k for k in TARGETS if k != "binding_assemble"]
        elif tgt in TARGETS:
            names.append(tgt)
        else:
            raise SystemExit(f"unknown target {tgt!r}; --list to see them")
    out = args.out or DATA
    os.makedirs(out, exist_ok=True)
    for name in dict.fromkeys(names):
        t0 = time.time()
        fn = TARGETS[name]
        df = fn(nproc=args.jobs) if "nproc" in inspect.signature(fn).parameters else fn()
        csv_name = "binding_validation" if name == "binding_assemble" else name
        path = os.path.join(out, csv_name + ".csv")
        df.to_csv(path, index=False)
        print(f"{name:26s} {len(df):5d} rows  {time.time() - t0:7.1f}s  -> {path}", flush=True)


if __name__ == "__main__":
    main()
