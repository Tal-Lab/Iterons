"""Linear-noise approximation (LNA) for the iteron copy-number model of iteron_cme.py.

State (n, x_T). Iteron binding is replaced by its deterministic (mean-field) equilibrium:
    x_T = x_f + M x_f/(kd + x_f),   M = n m,   p = x_f/(kd + x_f)
Propensities (same reactions as iteron_cme.solve, conditional averages replaced by
their values at the deterministic x_f):
    synthesis    a1 = n kx h(x_f)                    x_T += k,  k ~ Geom{1,2,..}, E[k]=B, E[k^2]=B(2B-1)
    loss         a2 = gf x_f + beta gf (x_T - x_f)   x_T -= 1
    replication  a3 = n krep x_f/(1 + K_hc p^2 (n-1))  n += 1
    dilution     a4 = gd n                           n -= 1
Drift F = (a3 - a4, B a1 - a2); Jacobian A = dF/d(n, x_T) at the fixed point;
diffusion D = diag(a3 + a4, E[k^2] a1 + a2); covariance S from A S + S A^T + D = 0.

Used by compute_tables.py (target lna_comparison), which writes data/lna_comparison.csv;
reads data/loss_rate_sweep.csv (the exact matched-mean sweep). Fig. 4 is drawn from it.
Runtime: a few seconds (no CME solves).
"""
import os
import sys
import numpy as np
from scipy.optimize import brentq
from scipy.linalg import solve_continuous_lyapunov
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iteron_cme  # noqa: E402

LN2 = np.log(2.0)
BASE = dict(kd=3.0, B=2.0, gf=LN2, beta=1.0, krep=0.05, gd=LN2, K=4.0, nH=2.0, nmin=1, K_hc=0.5)


def xf_of(n, T, m, kd):
    """Deterministic free initiator: root of x + n m x/(kd+x) = T (unique, increasing in T)."""
    M = n * m
    # quadratic x^2 + (kd + M - T) x - kd T = 0, positive root
    b = kd + M - T
    return 0.5 * (-b + np.sqrt(b * b + 4.0 * kd * T))


def xf_grad(n, T, m, kd):
    """(dx_f/dn, dx_f/dT) by implicit differentiation of T = x + n m x/(kd+x)."""
    x = xf_of(n, T, m, kd)
    dTdx = 1.0 + n * m * kd / (kd + x) ** 2
    dxdT = 1.0 / dTdx
    dxdn = -m * x / (kd + x) * dxdT
    return dxdn, dxdT


def h_fn(x, K, nH):
    return 1.0 if K is None else 1.0 / (1.0 + (x / K) ** nH)


def propensities(n, T, m, kx, P):
    x = xf_of(n, T, m, P["kd"])
    p = x / (P["kd"] + x)
    a1 = n * kx * h_fn(x, P["K"], P["nH"])
    a2 = P["gf"] * x + P["beta"] * P["gf"] * (T - x)
    a3 = n * P["krep"] * x / (1.0 + P["K_hc"] * p * p * (n - 1.0))
    a4 = P["gd"] * n
    return a1, a2, a3, a4


def drift(n, T, m, kx, P):
    a1, a2, a3, a4 = propensities(n, T, m, kx, P)
    return np.array([a3 - a4, P["B"] * a1 - a2])


def fixed_point(m, kx, P):
    """Solve in x_f: replication balance gives n(x), then synthesis-loss balance in x alone.
    Returns list of all positive roots (n*, x_T*, x_f*) found on a fine scan."""
    kd, krep, gd, Khc = P["kd"], P["krep"], P["gd"], P["K_hc"]
    B, gf, beta = P["B"], P["gf"], P["beta"]

    def n_of(x):
        p = x / (kd + x)
        if Khc == 0:
            return np.nan
        return 1.0 + (krep * x / gd - 1.0) / (Khc * p * p)

    def g(x):  # B a1 - a2 with n = n(x)
        n = n_of(x); p = x / (kd + x)
        return B * n * kx * h_fn(x, P["K"], P["nH"]) - (gf * x + beta * gf * n * m * p)

    if Khc == 0:
        # replication balance pins x* = gd/krep; then n from the x_T balance (linear in n)
        x = gd / krep; p = x / (kd + x)
        n = gf * x / (B * kx * h_fn(x, P["K"], P["nH"]) - beta * gf * m * p)
        return [(n, x + n * m * p, x)] if n > 0 else []
    xlo = gd / krep * (1 + 1e-12)          # n(x) >= 1 requires krep x >= gd
    xs = np.geomspace(xlo, 1e5, 4000)
    gs = g(xs)
    roots = []
    for i in range(len(xs) - 1):
        if np.sign(gs[i]) != np.sign(gs[i + 1]):
            x = brentq(g, xs[i], xs[i + 1], xtol=1e-14, rtol=1e-15)
            n = n_of(x); p = x / (kd + x)
            roots.append((n, x + n * m * p, x))
    return roots


def jacobian(n, T, m, kx, P, analytic=True):
    if not analytic:
        e = 1e-6
        J = np.zeros((2, 2))
        for j, (dn, dT) in enumerate(((e * n, 0.0), (0.0, e * T))):
            J[:, j] = (drift(n + dn, T + dT, m, kx, P) - drift(n - dn, T - dT, m, kx, P)) / (2 * (dn + dT))
        return J
    kd, K, nH = P["kd"], P["K"], P["nH"]
    x = xf_of(n, T, m, kd); p = x / (kd + x)
    xn, xT = xf_grad(n, T, m, kd)
    h = h_fn(x, K, nH)
    hp = 0.0 if K is None else -nH * (x / K) ** (nH - 1) / K * h * h
    dp = kd / (kd + x) ** 2
    den = 1.0 + P["K_hc"] * p * p * (n - 1.0)
    # replication a3 = n krep x / den
    a3 = n * P["krep"] * x / den
    da3_dx = n * P["krep"] / den - a3 / den * P["K_hc"] * 2 * p * dp * (n - 1.0)
    da3_dn_explicit = P["krep"] * x / den - a3 / den * P["K_hc"] * p * p
    # synthesis a1 = n kx h(x); loss a2 = gf x + beta gf (T - x)
    da1_dx = n * kx * hp
    da2_dx = P["gf"] * (1.0 - P["beta"])
    A = np.empty((2, 2))
    A[0, 0] = da3_dn_explicit + da3_dx * xn - P["gd"]
    A[0, 1] = da3_dx * xT
    A[1, 0] = P["B"] * (kx * h + da1_dx * xn) - da2_dx * xn
    A[1, 1] = P["B"] * da1_dx * xT - da2_dx * xT - P["beta"] * P["gf"]
    return A


def within_var(n, T, m, kd, K, nH):
    """Var(x_f | x_T, M) from the exact conditional binding law, bilinearly interpolated
    between the integer (n, x_T) lattice points around the LNA fixed point."""
    n0, T0 = int(np.floor(n)), int(np.floor(T))
    fn, fT = n - n0, T - T0
    out = 0.0
    for dn, wn in ((0, 1 - fn), (1, fn)):
        V = iteron_cme.cond_moments_cached((n0 + dn) * m, kd, T0 + 1, K, nH)[3]
        out += wn * ((1 - fT) * V[T0] + fT * V[T0 + 1])
    return out


def lna(m, kx, P=BASE):
    """LNA moments at the (unique) stable fixed point. Returns dict."""
    roots = fixed_point(m, kx, P)
    if len(roots) != 1:
        raise RuntimeError(f"m={m}: {len(roots)} fixed points")
    n, T, x = roots[0]
    A = jacobian(n, T, m, kx, P)
    a1, a2, a3, a4 = propensities(n, T, m, kx, P)
    Ek2 = P["B"] * (2 * P["B"] - 1)
    D = np.diag([a3 + a4, Ek2 * a1 + a2])
    ev = np.linalg.eigvals(A)
    S = solve_continuous_lyapunov(A, -D)
    g = np.array(xf_grad(n, T, m, P["kd"]))
    Vxf_lin = g @ S @ g
    Vw = within_var(n, T, m, P["kd"], P["K"], P["nH"])
    return dict(n=n, T=T, xf=x, A=A, D=D, eig=ev, S=S, Vn=S[0, 0], CV2n=S[0, 0] / n ** 2,
                CV2xf_lin=Vxf_lin / x ** 2, CV2xf_tot=(Vxf_lin + Vw) / x ** 2, Vxf_within=Vw,
                Vxf_lin=Vxf_lin)


def loss_gauss_closed(mu, var):
    """<2^(1-n)> for n ~ N(mu, var) on the real line."""
    return 2.0 ** (1.0 - mu) * np.exp(LN2 ** 2 * var / 2.0)


def gauss_discrete(mu, var, nmin=1, nmax=400):
    k = np.arange(nmin, nmax + 1)
    w = np.exp(-(k - mu) ** 2 / (2 * var)); w /= w.sum()
    Em = w @ k; V = w @ k ** 2 - Em ** 2
    return dict(L=float(w @ 2.0 ** (1.0 - k)), mean=Em, CV2=V / Em ** 2)


def match_kx_lna(m, P=BASE, target=5.0):
    """kx that puts the LNA (mean-field) copy number at `target`. At fixed n* the
    replication balance krep x = gd (1 + K_hc p^2 (n*-1)) fixes x_f* independently of m and
    kx, so the synthesis-loss balance gives kx in closed form (linear in m)."""
    kd, krep, gd, Khc = P["kd"], P["krep"], P["gd"], P["K_hc"]
    x = brentq(lambda x: krep * x - gd * (1 + Khc * (x / (kd + x)) ** 2 * (target - 1)),
               1e-9, 1e6, xtol=1e-14)
    p = x / (kd + x)
    return (P["gf"] * x + P["beta"] * P["gf"] * target * m * p) / (P["B"] * target * h_fn(x, P["K"], P["nH"]))


# ----------------------------------------------------------------------------- driver
def no_handcuff_eigs(m, kx, P=BASE):
    """Jacobian eigenvalues of the K_hc = 0 model at its (unique) fixed point, and the
    unit unstable direction in log coordinates (d ln n, d ln x_T)."""
    P0 = dict(P); P0["K_hc"] = 0.0
    n, T, x = fixed_point(m, kx, P0)[0]
    A = jacobian(n, T, m, kx, P0)
    ev, V = np.linalg.eig(A)
    i = int(np.argmax(ev.real))
    v = V[:, i].real * np.array([1 / n, 1 / T]); v /= np.linalg.norm(v); v *= np.sign(v[0])
    return dict(n=n, lam_max=float(ev.real.max()), lam_min=float(ev.real.min()),
                det=float(np.linalg.det(A)), dir_lnn=v[0], dir_lnT=v[1])


def kx_for_n_noHC(m, n, P=BASE):
    """K_hc = 0: x_f* = gd/krep, and the x_T balance is linear in n."""
    x = P["gd"] / P["krep"]; p = x / (P["kd"] + x)
    return (P["gf"] * x / n + P["beta"] * P["gf"] * m * p) / (P["B"] * h_fn(x, P["K"], P["nH"]))


def build_table(data_dir):
    import pandas as pd
    ex = pd.read_csv(os.path.join(data_dir, "loss_rate_sweep.csv"))
    rows = []
    for _, e in ex.iterrows():
        m, kx = int(e.m), float(e.kx)
        r = lna(m, kx)
        dg = gauss_discrete(r["n"], r["Vn"])
        km = match_kx_lna(m); rm = lna(m, km); dgm = gauss_discrete(rm["n"], rm["Vn"])
        nh_tab = no_handcuff_eigs(m, kx)
        nh_5 = no_handcuff_eigs(m, kx_for_n_noHC(m, 5.0))
        row = dict(m=m, kx=kx,
                   exact_En=e.En, exact_CV2n=e.CV2n, exact_CV2xf=e.CV2xf, exact_Exf=e.Exf, exact_L=e.L,
                   lna_n=r["n"], lna_xT=r["T"], lna_xf=r["xf"], lna_CV2n=r["CV2n"],
                   lna_CV2xf_lin=r["CV2xf_lin"], lna_CV2xf_tot=r["CV2xf_tot"],
                   lna_Vxf_within=r["Vxf_within"],
                   lna_L_closed=loss_gauss_closed(r["n"], r["Vn"]), lna_L_discrete=dg["L"],
                   lna_eig1=float(np.sort(r["eig"].real)[1]), lna_eig2=float(np.sort(r["eig"].real)[0]),
                   lnaM_kx=km, lnaM_n=rm["n"], lnaM_CV2n=rm["CV2n"], lnaM_CV2xf_lin=rm["CV2xf_lin"],
                   lnaM_CV2xf_tot=rm["CV2xf_tot"],
                   lnaM_L_closed=loss_gauss_closed(rm["n"], rm["Vn"]), lnaM_L_discrete=dgm["L"],
                   noHC_n_at_kx=nh_tab["n"], noHC_lam_max_at_kx=nh_tab["lam_max"],
                   noHC_lam_min_at_kx=nh_tab["lam_min"],
                   noHC_kx_n5=kx_for_n_noHC(m, 5.0), noHC_lam_max_n5=nh_5["lam_max"],
                   noHC_lam_min_n5=nh_5["lam_min"], noHC_det_n5=nh_5["det"],
                   noHC_unstable_dlnn_n5=nh_5["dir_lnn"], noHC_unstable_dlnT_n5=nh_5["dir_lnT"])
        for k in ("CV2n", "L"):
            row[f"relerr_{k}"] = (row["lna_" + ("CV2n" if k == "CV2n" else "L_closed")] / e[k] - 1)
        row["relerr_L_discrete"] = dg["L"] / e.L - 1
        row["relerr_CV2xf_tot"] = r["CV2xf_tot"] / e.CV2xf - 1
        row["relerr_CV2xf_lin"] = r["CV2xf_lin"] / e.CV2xf - 1
        row["relerr_mean"] = r["n"] / e.En - 1
        row["relerrM_CV2n"] = rm["CV2n"] / e.CV2n - 1
        row["relerrM_CV2xf_tot"] = rm["CV2xf_tot"] / e.CV2xf - 1
        row["relerrM_L_closed"] = row["lnaM_L_closed"] / e.L - 1
        row["relerrM_L_discrete"] = dgm["L"] / e.L - 1
        row["relerr_kx_matched"] = km / kx - 1
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    print(build_table(os.path.join(here, "data"))[["m", "exact_CV2n", "lna_CV2n", "lnaM_CV2n", "exact_L",
          "lnaM_L_closed", "lnaM_L_discrete", "exact_CV2xf", "lnaM_CV2xf_tot", "lna_n"]].round(4).to_string())
