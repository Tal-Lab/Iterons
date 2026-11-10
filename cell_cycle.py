"""The model of iteron_cme.py with discrete cell division instead of continuous dilution.

Used for the check in Supplemental Sec. S4. Within a generation (time 0 to 1) there is no
dilution: initiator is synthesized in bursts at rate n k_x E[h(x_f) | x_T] and plasmids
replicate at the handcuffed rate of Eq. (5), with binding at conditional equilibrium as in
the main model. At division every plasmid copy and every initiator molecule goes to the
followed daughter independently with probability 1/2 (random segregation). Daughters that
receive no plasmid are discarded, which conditions on retention as the reflection at n = 1
does in the main model. Removal of initiator by dilution alone (beta = 1 in the main model)
corresponds here to no turnover within the cycle.

The periodic steady state is the fixed point of the map "propagate one generation, then
divide", found by iteration; propagation uses Crank-Nicolson steps with one sparse LU
factorization. Moments are reported time-averaged over the cycle, at birth, and just
before division; the loss rate is taken from the distribution just before division.
"""
import numpy as np
from scipy.sparse import coo_matrix, identity
from scipy.sparse.linalg import splu
from scipy.stats import binom

import iteron_cme


def generator(m, kx, nmax, xTmax, kd, B, krep, K, nH, K_hc):
    """Generator of synthesis and replication on n = 1..nmax, x_T = 0..xTmax (no dilution)."""
    nvals = np.arange(1, nmax + 1)
    NN, XX = len(nvals), xTmax + 1
    ks, pk = iteron_cme._burst_pmf(B)
    t = np.arange(XX)
    rows, cols, vals = [], [], []

    def add(src, dst, rate):
        ok = (rate > 0) & (dst != src)
        rows.extend([dst[ok], src[ok]]); cols.extend([src[ok], src[ok]])
        vals.extend([rate[ok], -rate[ok]])

    tabs = {}
    for i, n in enumerate(nvals):
        Exf, Exb, Eh, Vxf, Ep2 = iteron_cme.cond_moments_cached(int(n) * m, kd, xTmax, K, nH)
        tabs[int(n)] = (Exf, Vxf)
        base = i * XX
        for kk, pp in zip(ks, pk):          # bursts overflowing xTmax are lumped at xTmax
            add(base + t, base + np.minimum(t + int(kk), xTmax), n * kx * Eh * pp)
        if i < NN - 1:
            add(base + t, base + XX + t, n * krep * Exf / (1.0 + K_hc * Ep2 * (n - 1)))
    Q = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(NN * XX, NN * XX)).tocsc()
    return Q, nvals, XX, tabs


def periodic(m, kx, nmax, xTmax, nsteps=256, tol=1e-11, maxit=500, P0=None, **par):
    """Periodic steady state and its moments. par: kd, B, krep, K, nH, K_hc."""
    Q, nvals, XX, tabs = generator(m, kx, nmax, xTmax, **par)
    NN, N = len(nvals), len(nvals) * (xTmax + 1)
    dt = 1.0 / nsteps
    I = identity(N, format="csc")
    lu = splu((I - 0.5 * dt * Q).tocsc())
    Bp = (I + 0.5 * dt * Q).tocsr()
    Bn = binom.pmf(np.arange(0, nmax + 1)[None, :], nvals[:, None], 0.5)    # n -> k copies
    Bx = binom.pmf(np.arange(XX)[None, :], np.arange(XX)[:, None], 0.5)     # x_T -> y
    if P0 is None:
        P = np.zeros((NN, XX)); P[min(3, NN - 1), min(40, XX - 1)] = 1.0
    else:
        P = P0 / P0.sum()
    for it in range(maxit):
        v = P.ravel()
        for _ in range(nsteps):
            v = lu.solve(Bp @ v)
        D = Bn.T @ np.maximum(v.reshape(NN, XX), 0.0) @ Bx
        Pn = D[1:, :] / D[1:, :].sum()
        err = np.abs(Pn - P).sum()
        P = Pn
        if err < tol:
            break
    v = P.ravel(); acc = 0.5 * P
    for s in range(nsteps):
        v = lu.solve(Bp @ v)
        acc = acc + (0.5 if s == nsteps - 1 else 1.0) * v.reshape(NN, XX)
    Pavg, Ppre = acc / nsteps, v.reshape(NN, XX)

    def n_moments(Pm):
        pn = Pm.sum(1) / Pm.sum(); En = pn @ nvals
        return pn, En, (pn @ nvals ** 2 - En ** 2) / En ** 2

    pn_a, En, CV2n = n_moments(Pavg)
    _, En_b, CV2n_b = n_moments(P)
    pn_p, En_p, CV2n_p = n_moments(Ppre)
    W = Pavg / Pavg.sum()          # free initiator, law of total variance, cycle average
    Exf = sum(W[i] @ tabs[int(n)][0] for i, n in enumerate(nvals))
    within = sum(W[i] @ tabs[int(n)][1] for i, n in enumerate(nvals))
    cm2 = sum(W[i] @ tabs[int(n)][0] ** 2 for i, n in enumerate(nvals))
    return dict(En=float(En), CV2n=float(CV2n), En_birth=float(En_b), CV2n_birth=float(CV2n_b),
                En_pre=float(En_p), CV2n_pre=float(CV2n_p), Exf=float(Exf),
                CV2xf=float((within + cm2 - Exf ** 2) / Exf ** 2),
                L=float(pn_p @ (2.0 ** (1 - nvals))), edge_n=float(pn_p[-1]),
                edge_T=float(Ppre[:, -1].sum() / Ppre.sum()), iterations=it + 1,
                residual=float(err), P=P)


def matched(m, kx0, nmax, xTmax, target=5.0, tol=1e-6, **par):
    """k_x giving a cycle-averaged mean copy number `target` (secant in ln k_x)."""
    x0 = np.log(kx0)
    r0 = periodic(m, np.exp(x0), nmax, xTmax, **par); f0 = r0["En"] - target
    x1 = x0 + (0.05 if f0 < 0 else -0.05)
    r1 = periodic(m, np.exp(x1), nmax, xTmax, P0=r0["P"], **par); f1 = r1["En"] - target
    for _ in range(40):
        if abs(f1) < tol:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x2 = min(max(x2, x1 - 0.5), x1 + 0.5)
        x0, f0 = x1, f1
        x1 = x2
        r1 = periodic(m, np.exp(x1), nmax, xTmax, P0=r1["P"], **par); f1 = r1["En"] - target
    else:
        raise RuntimeError(f"matching did not converge at m={m}")
    r1.pop("P")
    return dict(m=m, kx=float(np.exp(x1)), **r1)
