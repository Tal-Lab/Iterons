"""Stationary CME with EXPLICIT iteron-binding kinetics, state (n, x_T, b).

Validates the fast-binding (conditional-equilibrium) reduction in iteron_cme.py.

State: n = copy number (nmin..nmax, reflected at both ends as in iteron_cme), x_T = total
initiator (0..xTmax), b = bound initiator, 0 <= b <= min(x_T, n*m). Free x_f = x_T - b.

Reactions from (n, x_T, b)
  binding      kon * x_f * (n m - b)                      b -> b+1
  unbinding    koff * b                                   b -> b-1          kd = koff/kon
  synthesis    n kx h(x_f), h = 1/(1+(x_f/K)^nH)           x_T -> x_T + k, k ~ Geom{1,2..}, mean B
  free decay   gf x_f                                     x_T -> x_T - 1
  bound decay  beta gf b                                  x_T -> x_T - 1, b -> b - 1
  replication  n krep x_f / (1 + K_hc p(x_f)^2 (n-1))      n -> n+1 (new sites empty)
               p(x) = x/(kd+x)
  dilution     gd n                                       n -> n-1, b -> b - j,
               j ~ Hypergeom(population n m, successes b, draws m); the j released
               molecules join the free pool, x_T unchanged.
Bursts overflowing xTmax are lumped at xTmax (as in iteron_cme).

Geometric bursts are represented exactly with one auxiliary variable per state:
along a line of fixed (n, b) the burst inflow S(t) = sum_{s<t} a(s) pi(s) q (1-q)^(t-s-1)
obeys S(t) = (1-q) S(t-1) + q a(t-1) pi(t-1), and the lumped inflow into xTmax is S/q.
This keeps the linear system at ~10 nonzeros per state instead of ~50.

Time unit: one generation when gd = ln 2. The time-scale separation of
data/timescale_separation.csv (T_gen / residence time) is then koff itself in model units,
i.e. koff/gd = separation / ln 2.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.special import gammaln


def pardiso_solver(A, rhs):
    """Direct solve with MKL PARDISO (METIS nested dissection). SuperLU's fill on this
    3D lattice is band-like and infeasible beyond ~1e4 states."""
    import pypardiso
    return pypardiso.spsolve(A.tocsr(), rhs)


def _hypergeom_pmf(N, K, nd):
    """pmf of j ~ Hypergeom(N population, K successes, nd draws), j = 0..min(K, nd)."""
    j = np.arange(0, min(K, nd) + 1)
    ok = (nd - j <= N - K)
    j = j[ok]
    lp = (gammaln(K + 1) - gammaln(j + 1) - gammaln(K - j + 1)
          + gammaln(N - K + 1) - gammaln(nd - j + 1) - gammaln(N - K - nd + j + 1)
          - (gammaln(N + 1) - gammaln(nd + 1) - gammaln(N - nd + 1)))
    return j, np.exp(lp)


def default_pivot(g, kd, n0=5, xf0=39.0):
    """A high-probability state to pin: n = n0, x_f near its typical value, b at the mode of
    the conditional-equilibrium law. Pinning a near-zero-probability state (e.g. b = 0 at
    large m) inflates the solution by many orders of magnitude and loses precision."""
    n0 = min(n0, g.nmax); M = n0 * g.m
    b = int(round(M * xf0 / (kd + xf0)))
    t = min(int(round(xf0)) + b, g.xTmax)
    return int(g.idx(n0, t, min(b, t, M)))


class Grid:
    def __init__(self, m, nmin, nmax, xTmax):
        self.m, self.nmin, self.nmax, self.xTmax = m, nmin, nmax, xTmax
        ns, ts, bs = [], [], []
        for n in range(nmin, nmax + 1):
            for t in range(xTmax + 1):
                bmax = min(t, n * m)
                ns.append(np.full(bmax + 1, n)); ts.append(np.full(bmax + 1, t))
                bs.append(np.arange(bmax + 1))
        self.n = np.concatenate(ns); self.t = np.concatenate(ts); self.b = np.concatenate(bs)
        self.N = len(self.n)
        # dense lookup (n, t, b) -> index, -1 if invalid
        self.lut = -np.ones((nmax + 2, xTmax + 2, nmax * m + 2), dtype=np.int64)
        self.lut[self.n, self.t, self.b] = np.arange(self.N)

    def idx(self, n, t, b):
        return self.lut[n, t, b]


def build_and_solve(m, kd, kx, B, gf, beta, krep, gd, K, nH, K_hc, koff,
                    nmin=1, nmax=25, xTmax=200, grid=None, return_P=False,
                    pivot=None, permc="COLAMD", solver=None):
    g = grid if grid is not None else Grid(m, nmin, nmax, xTmax)
    n, t, b = g.n, g.t, g.b
    N = g.N
    kon = koff / kd
    xf = (t - b).astype(float)
    R, C, V = [], [], []          # generator entries Q[dst, src] (column = source)

    def add(src, dst, rate):
        k = rate > 0
        src, dst, rate = src[k], dst[k], rate[k]
        R.append(dst); C.append(src); V.append(rate)
        R.append(src); C.append(src); V.append(-rate)

    allidx = np.arange(N)
    # binding
    a = kon * xf * (n * m - b)
    s = a > 0
    add(allidx[s], g.idx(n[s], t[s], b[s] + 1), a[s])
    # unbinding
    s = b > 0
    add(allidx[s], g.idx(n[s], t[s], b[s] - 1), koff * b[s].astype(float))
    # free decay
    s = (xf > 0)
    add(allidx[s], g.idx(n[s], t[s] - 1, b[s]), gf * xf[s])
    # bound decay
    s = b > 0
    add(allidx[s], g.idx(n[s], t[s] - 1, b[s] - 1), beta * gf * b[s].astype(float))
    # replication (new plasmid's sites empty)
    s = n < g.nmax
    p = xf / (kd + xf)
    arep = n * krep * xf / (1.0 + K_hc * p ** 2 * (n - 1))
    add(allidx[s], g.idx(n[s] + 1, t[s], b[s]), arep[s])
    # dilution with hypergeometric release of the bound molecules on the lost plasmid
    for nn in range(g.nmin + 1, g.nmax + 1):
        for bb in range(0, min(g.xTmax, nn * m) + 1):
            sel = np.where((n == nn) & (b == bb))[0]
            if len(sel) == 0:
                continue
            js, pj = _hypergeom_pmf(nn * m, bb, m)
            for j, pp in zip(js, pj):
                if pp < 1e-300:
                    continue
                add(sel, g.idx(np.full(len(sel), nn - 1), t[sel], np.full(len(sel), bb - j)),
                    np.full(len(sel), gd * nn * pp))
    # synthesis: outflow on the diagonal; inflow through auxiliary S variables
    asyn = n * kx / (1.0 + (xf / K) ** nH)
    # a burst from x_T = xTmax is lumped onto itself (self-loop): no outflow there
    R.append(allidx); C.append(allidx); V.append(-np.where(t < g.xTmax, asyn, 0.0))
    q = 1.0 / B
    Sidx = N + allidx                           # S variable for each state
    # pi-equation rows: + c * S(state)
    top = (t == g.xTmax)
    hasS = t > b                                # S(t=b) = 0 on each (n,b) line
    cS = np.where(top, 1.0 / q, 1.0)
    R.append(allidx[hasS]); C.append(Sidx[hasS]); V.append(cS[hasS])
    # S-equation rows: S(t) - (1-q) S(t-1) - q a(t-1) pi(t-1) = 0  (t > b);  S = 0 (t = b)
    R.append(Sidx); C.append(Sidx); V.append(np.ones(N))
    prv = np.where(hasS, g.lut[n, np.maximum(t - 1, 0), b], -1)
    ok = hasS & (prv >= 0)
    okS = ok & (t - 1 > b)                       # S(t-1) exists and is nonzero-able
    R.append(Sidx[okS]); C.append(N + prv[okS]); V.append(-(1 - q) * np.ones(okS.sum()))
    R.append(Sidx[ok]); C.append(prv[ok]); V.append(-q * asyn[prv[ok]])
    R = np.concatenate(R); C = np.concatenate(C); V = np.concatenate(V)
    # normalisation: the generator is singular (rank N-1 on the pi block). Replace ONE
    # balance equation (at an interior pivot state k) by pi_k = 1, solve, renormalise.
    # A single-entry row keeps the sparsity; a dense sum(pi)=1 row ruins the LU ordering.
    k = int(pivot) if pivot is not None else default_pivot(g, kd)
    keep = R != k
    R = np.concatenate([R[keep], [k]]); C = np.concatenate([C[keep], [k]])
    V = np.concatenate([V[keep], [1.0]])
    A = coo_matrix((V, (R, C)), shape=(2 * N, 2 * N)).tocsc()
    rhs = np.zeros(2 * N); rhs[k] = 1.0
    x = spsolve(A, rhs, permc_spec=permc) if solver is None else solver(A, rhs)
    res = A @ x - rhs
    resid = float(np.abs(res).max() / np.abs(A.diagonal() * x).max())
    pi = x[:N]
    neg = float(-pi[pi < 0].sum())
    pi = np.maximum(pi, 0.0); pi /= pi.sum()
    nv = n.astype(float)
    En = pi @ nv; Vn = pi @ nv ** 2 - En ** 2
    Exf = pi @ xf; Vxf = pi @ xf ** 2 - Exf ** 2
    Eb = pi @ b
    L = pi @ 2.0 ** (1 - nv)
    out = dict(m=m, koff=koff, N=N, En=En, CV2n=Vn / En ** 2, Exf=Exf, CV2xf=Vxf / Exf ** 2,
               Eb=Eb, L=L, neg_mass=neg, resid=resid,
               edge_n=float(pi[n == g.nmax].sum()), edge_T=float(pi[t == g.xTmax].sum()),
               edge_n_lo=float(pi[n == g.nmin].sum()))
    if return_P:
        out["pi"] = pi; out["grid"] = g
    return out


# ------------------------------------------------------------------ reduced (2D) models
def reduced(m, kd, kx, B, gf, beta, krep, gd, K, nH, K_hc, nmin=1, nmax=25, xTmax=200,
            closure="ratio"):
    """Fast-binding reduction on (n, x_T), vectorised.

    closure="ratio": replication n krep E[x_f|.] / (1 + K_hc E[p^2|.] (n-1))  (iteron_cme.py)
    closure="exact": replication n krep E[x_f / (1 + K_hc p^2 (n-1)) | .]     (exact fast
                     limit of the explicit-binding model above)
    """
    nvals = np.arange(nmin, nmax + 1); XX = xTmax + 1; NN = len(nvals)
    q = 1.0 / B
    tv = np.arange(XX)
    tabs = {}
    for i, nn in enumerate(nvals):
        M = nn * m
        Exf = np.zeros(XX); Exb = np.zeros(XX); Eh = np.zeros(XX); Erep = np.zeros(XX)
        Ep2 = np.zeros(XX); Exf2 = np.zeros(XX)
        for tt in range(XX):
            bb = np.arange(min(tt, M) + 1)
            lw = (gammaln(tt + 1) - gammaln(tt - bb + 1) + gammaln(M + 1) - gammaln(M - bb + 1)
                  - gammaln(bb + 1) - bb * np.log(kd))
            w = np.exp(lw - lw.max()); w /= w.sum()
            x = tt - bb; p = x / (kd + x)
            Exf[tt] = w @ x; Exb[tt] = w @ bb; Eh[tt] = w @ (1 / (1 + (x / K) ** nH))
            Ep2[tt] = w @ p ** 2; Exf2[tt] = w @ x ** 2.0
            Erep[tt] = w @ (x / (1 + K_hc * p ** 2 * (nn - 1)))
        tabs[nn] = (Exf, Exb, Eh, Ep2, Exf2, Erep)
    rows, cols, vals = [], [], []

    def add(src, dst, rate):
        k = rate > 0
        rows.append(dst[k]); cols.append(src[k]); vals.append(rate[k])
        rows.append(src[k]); cols.append(src[k]); vals.append(-rate[k])
    for i, nn in enumerate(nvals):
        Exf, Exb, Eh, Ep2, Exf2, Erep = tabs[nn]
        base = i * XX
        src = base + tv
        a = nn * kx * Eh
        # geometric bursts, explicit (small system), lumped at xTmax
        for tt in range(XX):
            k = np.arange(1, xTmax - tt)
            if len(k):
                pk = q * (1 - q) ** (k - 1)
                rows.append(base + tt + k); cols.append(np.full(len(k), base + tt)); vals.append(a[tt] * pk)
            if tt < xTmax:
                rows.append([base + xTmax]); cols.append([base + tt]); vals.append([a[tt] * (1 - q) ** (xTmax - tt - 1)])
                rows.append([base + tt]); cols.append([base + tt]); vals.append([-a[tt]])
        s = tv > 0
        add(src[s], src[s] - 1, gf * Exf[s] + beta * gf * Exb[s])
        if i < NN - 1:
            if closure == "ratio":
                r = nn * krep * Exf / (1 + K_hc * Ep2 * (nn - 1))
            else:
                r = nn * krep * Erep
            add(src, src + XX, r)
        if i > 0:
            add(src, src - XX, np.full(XX, gd * nn))
    rows = np.concatenate([np.atleast_1d(r) for r in rows]).astype(np.int64)
    cols = np.concatenate([np.atleast_1d(c) for c in cols]).astype(np.int64)
    vals = np.concatenate([np.atleast_1d(v) for v in vals])
    A = coo_matrix((vals, (rows, cols)), shape=(NN * XX, NN * XX)).tolil()
    A[0, :] = 1.0
    rhs = np.zeros(NN * XX); rhs[0] = 1
    pi = spsolve(A.tocsr(), rhs); pi = np.maximum(pi, 0); pi /= pi.sum()
    P = pi.reshape(NN, XX); pn = P.sum(1)
    En = pn @ nvals; Vn = pn @ nvals ** 2.0 - En ** 2
    Exf = sum(P[i] @ tabs[nn][0] for i, nn in enumerate(nvals))
    Exf2 = sum(P[i] @ tabs[nn][4] for i, nn in enumerate(nvals))
    L = pn @ 2.0 ** (1 - nvals)
    return dict(En=En, CV2n=Vn / En ** 2, Exf=Exf, CV2xf=(Exf2 - Exf ** 2) / Exf ** 2, L=L,
                edge_n=float(pn[-1]), edge_T=float(P[:, -1].sum()))
