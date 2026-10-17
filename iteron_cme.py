"""Exact stationary CME for an iteron plasmid: copy number coupled to initiator.

State (n, x_T): n = plasmid copy number, x_T = total initiator (free + iteron-bound).
The iteron reservoir co-scales with the replicon, M = n*m, so changing n changes the
reservoir instantaneously.

Adiabatic binding
-----------------
Iteron binding/unbinding is fast relative to synthesis, decay and replication, so
x_b | x_T is at conditional equilibrium with the exact detailed-balance weights

    P(b | x_T, M)  proportional to  C(x_T, b) C(M, b) b! / k_d^b

and every propensity that depends on free initiator is averaged over that law. This is
the same reduction used in decoy_sim.adiabatic_cme, extended to an n-dependent M and to
propensities that are nonlinear in x_f.

Reactions
---------
  synthesis    n * kx * E[h(x_f) | x_T]        x_T -> x_T + Geom(mean B)
  initiator loss   gf * E[x_f|x_T] + gb * E[x_b|x_T]        x_T -> x_T - 1
  replication  n * krep * E[x_f | x_T]         n -> n + 1
  plasmid dilution  gd * n                     n -> n - 1   (reflected at n = nmin)

h is the autorepression function of FREE initiator, h(x) = 1/(1 + (x/K)**nH); nH = 1 is
a monomeric repressor, nH = 2 approximates a dimeric one. gb = beta * gf.

Reflection at n = nmin conditions on plasmid retention: cells that lose the plasmid are
not tracked. State that assumption wherever results are reported.
"""
import numpy as np
from scipy.sparse import coo_matrix, identity
from scipy.sparse.linalg import spsolve


_CACHE = {}


def cond_moments_cached(M, kd, xTmax, K=None, nH=1.0):
    """Memoised cond_moments. The tables depend only on the reservoir and the repression
    function, not on any rate, so a parameter sweep over rates reuses them."""
    key = (int(M), float(kd), int(xTmax), None if K is None else float(K), float(nH))
    if key not in _CACHE:
        _CACHE[key] = cond_moments(*key)
    return _CACHE[key]


def cond_moments(M, kd, xTmax, K=None, nH=1.0):
    """Conditional expectations of free/bound initiator given x_T, for one reservoir size.

    Returns (Exf, Exb, Eh) each of length xTmax+1:
      Exf[t] = E[x_f | x_T=t],  Exb[t] = E[x_b | x_T=t],  Eh[t] = E[h(x_f) | x_T=t],
      Vxf[t] = Var(x_f | x_T=t),  Ep2[t] = E[p(x_f)^2 | x_T=t] with p = x_f/(k_d+x_f).
    With K=None, Eh is all ones (no autorepression).
    """
    M = int(M); xTmax = int(xTmax)
    Exf = np.zeros(xTmax + 1); Exb = np.zeros(xTmax + 1); Eh = np.ones(xTmax + 1)
    Vxf = np.zeros(xTmax + 1)        # Var(x_f | x_T) = Var(x_b | x_T)
    Ep2 = np.zeros(xTmax + 1)        # E[p(x_f)^2 | x_T], p = x_f/(k_d+x_f): handcuffing
    logkd = np.log(kd)
    for t in range(xTmax + 1):
        bmax = min(t, M)
        if bmax == 0:
            Exf[t] = t
            Ep2[t] = (t / (kd + t)) ** 2 if t > 0 else 0.0
            if K is not None:
                Eh[t] = 1.0 / (1.0 + (t / K) ** nH)
            continue
        b = np.arange(bmax + 1)
        i = np.arange(bmax)
        lg = np.zeros(bmax + 1)
        lg[1:] = (np.cumsum(np.log(t - i)) + np.cumsum(np.log(M - i))
                  - np.cumsum(np.log(i + 1.0)) - (i + 1.0) * logkd)
        lg -= lg.max()
        w = np.exp(lg); w /= w.sum()
        xf = t - b
        Exf[t] = w @ xf
        Exb[t] = w @ b
        Vxf[t] = w @ (b * b) - Exb[t] ** 2
        Ep2[t] = w @ ((xf / (kd + xf)) ** 2)
        if K is not None:
            Eh[t] = w @ (1.0 / (1.0 + (xf / K) ** nH))
    return Exf, Exb, Eh, Vxf, Ep2


def _burst_pmf(B, tol=1e-14):
    """Geometric burst on {1,2,...} with mean B."""
    if B <= 1.0:
        return np.array([1]), np.array([1.0])
    q = 1.0 / B
    kmax = max(1, int(np.ceil(np.log(tol) / np.log(1 - q))))
    k = np.arange(1, kmax + 1)
    p = q * (1 - q) ** (k - 1)
    return k, p / p.sum()


def solve(m, kd, kx, B, gf, beta, krep, gd, K=None, nH=1.0,
          nmin=1, nmax=24, xTmax=None, tol=1e-14, M_fix=0, K_hc=0.0, trans_hc=0.0,
          hc_loaded=False, hc_mref=1.0, hc_cap=0.0, hc_foci=0.0):
    """Stationary distribution of (n, x_T). Returns a dict of moments and the pmf.

    M_fix adds a reservoir that does NOT scale with n (iterons supplied in trans).
    K_hc > 0 adds handcuffing: origin pairing is bimolecular in origins, so the
    per-plasmid initiation rate is divided by 1 + K_hc * E[p^2|x_T] * (n-1), taken at
    quasi-equilibrium in the pairing reaction."""
    if xTmax is None:
        xTmax = int(40 + 6 * nmax * kx * B / max(gf, 1e-12))
    nvals = np.arange(nmin, nmax + 1)
    NN, XX = len(nvals), xTmax + 1
    # M_fix: a reservoir on a DIFFERENT replicon, so it does not co-scale with n.
    # This is the architecture of an incompatibility assay (iterons supplied in trans).
    tabs = {int(n): cond_moments_cached(int(n) * m + int(M_fix), kd, xTmax, K, nH)
            for n in nvals}
    ks, pk = _burst_pmf(B, tol)

    def idx(i, t):
        return i * XX + t

    rows, cols, vals = [], [], []

    def add(src, dst, rate):
        if rate <= 0.0:
            return
        rows.append(dst); cols.append(src); vals.append(rate)
        rows.append(src); cols.append(src); vals.append(-rate)

    for i, n in enumerate(nvals):
        Exf, Exb, Eh, _Vxf, Ep2 = tabs[int(n)]
        for t in range(XX):
            s = idx(i, t)
            # bursty synthesis, gene dosage n
            a_syn = n * kx * Eh[t]
            if a_syn > 0:
                for kk, pp in zip(ks, pk):
                    tt = min(t + int(kk), xTmax)
                    add(s, idx(i, tt), a_syn * pp)
            # initiator loss (free and bound turn over at gf and beta*gf)
            a_loss = gf * Exf[t] + beta * gf * Exb[t]
            if t > 0:
                add(s, idx(i, t - 1), a_loss)
            # replication: free initiator fires the origin, suppressed by handcuffing
            if i < NN - 1:
                # trans arrays can pair with the resident origin too (the classical
                # incompatibility mechanism), counted in units of one cis array so that
                # no new parameter is introduced.
                partners = (n - 1) + (trans_hc * M_fix / m if (m and trans_hc) else 0.0)
                # Spatial models for how many partners an origin can actually reach.
                # hc_foci: plasmids distribute over a FIXED number of clusters, so the
                #   reachable partner count is (n-1)/foci -- still proportional to n.
                # hc_cap:  an origin reaches at most hc_cap neighbours no matter how
                #   many plasmids exist, so above the cap there is no n-dependence.
                if hc_foci:
                    partners = partners / hc_foci
                if hc_cap:
                    partners = min(partners, hc_cap)
                # hc_loaded=False: pairing needs one loaded iteron on each origin, so the
                # propensity goes as p^2 and K_hc absorbs per-site multiplicity.
                # hc_loaded=True: propensity goes as the square of the NUMBER of loaded
                # iterons, (m p)^2, normalised at m = hc_mref so the two agree there.
                scale = ((m / hc_mref) ** 2 if (hc_loaded and m) else 1.0)
                hc = 1.0 + K_hc * scale * Ep2[t] * partners
                add(s, idx(i + 1, t), n * krep * Exf[t] / hc)
            # plasmid dilution, reflected at nmin
            if i > 0:
                add(s, idx(i - 1, t), gd * n)

    Q = coo_matrix((vals, (rows, cols)), shape=(NN * XX, NN * XX)).tocsr()
    # stationary: Q pi = 0 with sum pi = 1, imposed by replacing one row
    A = Q.tolil()
    A[0, :] = 1.0
    b = np.zeros(NN * XX); b[0] = 1.0
    pi = spsolve(A.tocsr(), b)
    pi = np.maximum(pi, 0.0); pi /= pi.sum()
    P = pi.reshape(NN, XX)

    pn = P.sum(axis=1)
    pT = P.sum(axis=0)
    tvals = np.arange(XX)
    En = pn @ nvals
    EN2 = pn @ (nvals ** 2.0)
    ET = pT @ tvals
    # free initiator: E[x_f] = sum_{n,t} pi(n,t) E[x_f|t,n]; second moment needs E[x_f^2|t,n],
    # which the moment tables do not carry, so report the conditional-mean surrogate and flag it
    # law of total variance over the joint (n, x_T): within-state binding noise plus
    # the variance of the conditional mean across states
    Exf_mean = sum(P[i] @ tabs[int(n)][0] for i, n in enumerate(nvals))
    E_within = sum(P[i] @ tabs[int(n)][3] for i, n in enumerate(nvals))
    E_cm2    = sum(P[i] @ (tabs[int(n)][0] ** 2) for i, n in enumerate(nvals))
    Vxf_tot  = E_within + (E_cm2 - Exf_mean ** 2)
    return dict(nvals=nvals, P=P, pn=pn, pT=pT,
                En=En, Vn=EN2 - En ** 2, CV2n=(EN2 - En ** 2) / En ** 2,
                ET=ET, VT=pT @ (tvals ** 2.0) - ET ** 2,
                Exf=Exf_mean, Vxf=Vxf_tot, CV2xf=Vxf_tot / Exf_mean ** 2,
                Vxf_within=E_within, Vxf_between=E_cm2 - Exf_mean ** 2, xTmax=xTmax,
                edge_n=float(pn[-1]), edge_T=float(pT[-1]))


def loss_rate(m, kd, kx, B, gf, beta, krep, gd, K=None, nH=1.0,
              nmax=24, xTmax=None, M_fix=0, K_hc=0.0, trans_hc=0.0):
    """Per-unit-time decay rate of the quasi-stationary distribution with n=0 absorbing.

    States n >= 1 only; the n=1 -> n=0 dilution transition leaks probability out. The
    decay rate is -Re(lambda_max) of the restricted generator, so survival over t time
    units is exp(-lambda t). With gd = ln2 per generation, t is in generations.
    """
    from scipy.sparse.linalg import eigs
    if xTmax is None:
        xTmax = int(40 + 6 * nmax * kx * B / max(gf, 1e-12))
    nvals = np.arange(1, nmax + 1)
    NN, XX = len(nvals), xTmax + 1
    tabs = {int(n): cond_moments_cached(int(n) * m + int(M_fix), kd, xTmax, K, nH)
            for n in nvals}
    ks, pk = _burst_pmf(B)
    rows, cols, vals = [], [], []

    def add(src, dst, rate):
        if rate <= 0.0:
            return
        if dst is not None:                      # dst=None => leaves the n>=1 system
            rows.append(dst); cols.append(src); vals.append(rate)
        rows.append(src); cols.append(src); vals.append(-rate)

    for i, n in enumerate(nvals):
        Exf, Exb, Eh, _, Ep2 = tabs[int(n)]
        for t in range(XX):
            s_ = i * XX + t
            a_syn = n * kx * Eh[t]
            if a_syn > 0:
                for kk, pp in zip(ks, pk):
                    add(s_, i * XX + min(t + int(kk), xTmax), a_syn * pp)
            if t > 0:
                add(s_, i * XX + t - 1, gf * Exf[t] + beta * gf * Exb[t])
            if i < NN - 1:
                partners_ = (n - 1) + (trans_hc * M_fix / m if (m and trans_hc) else 0.0)
                add(s_, (i + 1) * XX + t,
                    n * krep * Exf[t] / (1.0 + K_hc * Ep2[t] * partners_))
            add(s_, (i - 1) * XX + t if i > 0 else None, gd * n)

    Q = coo_matrix((vals, (rows, cols)), shape=(NN * XX, NN * XX)).tocsr()
    lam = eigs(Q, k=1, which="LR", return_eigenvectors=False, maxiter=20000, tol=1e-10)
    return float(-np.real(lam[0]))
