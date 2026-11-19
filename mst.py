# MST-CP v1.0 and baselines with random-sample audit labels; one job per (dataset, shift, seed).

from common import *
from multiprocessing import Pool
NTS = 100, 200, 300
NREP = int(os.environ.get('NREP', 20))
LAMS = np.logspace(-1, 3, 9)
NRFF = 64


# The ceil((n + 1)(1 - ALPHA))-th smallest score; +inf if that rank exceeds n.
def qstrict(s):
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - ALPHA)))
    return np.inf if n == 0 or k > n else np.sort(s)[k - 1]


# Largest m0 in (8, 5, 4) that leaves at least 15 micro-cohorts.
def pick_m0(n):
    for m0 in (8, 5, 4):
        if n // m0 >= 15:
            return m0
    return 4


# Disjoint cohorts by mode: random, decile (by prediction), slice (block after sorting by a random
# feature), otherwise kNN around a random centre.
def group(rows, size, mode, Z, X, f, rng):
    rows = np.asarray(rows)
    n = len(rows)
    K = n // size
    if K == 0:
        return []
    if mode == 'random':
        p = rng.permutation(rows)
        return [p[k * size:(k + 1) * size] for k in range(K)]
    if mode == 'decile':
        o = rows[np.argsort(f[rows] + 1e-09 * rng.rand(n))]
        return [o[k * size:(k + 1) * size] for k in range(K)]
    rem = rows.copy()
    out = []
    for _ in range(K):
        if mode == 'slice':
            j = rng.randint(X.shape[1])
            o = rem[np.lexsort((rng.rand(len(rem)), X[rem, j]))]
            s0 = rng.randint(0, len(o) - size + 1)
            blk = o[s0:s0 + size]
        else:
            c = rem[rng.randint(len(rem))]
            blk = rem[np.argsort(((Z[rem] - Z[c]) ** 2).sum(1))[:size]]
        out.append(blk)
        rem = np.setdiff1d(rem, blk)
    return out


# Drift model on audit residuals. 'ridge': penalty chosen by generalised cross-validation (at
# least 5 residual degrees of freedom, else the largest penalty); 'glob': a constant shift.
class Ridge:
    def __init__(self, Phi, e, rows, kind, s2):
        self.kind = kind
        P = Phi[rows]
        y = e[rows]
        n = len(rows)
        if kind == 'glob':
            self.mu = float(y.mean())
            self.sig2 = float(y.var(ddof=1))
            self.n1 = n
            self.loo = (y - self.mu) * n / (n - 1)
            self.lam = np.nan
        else:
            D = P.shape[1]
            pen = np.ones(D)
            pen[0] = 1e-06
            G = P.T @ P
            b = P.T @ y
            best = None
            for lam in LAMS:
                Ai = np.linalg.inv(G + lam * np.diag(pen))
                w = Ai @ b
                h = np.einsum('ij,jk,ik->i', P, Ai, P)
                res = y - P @ w
                df = h.sum()
                if n - df < 5:
                    continue
                g = n * (res @ res) / (n - df) ** 2
                if best is None or g < best[0]:
                    best = g, lam, Ai, w, h, res, df
            if best is None:
                lam = LAMS[-1]
                Ai = np.linalg.inv(G + lam * np.diag(pen))
                w = Ai @ b
                h = np.einsum('ij,jk,ik->i', P, Ai, P)
                res = y - P @ w
                df = h.sum()
                best = 0, lam, Ai, w, h, res, df
            _, self.lam, self.Ai, self.w, h, res, df = best
            self.sig2 = float(res @ res / max(n - df, 1.0))
            # Leave-one-out residuals from the hat diagonal, leverage floored at 0.05.
            self.loo = res / np.maximum(1 - h, 0.05)
        # kappa rescales the local noise model s2 to the audit rows, clipped to [0.05, 50].
        self.kappa = float(np.clip(self.sig2 / s2[rows].mean(), 0.05, 50))
        self.rows = rows

    def bhat(self, Pb):
        return np.full(len(Pb), self.mu) if self.kind == 'glob' else Pb @ self.w

    # Plug-in drift variance sig2 * Pb' Ai Pb.
    def var(self, Pb):
        return np.full(len(Pb), self.sig2 / self.n1) if self.kind == 'glob' else self.sig2 * np.einsum(
            'ij,jk,ik->i', Pb, self.Ai, Pb
        )


def run_mst(args):
    d, s, seed = args
    t0 = time.time()
    X, y, task, fn, order = load(d)
    sp = make_split(X, fn, order, d, s, seed)
    tr, cal, tc, te = sp['tr'], sp['cal'], sp['tc'], sp['te']
    n, p = X.shape
    # ORDERED: audit labels from the earliest 25% of the target period, test on the rest.
    if os.environ.get('ORDERED') and sp['spec']['kind'] == 'time':
        tg = np.concatenate([tc, te])
        tg = tg[np.argsort(np.asarray(order)[tg], kind='stable')]
        b_ = int(0.25 * len(tg))
        tc, te = tg[:b_], tg[b_:]
    mdl = (HR if task == 'reg' else HC)(
        max_iter=200, early_stopping=False, random_state=seed
    ).fit(X[tr], y[tr])
    f = mdl.predict(X) if task == 'reg' else mdl.predict_proba(X)[:, 1]
    # cap: source label range, applied to every interval half-width.
    e = y - f
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-09
    Z = np.clip((X - mu) / sd, -5, 5)
    ysrc = y[np.concatenate([tr, cal])]
    cap = float(ysrc.max() - ysrc.min())
    c_s = float(e[cal].mean())
    # Local noise model s2: kNN residual variance on the source calibration rows, smoothed by
    # gradient boosting on the log scale.
    nb = NearestNeighbors(n_neighbors=31).fit(Z[cal])
    ii = nb.kneighbors(Z[cal], return_distance=False)[:, 1:]
    v = e[cal][ii].var(1) + 1e-06
    sg = HR(max_iter=60, max_leaf_nodes=8, random_state=seed).fit(X[cal], np.log(v))
    s2 = np.exp(np.clip(sg.predict(X), np.log(1e-05), None))
    # Drift features: constant, zf (standardised features and prediction) and 64 random Fourier
    # features of zf.
    rf = np.random.RandomState(seed + 99)
    zf = np.hstack([Z, np.clip((f - f[tr].mean()) / (f[tr].std() + 1e-09), -5, 5)[:, None]])
    Wr = rf.randn(zf.shape[1], NRFF)
    bph = rf.rand(NRFF) * 2 * np.pi
    Phi = np.hstack([np.ones((n, 1)), zf, np.sqrt(2.0 / NRFF) * np.cos(zf @ Wr + bph)])
    ones = np.ones(n)
    cte = cells_for(X[te], Z[te], f[te], y[te], ones[te], seed + 3, 200)
    ctc = cells_for(X[tc], Z[tc], f[tc], y[tc], ones[tc], seed + 2, 300)
    ccal = cells_for(X[cal], Z[cal], f[cal], y[cal], ones[cal], seed + 1, 300)
    # o1: Winkler score of the uncorrected labelled-target oracle (conformal cohort threshold from
    # tc), the normaliser. q3, q3c: source-calibrated thresholds without and with the shift c_s.
    cell = {}
    for (mode, m), T in cte.items():
        I = T['I']
        Pb = Phi[te][I].mean(1)
        s2b = s2[te][I].mean(1)
        Co = ctc[mode, m]
        o1 = np.mean(
            winkler(*(lambda q: (T['fb'] - q, T['fb'] + q))(qconf(np.abs(Co['yb'] - Co['fb']))), T['yb'])
        )
        Cc = ccal[mode, m]
        cell[mode, m] = dict(
            T=T,
            Pb=Pb,
            s2b=s2b,
            o1=o1,
            q3=qconf(np.abs(Cc['yb'] - Cc['fb'])),
            q3c=qconf(np.abs(Cc['yb'] - Cc['fb'] - c_s))
        )

    # Coverage, Winkler / o1 and infinite fraction of cohort-mean intervals, capped at the source
    # range. Sum intervals scale by m, so both ratios are unchanged.
    def ev(c, ctr, hw):
        T = c['T']
        yb = T['yb']
        hw0 = np.asarray(hw, float) * np.ones_like(yb)
        inf = float(np.mean(~np.isfinite(hw0)))
        hwc = np.minimum(hw0, cap)
        L, U = ctr - hwc, ctr + hwc
        return float(np.mean((yb >= L) & (yb <= U))), float(np.mean(winkler(L, U, yb)) / c['o1']), inf

    res = {}
    diag = {}
    rng_d = np.random.RandomState(seed * 100 + 7)
    for rep in range(NREP):
        perm = rng_d.permutation(len(tc))
        for nT in NTS:
            if nT > len(tc):
                continue
            # D_T1 (first half) fits the drift and variance; D_T2 (second half) calibrates the
            # micro-cohort threshold.
            rows = tc[perm[:nT]]
            h = nT // 2
            r1, r2 = rows[:h], rows[h:]
            n2 = len(r2)
            m0 = pick_m0(n2)
            m0T = pick_m0(nT)
            rg = np.random.RandomState(seed * 100000 + rep * 10 + nT // 100)
            MR = Ridge(Phi, e, r1, 'ridge', s2)
            MG = Ridge(Phi, e, r1, 'glob', s2)
            G1 = {md: group(r1, m0, md, Z, X, f, rg) for md in MODES}
            G2 = {md: group(r2, m0, md, Z, X, f, rg) for md in MODES}
            looR = np.zeros(n)
            looR[r1] = MR.loo
            looG = np.zeros(n)
            looG[r1] = MG.loo

            # tau^2: mean over D_T1 micro-cohorts of (mean leave-one-out residual)^2 - kappa * s2
            # / m0, floored at 0.
            def tau2(M, loo, md):
                v_ = [loo[g].mean() ** 2 - M.kappa * s2[g].mean() / m0 for g in G1[md]]
                return max(0.0, float(np.mean(v_))) if v_ else 0.0

            tauR = {md: tau2(MR, looR, md) for md in MODES}
            tauG = {md: tau2(MG, looG, md) for md in MODES}

            # |mean residual - drift| / sqrt(kappa * s2 / size + var(drift) + tau).
            def micro_scores(M, groups, tau, size):
                if not groups:
                    return np.array([])
                eb = np.array([e[g].mean() for g in groups])
                Pb_ = np.array([Phi[g].mean(0) for g in groups])
                sb = np.array([s2[g].mean() for g in groups])
                return np.abs(eb - M.bhat(Pb_)) / np.sqrt(M.kappa * sb / size + M.var(Pb_) + tau)

            # Thresholds from D_T2 scores. qR_not drops tau; qR_lite keeps only the noise term
            # (ablations).
            qR = {md: qstrict(micro_scores(MR, G2[md], tauR[md], m0)) for md in MODES}
            qG = {md: qstrict(micro_scores(MG, G2[md], tauG[md], m0)) for md in MODES}
            qR_not = {
                md: qstrict(
                    np.abs(np.array([e[g].mean() for g in G2[md]]) - MR.bhat(np.array([Phi[g].mean(0) for g in G2[md]]))) / np.sqrt(
                        MR.kappa * np.array([s2[g].mean() for g in G2[md]]) / m0 + MR.var(
                            np.array([Phi[g].mean(0) for g in G2[md]])
                        )
                    )
                )
                for md in MODES
            }
            qR_lite = {
                md: qstrict(
                    np.abs(
                        np.array([e[g].mean() for g in G2[md]]) - MR.bhat(
                            np.array([Phi[g].mean(0) for g in G2[md]])
                        )
                    ) / np.sqrt(MR.kappa * np.array([s2[g].mean() for g in G2[md]]) / m0)
                )
                for md in MODES
            }
            tau_mix = float(np.mean(list(tauR.values())))
            Gmix = [g for md in MODES for g in G2[md]]
            qmix = qstrict(micro_scores(MR, Gmix, tau_mix, m0))
            qrnd = qR['random']
            chat = float(e[rows].mean())
            q_pt = qstrict(np.abs(e[rows]))
            q_c3 = qstrict(np.abs(e[r2] - Phi[r2] @ MR.w))
            G0 = {md: group(rows, m0T, md, Z, X, f, rg) for md in MODES}
            q_c1 = {
                md: qstrict(np.abs(np.array([e[g].mean() for g in G0[md]])) * np.sqrt(m0T))
                for md in MODES
            }
            # Methods: S0 source-calibrated; T1, T2 target size-m cohorts (disjoint, random); T3
            # global debiasing; T4, C3 pointwise sums (without, with drift); C1 no-drift micro-
            # cohort; C2 drift centre with source width; M0 MST-CP; M1 global shift; M2 no tau; M3
            # noise only; M4, M5 size-m cohorts from D_T2; M6 pooled modes; M7 random grouping.
            for (md, m), c in cell.items():
                T = c['T']
                fb, Pb, s2b = T['fb'], c['Pb'], c['s2b']
                out = {}
                out['S0_B3_source'] = ev(c, fb, c['q3'])
                g1 = group(rows, m, md, Z, X, f, rg)
                out['T1_B3_target_disjoint'] = ev(
                    c, fb, qstrict(np.array([abs(y[g].mean() - f[g].mean()) for g in g1])) if g1 else np.inf
                )
                Cs = draw(Z[rows], X[rows], f[rows], md, m, 300, rg, {})
                out['T2_B3_target_randsub'] = ev(
                    c, fb, qstrict(np.abs(y[rows][Cs].mean(1) - f[rows][Cs].mean(1)))
                )
                out['T3_global_debiased_B3'] = ev(c, fb + chat, c['q3c'])
                out['T4_pointwise_minkowski'] = ev(c, fb, q_pt)
                out['C1_nodrift_micro_cia'] = ev(c, fb, q_c1[md] / np.sqrt(m))
                bR = MR.bhat(Pb)
                out['C2_localdrift_source_width'] = ev(c, fb + bR, c['q3c'])
                out['C3_driftcorr_pointwise_mink'] = ev(c, fb + bR, q_c3)

                # Cohort-mean half-width q * sqrt(kappa * s2 / m + var(drift) + tau) at the target
                # size m; the interval for the sum is m times this.
                def mst(M, q, tau, ctr_b):
                    return ev(c, fb + ctr_b, q * np.sqrt(M.kappa * s2b / m + M.var(Pb) + tau))

                out['M0_MST'] = mst(MR, qR[md], tauR[md], bR)
                out['M1_global_shift'] = mst(MG, qG[md], tauG[md], MG.bhat(Pb))
                out['M2_no_tau'] = mst(MR, qR_not[md], 0.0, bR)
                out['M3_lite'] = ev(c, fb + bR, qR_lite[md] * np.sqrt(MR.kappa * s2b / m))
                out['M6_agnostic_mix'] = mst(MR, qmix, tau_mix, bR)
                out['M7_random_grouping'] = mst(MR, qrnd, tauR['random'], bR)
                g4 = group(r2, m, md, Z, X, f, rg)
                out['M4_naive_disjoint'] = mst(MR, qstrict(micro_scores(MR, g4, tauR[md], m)), tauR[md], bR)
                # M5 needs m rows in D_T2; otherwise its interval is infinite.
                if m <= n2:
                    Cs2 = draw(Z[r2], X[r2], f[r2], md, m, 300, rg, {})
                    g5 = [r2[i] for i in Cs2]
                    out['M5_naive_randsub'] = mst(
                        MR, qstrict(micro_scores(MR, g5, tauR[md], m)), tauR[md], bR
                    )
                else:
                    out['M5_naive_randsub'] = mst(MR, np.inf, tauR[md], bR)
                for k, (cv, wr, inf) in out.items():
                    r = res.setdefault(str(nT), {}).setdefault(f'{md}|{m}', {}).setdefault(
                        k, dict(cov=[], wr=[], inf=[])
                    )
                    r['cov'].append(cv)
                    r['wr'].append(wr)
                    r['inf'].append(inf)
            res.setdefault(str(nT), {}).setdefault(
                '_info', dict(m0=m0, K=len(G2['random']), lam=[], tau=[], kappa=[])
            )
            res[str(nT)]['_info']['lam'].append(float(MR.lam))
            res[str(nT)]['_info']['tau'].append(float(np.mean(list(tauR.values()))))
            res[str(nT)]['_info']['kappa'].append(MR.kappa)
            # Size-invariance diagnostic: score quantiles on cohorts of several sizes (first draw,
            # n_T = 200).
            if rep == 0 and nT == 200:
                pool = np.setdiff1d(tc, r1)
                for sz in (m0, 25, 50, 100):
                    cc = cells_for(X[pool], Z[pool], f[pool], y[pool], ones[pool], seed + 5, 300, sizes=(sz,))
                    for md in MODES:
                        I = cc[md, sz]['I']
                        rowsC = pool[I]
                        eb = e[rowsC].mean(1)
                        Pb_ = Phi[rowsC].mean(1)
                        sb = s2[rowsC].mean(1)
                        sc = np.abs(eb - MR.bhat(Pb_)) / np.sqrt(MR.kappa * sb / sz + MR.var(Pb_) + tauR[md])
                        diag[f'{md}|{sz}'] = dict(q90=float(np.quantile(sc, 0.9)), mean=float(sc.mean()))
    # meta holds wall-clock time, which differs between runs.
    meta = dict(d=d, s=s, seed=seed, n_tc=int(len(tc)), n_te=int(len(te)), secs=time.time() - t0)
    json.dump(
        dict(meta=meta, res=res, diag=diag),
        open(f"{HERE}/results/MST{os.environ.get('TAG', '')}_{d}_{s}_{seed}.json", 'w')
    )
    return f'{d} {s} {seed} {time.time() - t0:.0f}s'


# Usage: python mst.py <comma-separated seeds> [dataset]; TAG, NREP, NP and ORDERED come from the
# environment.
if __name__ == '__main__':
    seeds = [int(x) for x in sys.argv[1].split(',')]
    only = sys.argv[2] if len(sys.argv) > 2 else None
    jobs = [(d, s, sd) for d, s in workloads() for sd in seeds if only is None or d == only]
    with Pool(int(os.environ.get('NP', 12))) as pl:
        for r in pl.imap_unordered(run_mst, jobs):
            pass
