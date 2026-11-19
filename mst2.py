# MST-CP v1.1 harness: configurable variants (V10 is the v1.0 configuration), ordered audits and
# the oracle O1.

from common import *
from multiprocessing import Pool
NTS = 100, 200, 300
NREP = int(os.environ.get('NREP', 20))
NRFF = 64
LAMS_OLD = np.logspace(-1, 3, 9)
LAMS_NEW = np.logspace(-1, 5, 13)
# Variants: penalty grid and optional shrinkage of the drift fit toward the source mean residual
# c0 by gam in [0, 1] ('slope': leave-one-out regression slope; otherwise leave-one-out R^2).
CFG = {
    'v10': dict(lams=LAMS_OLD, shrink=None),
    'g1': dict(lams=LAMS_NEW, shrink=None),
    'v11': dict(lams=LAMS_NEW, shrink=os.environ.get('SHRINK', 'slope'))
}


# qstrict, pick_m0, group and the micro-cohort score are as in mst.py.
def qstrict(s):
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - ALPHA)))
    return np.inf if n == 0 or k > n else np.sort(s)[k - 1]


def pick_m0(n):
    for m0 in (8, 5, 4):
        if n // m0 >= 15:
            return m0
    return 4


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


# As in mst.py, with a configurable penalty grid and optional shrinkage (gam) of the drift fit.
class Ridge:
    def __init__(self, Phi, e, rows, kind, s2, lams=LAMS_OLD, shrink=None, c0=0.0):
        self.kind = kind
        P = Phi[rows]
        y = e[rows]
        n = len(rows)
        self.shrink = shrink
        self.c0 = c0
        self.gam = 1.0
        self.gam_r2 = np.nan
        self.gam_slope = np.nan
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
            for lam in lams:
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
                lam = lams[-1]
                Ai = np.linalg.inv(G + lam * np.diag(pen))
                w = Ai @ b
                h = np.einsum('ij,jk,ik->i', P, Ai, P)
                res = y - P @ w
                df = h.sum()
                best = 0, lam, Ai, w, h, res, df
            _, self.lam, self.Ai, self.w, h, res, df = best
            self.sig2 = float(res @ res / max(n - df, 1.0))
            self.loo = res / np.maximum(1 - h, 0.05)
            pl = y - self.loo
            a_ = pl - c0
            b_ = y - c0
            self.gam_r2 = float(np.clip(1 - np.mean(self.loo ** 2) / max(np.mean(b_ ** 2), 1e-12), 0, 1))
            self.gam_slope = float(np.clip(a_ @ b_ / max(a_ @ a_, 1e-12), 0, 1))
            if shrink:
                self.gam = self.gam_slope if shrink == 'slope' else self.gam_r2
                self.loo = y - (c0 + self.gam * (pl - c0))
        self.kappa = float(np.clip(self.sig2 / s2[rows].mean(), 0.05, 50))
        self.rows = rows

    def bhat(self, Pb):
        if self.kind == 'glob':
            return np.full(len(Pb), self.mu)
        return Pb @ self.w if not self.shrink else self.c0 + self.gam * (Pb @ self.w - self.c0)

    def var(self, Pb):
        if self.kind == 'glob':
            return np.full(len(Pb), self.sig2 / self.n1)
        return self.gam ** 2 * self.sig2 * np.einsum('ij,jk,ik->i', Pb, self.Ai, Pb)


def prep(d, s, seed, ordered):
    X, y, task, fn, order = load(d)
    sp = make_split(X, fn, order, d, s, seed)
    tr, cal, tc, te = sp['tr'], sp['cal'], sp['tc'], sp['te']
    n, p = X.shape
    # key: time order, or position along the shift variable counted from the source side.
    key = None
    if ordered and s != 'IID':
        spec = sp['spec']
        key = np.asarray(order) if spec['kind'] == 'time' else (
            lambda v: v if spec['dirn'] == 'lo2hi' else -v
        )(X[:, fn.index(spec['var'])])
        tg = np.concatenate([tc, te])
        tg = tg[np.argsort(key[tg], kind='stable')]
        b_ = int(0.25 * len(tg))
        tc, te = tg[:b_], tg[b_:]
    mdl = (HR if task == 'reg' else HC)(
        max_iter=200, early_stopping=False, random_state=seed
    ).fit(X[tr], y[tr])
    f = mdl.predict(X) if task == 'reg' else mdl.predict_proba(X)[:, 1]
    e = y - f
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-09
    Z = np.clip((X - mu) / sd, -5, 5)
    ysrc = y[np.concatenate([tr, cal])]
    cap = float(ysrc.max() - ysrc.min())
    c_s = float(e[cal].mean())
    nb = NearestNeighbors(n_neighbors=31).fit(Z[cal])
    ii = nb.kneighbors(Z[cal], return_distance=False)[:, 1:]
    v = e[cal][ii].var(1) + 1e-06
    sg = HR(max_iter=60, max_leaf_nodes=8, random_state=seed).fit(X[cal], np.log(v))
    s2 = np.exp(np.clip(sg.predict(X), np.log(1e-05), None))
    rf = np.random.RandomState(seed + 99)
    zf = np.hstack([Z, np.clip((f - f[tr].mean()) / (f[tr].std() + 1e-09), -5, 5)[:, None]])
    Wr = rf.randn(zf.shape[1], NRFF)
    bph = rf.rand(NRFF) * 2 * np.pi
    Phi = np.hstack([np.ones((n, 1)), zf, np.sqrt(2.0 / NRFF) * np.cos(zf @ Wr + bph)])
    return dict(
        X=X,
        y=y,
        fn=fn,
        sp=sp,
        tr=tr,
        cal=cal,
        tc=tc,
        te=te,
        n=n,
        f=f,
        e=e,
        Z=Z,
        cap=cap,
        c_s=c_s,
        s2=s2,
        Phi=Phi,
        key=key
    )


def run_mst2(args):
    d, s, seed = args
    t0 = time.time()
    ordered = bool(os.environ.get('ORDERED')) and s != 'IID'
    P = prep(d, s, seed, ordered)
    # Expose the prepared arrays as module globals for the nested helpers.
    globals().update(
        {
            k: P[k]
            for k in ('X', 'y', 'tr', 'cal', 'tc', 'te', 'n', 'f', 'e', 'Z', 'cap', 'c_s', 's2', 'Phi', 'key')
        }
    )
    ones = np.ones(n)
    cte = cells_for(X[te], Z[te], f[te], y[te], ones[te], seed + 3, 200)
    ctc = cells_for(X[tc], Z[tc], f[tc], y[tc], ones[tc], seed + 2, 300)
    ccal = cells_for(X[cal], Z[cal], f[cal], y[cal], ones[cal], seed + 1, 300)
    cell = {}
    for (mode, m), T in cte.items():
        I = T['I']
        Pb = Phi[te][I].mean(1)
        s2b = s2[te][I].mean(1)
        Co = ctc[mode, m]
        q_o1 = qconf(np.abs(Co['yb'] - Co['fb']))
        o1 = np.mean(winkler(T['fb'] - q_o1, T['fb'] + q_o1, T['yb']))
        Cc = ccal[mode, m]
        cell[mode, m] = dict(
            T=T,
            Pb=Pb,
            s2b=s2b,
            o1=o1,
            q_o1=q_o1,
            q3=qconf(np.abs(Cc['yb'] - Cc['fb'])),
            q3c=qconf(np.abs(Cc['yb'] - Cc['fb'] - c_s))
        )
    # Ordered audits: test cohorts from each third of the ordered test rows.
    tcell = {}
    if ordered:
        ts = te[np.argsort(key[te], kind='stable')]
        parts = np.array_split(np.arange(len(ts)), 3)
        for k, pi in enumerate(parts):
            tk = ts[pi]
            cc = cells_for(
                X[tk], Z[tk], f[tk], y[tk], ones[tk], seed + 11 + k, 100, modes=('knn', 'slice', 'decile')
            )
            for (mode, m), T in cc.items():
                I = T['I']
                tcell[k, mode, m] = dict(T=T, Pb=Phi[tk][I].mean(1), s2b=s2[tk][I].mean(1), o1=1.0)

    def ev(c, ctr, hw):
        T = c['T']
        yb = T['yb']
        hw0 = np.asarray(hw, float) * np.ones_like(yb)
        inf = float(np.mean(~np.isfinite(hw0)))
        hwc = np.minimum(hw0, cap)
        L, U = ctr - hwc, ctr + hwc
        return float(np.mean((yb >= L) & (yb <= U))), float(np.mean(winkler(L, U, yb)) / c['o1']), inf

    # (variant, configuration, D_T1 / D_T2 split): 'rand' uses random halves, 'chrono' the earlier
    # half as D_T1. M0_MST is the v1.1 configuration and V10 the v1.0 one; FWD adds forward
    # chaining.
    VARS = [
        ('M0_MST', 'v11', 'chrono'),
        ('V10', 'v10', 'rand'),
        ('V11_rand', 'v11', 'rand'),
        ('V10_chrono', 'v10', 'chrono'),
        ('V11_gridonly_chrono', 'g1', 'chrono')
    ] + ([('FWD', 'v11', 'fwd')] if os.environ.get('FWD') else []) if ordered else [('M0_MST', 'v11', 'rand'), ('V10', 'v10', 'rand'), ('V11_gridonly', 'g1', 'rand')]
    res = {}
    diag = {}
    info = {}
    rng_d = np.random.RandomState(seed * 100 + 7)
    for rep in range(NREP):
        perm = rng_d.permutation(len(tc))
        for nT in NTS:
            if nT > len(tc):
                continue
            rows = tc[perm[:nT]]
            h = nT // 2
            n2 = nT - h
            m0 = pick_m0(n2)
            m0T = pick_m0(nT)
            rows_o = rows[np.argsort(key[rows], kind='stable')] if ordered else rows
            SPL = {'rand': (rows[:h], rows[h:]), 'chrono': (rows_o[:h], rows_o[h:])}
            rg = np.random.RandomState(seed * 100000 + rep * 10 + nT // 100)
            rg2 = np.random.RandomState(seed * 100000 + rep * 10 + nT // 100 + 777)
            # The random split is drawn first from rg (the v1.0 random-number order); the
            # chronological split uses rg2.
            GG = {}
            for sp_name, rr in (('rand', SPL['rand']), ('chrono', SPL['chrono'])):
                if sp_name == 'chrono' and (not ordered):
                    continue
                r1_, r2_ = rr
                R = rg if sp_name == 'rand' else rg2
                G1 = {md: group(r1_, m0, md, Z, X, f, R) for md in MODES}
                G2 = {md: group(r2_, m0, md, Z, X, f, R) for md in MODES}
                GG[sp_name] = G1, G2

            def micro_scores(M, groups, tau, size):
                if not groups:
                    return np.array([])
                eb = np.array([e[g].mean() for g in groups])
                Pb_ = np.array([Phi[g].mean(0) for g in groups])
                sb = np.array([s2[g].mean() for g in groups])
                return np.abs(eb - M.bhat(Pb_)) / np.sqrt(M.kappa * sb / size + M.var(Pb_) + tau)

            FIT = {}
            for vn, cn, sn in VARS:
                if sn == 'fwd':
                    continue
                r1_, r2_ = SPL[sn]
                G1, G2 = GG[sn]
                M = Ridge(Phi, e, r1_, 'ridge', s2, CFG[cn]['lams'], CFG[cn]['shrink'], c_s)
                lo = np.zeros(n)
                lo[r1_] = M.loo
                tau = {
                    md: (lambda v_: max(0.0, float(np.mean(v_))) if v_ else 0.0)(
                        [lo[g].mean() ** 2 - M.kappa * s2[g].mean() / m0 for g in G1[md]]
                    )
                    for md in MODES
                }
                q = {md: qstrict(micro_scores(M, G2[md], tau[md], m0)) for md in MODES}
                FIT[vn] = M, tau, q
            # Forward chaining: blocks 2 to 4 of the ordered audit rows are scored by models
            # fitted on the earlier blocks; the pooled scores give one threshold per mode.
            if ordered and os.environ.get('FWD'):
                rg3 = np.random.RandomState(seed * 100000 + rep * 10 + nT // 100 + 1555)
                blocks = np.array_split(rows_o, 4)
                cfg = CFG['v11']
                qs = {md: [] for md in MODES}
                for j in (1, 2, 3):
                    trn = np.concatenate(blocks[:j])
                    Mj = Ridge(Phi, e, trn, 'ridge', s2, cfg['lams'], cfg['shrink'], c_s)
                    loj = np.zeros(n)
                    loj[trn] = Mj.loo
                    for md in MODES:
                        Ga = group(trn, m0, md, Z, X, f, rg3)
                        Gb = group(blocks[j], m0, md, Z, X, f, rg3)
                        tj = (lambda v_: max(0.0, float(np.mean(v_))) if v_ else 0.0)(
                            [loj[g].mean() ** 2 - Mj.kappa * s2[g].mean() / m0 for g in Ga]
                        )
                        qs[md].append(micro_scores(Mj, Gb, tj, m0))
                Mf = Ridge(Phi, e, rows_o, 'ridge', s2, cfg['lams'], cfg['shrink'], c_s)
                lof = np.zeros(n)
                lof[rows_o] = Mf.loo
                tauF = {}
                for md in MODES:
                    Gf = group(rows_o, m0, md, Z, X, f, rg3)
                    tauF[md] = (lambda v_: max(0.0, float(np.mean(v_))) if v_ else 0.0)(
                        [lof[g].mean() ** 2 - Mf.kappa * s2[g].mean() / m0 for g in Gf]
                    )
                FIT['FWD'] = Mf, tauF, {md: qstrict(np.concatenate(qs[md])) for md in MODES}
            MR, tauR, qR = FIT['M0_MST']
            sn0 = VARS[0][2]
            r1, r2 = SPL[sn0]
            G1, G2 = GG[sn0]
            MG = Ridge(Phi, e, r1, 'glob', s2)
            looG = np.zeros(n)
            looG[r1] = MG.loo
            tauG = {
                md: (lambda v_: max(0.0, float(np.mean(v_))) if v_ else 0.0)(
                    [looG[g].mean() ** 2 - MG.kappa * s2[g].mean() / m0 for g in G1[md]]
                )
                for md in MODES
            }
            qG = {md: qstrict(micro_scores(MG, G2[md], tauG[md], m0)) for md in MODES}

            def pre(M, md):
                eb = np.array([e[g].mean() for g in G2[md]])
                Pb_ = np.array([Phi[g].mean(0) for g in G2[md]])
                sb = np.array([s2[g].mean() for g in G2[md]])
                return eb, Pb_, sb

            qR_not = {}
            qR_lite = {}
            for md in MODES:
                eb, Pb_, sb = pre(MR, md)
                qR_not[md] = qstrict(np.abs(eb - MR.bhat(Pb_)) / np.sqrt(MR.kappa * sb / m0 + MR.var(Pb_)))
                qR_lite[md] = qstrict(np.abs(eb - MR.bhat(Pb_)) / np.sqrt(MR.kappa * sb / m0))
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
            shares = []
            # Methods as in mst.py, plus O1_oracle and the VARS variants.
            for (md, m), c in cell.items():
                T = c['T']
                fb, Pb, s2b = T['fb'], c['Pb'], c['s2b']
                out = {}
                out['O1_oracle'] = ev(c, fb, c['q_o1'])
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

                def mst(M, q, tau, ctr_b):
                    return ev(c, fb + ctr_b, q * np.sqrt(M.kappa * s2b / m + M.var(Pb) + tau))

                out['M0_MST'] = mst(MR, qR[md], tauR[md], bR)
                for vn, _, _ in VARS[1:]:
                    Mv, tv, qv = FIT[vn]
                    out[vn] = mst(Mv, qv[md], tv[md], Mv.bhat(Pb))
                out['M1_global_shift'] = mst(MG, qG[md], tauG[md], MG.bhat(Pb))
                out['M2_no_tau'] = mst(MR, qR_not[md], 0.0, bR)
                out['M3_lite'] = ev(c, fb + bR, qR_lite[md] * np.sqrt(MR.kappa * s2b / m))
                out['M6_agnostic_mix'] = mst(MR, qmix, tau_mix, bR)
                out['M7_random_grouping'] = mst(MR, qrnd, tauR['random'], bR)
                g4 = group(r2, m, md, Z, X, f, rg)
                out['M4_naive_disjoint'] = mst(MR, qstrict(micro_scores(MR, g4, tauR[md], m)), tauR[md], bR)
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
                # Share of the squared studentiser from the drift-variance term at m = 100.
                if m == 100 and md in ('knn', 'slice', 'decile'):
                    ev_v = MR.var(Pb)
                    shares.append(float(np.mean(ev_v / (MR.kappa * s2b / m + ev_v + tauR[md]))))
            # Coverage and Winkler ratio by third of the test period (ordered runs, n_T = 200).
            if ordered and nT == 200:
                for (k, md, m), c in tcell.items():
                    T = c['T']
                    fb, Pb, s2b = T['fb'], c['Pb'], c['s2b']
                    o = {'T4_pointwise_minkowski': ev(c, fb, q_pt), 'S0_B3_source': ev(c, fb, np.inf)}
                    for vn, _, _ in VARS:
                        Mv, tv, qv = FIT[vn]
                        o[vn] = ev(
                            c, fb + Mv.bhat(Pb), qv[md] * np.sqrt(Mv.kappa * s2b / m + Mv.var(Pb) + tv[md])
                        )
                    o.pop('S0_B3_source')
                    for kk, (cv, wr, inf) in o.items():
                        r = res.setdefault('terc', {}).setdefault(f'{k}|{md}|{m}', {}).setdefault(kk, dict(cov=[], wr=[]))
                        r['cov'].append(cv)
                        r['wr'].append(wr)
            I_ = info.setdefault(
                str(nT), dict(m0=m0, K=len(G2['random']), lam={}, gam={}, tau=[], kappa=[], q=[], share=[])
            )
            for vn in FIT:
                I_['lam'].setdefault(vn, []).append(float(FIT[vn][0].lam))
                I_['gam'].setdefault(vn, []).append(float(FIT[vn][0].gam))
            I_.setdefault('gam_r2', []).append(float(FIT['M0_MST'][0].gam_r2))
            I_.setdefault('gam_slope', []).append(float(FIT['M0_MST'][0].gam_slope))
            I_['tau'].append(float(np.mean(list(tauR.values()))))
            I_['kappa'].append(MR.kappa)
            I_['q'].append(float(np.mean([qR[md] for md in ('knn', 'slice', 'decile')])))
            I_['share'].append(float(np.mean(shares)))
            if rep == 0 and nT == 200 and (not ordered):
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
    meta = dict(
        d=d,
        s=s,
        seed=seed,
        ordered=ordered,
        n_tc=int(len(tc)),
        n_te=int(len(te)),
        secs=time.time() - t0,
        shrink=CFG['v11']['shrink']
    )
    json.dump(
        dict(meta=meta, res=res, diag=diag, info=info),
        open(
            f"{HERE}/results/V11{('O' if ordered else 'R')}{os.environ.get('TAG', '')}_{d}_{s}_{seed}.json",
            'w'
        )
    )
    return f'{d} {s} {seed} {time.time() - t0:.0f}s'


# Usage: python mst2.py <comma-separated seeds> [dataset]; TAG, NREP, NP, SHRINK, ORDERED and FWD
# come from the environment.
if __name__ == '__main__':
    seeds = [int(x) for x in sys.argv[1].split(',')]
    only = sys.argv[2] if len(sys.argv) > 2 else None
    jobs = [(d, s, sd) for d, s in workloads() for sd in seeds if only is None or d == only]
    if os.environ.get('ORDERED'):
        jobs = [j for j in jobs if j[1] != 'IID']
    with Pool(int(os.environ.get('NP', 12))) as pl:
        for r in pl.imap_unordered(run_mst2, jobs):
            pass
