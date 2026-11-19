# Shared core of the final-round experiments: swappable base predictor, optional quadratic drift
# basis (L2), held-out micro-cohort evaluation and windowed audits.

import os, sys
# kNN tie-breaking depends on the OpenMP thread count, so bit-reproducibility needs these single-
# thread settings (the script re-executes itself when they are unset).
PIN = dict(
    OMP_NUM_THREADS='1',
    VECLIB_MAXIMUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',
    MKL_NUM_THREADS='1',
    PYTHONHASHSEED='0'
)
if any((os.environ.get(k) != v for k, v in PIN.items())):
    os.execve(
        sys.executable,
        [sys.executable] + (['-I'] if sys.flags.isolated else []) + sys.argv,
        dict(os.environ, **PIN)
    )
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
import json, time
import numpy as np
from scipy import stats
import common, mst
from common import load, make_split, cells_for, winkler, HR, HC, NearestNeighbors, ALPHA, MODES
from mst import Ridge, group, qstrict, pick_m0, NRFF, NREP
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge as SkRidge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
qconf_ = common.qconf
STRUCT = 'knn', 'slice', 'decile'
MS = 25, 50, 100
SEEDS = 3, 4, 5
WL10 = [(d, s) for d in ('adult', 'bike', 'california', 'house_sales', 'wine_quality') for s in ('S1', 'S2')]


def mk_pred(fam, task, seed):
    if fam == 'HGB':
        return (HR if task == 'reg' else HC)(max_iter=200, early_stopping=False, random_state=seed)
    if task == 'reg':
        if fam == 'MLP':
            return TransformedTargetRegressor(
                regressor=make_pipeline(
                    StandardScaler(),
                    MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed)
                ),
                transformer=StandardScaler()
            )
        if fam == 'RF':
            return RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1)
        return TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), SkRidge(alpha=1.0)), transformer=StandardScaler()
        )
    if fam == 'MLP':
        return make_pipeline(
            StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed)
        )
    if fam == 'RF':
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


# As in mst.run_mst and aux_revisions.run_aux_job for basis='base'; 'quad' (L2) adds squares of
# the standardised features and prediction to the drift features.
def prep(d, s, seed, fam='HGB', basis='base', ordered=False):
    X, y, task, fn, order = load(d)
    sp = make_split(X, fn, order, d, s, seed)
    tr, cal, tc, te = sp['tr'], sp['cal'], sp['tc'], sp['te']
    n, p = X.shape
    mdl = mk_pred(fam, task, seed).fit(X[tr], y[tr])
    f = mdl.predict(X) if task == 'reg' else mdl.predict_proba(X)[:, 1]
    e = y - f
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-09
    Z = np.clip((X - mu) / sd, -5, 5)
    ysrc = y[np.concatenate([tr, cal])]
    cap = float(ysrc.max() - ysrc.min())
    nb = NearestNeighbors(n_neighbors=31).fit(Z[cal])
    ii = nb.kneighbors(Z[cal], return_distance=False)[:, 1:]
    v = e[cal][ii].var(1) + 1e-06
    # The local noise model is always gradient boosting, fitted on the residuals of the predictor
    # in use.
    sg = HR(max_iter=60, max_leaf_nodes=8, random_state=seed).fit(X[cal], np.log(v))
    s2 = np.exp(np.clip(sg.predict(X), np.log(1e-05), None))
    rf = np.random.RandomState(seed + 99)
    ft = np.clip((f - f[tr].mean()) / (f[tr].std() + 1e-09), -5, 5)
    zf = np.hstack([Z, ft[:, None]])
    Wr = rf.randn(zf.shape[1], NRFF)
    bph = rf.rand(NRFF) * 2 * np.pi
    Phi = np.hstack([np.ones((n, 1)), zf, np.sqrt(2.0 / NRFF) * np.cos(zf @ Wr + bph)])
    if basis == 'quad':
        Q = zf ** 2
        Q = np.clip((Q - Q[tr].mean(0)) / (Q[tr].std(0) + 1e-09), -5, 5)
        Phi = np.hstack([Phi, Q])
    ones = np.ones(n)
    P = dict(
        d=d,
        s=s,
        seed=seed,
        fam=fam,
        basis=basis,
        X=X,
        y=y,
        task=task,
        order=order,
        sp=sp,
        tr=tr,
        cal=cal,
        tc=tc,
        te=te,
        f=f,
        e=e,
        Z=Z,
        s2=s2,
        Phi=Phi,
        ones=ones,
        cap=cap,
        n=n
    )
    return P


def make_cells(P, te_rows=None, tc_rows=None):
    X, Z, f, y, ones, Phi, s2, seed = P['X'], P['Z'], P['f'], P['y'], P['ones'], P['Phi'], P['s2'], P['seed']
    te = P['te'] if te_rows is None else te_rows
    tc = P['tc'] if tc_rows is None else tc_rows
    cte = cells_for(X[te], Z[te], f[te], y[te], ones[te], seed + 3, 200, modes=STRUCT)
    ctc = cells_for(X[tc], Z[tc], f[tc], y[tc], ones[tc], seed + 2, 300, modes=STRUCT)
    cell = {}
    for (mode, m), T in cte.items():
        I = T['I']
        Co = ctc[mode, m]
        q_o1 = qconf_(np.abs(Co['yb'] - Co['fb']))
        cell[mode, m] = dict(
            T=T,
            Pb=Phi[te][I].mean(1),
            s2b=s2[te][I].mean(1),
            q_o1=float(q_o1),
            o1=float(np.mean(winkler(T['fb'] - q_o1, T['fb'] + q_o1, T['yb'])))
        )
    return cell


def fit_mst(P, rows, rg):
    Phi, e, s2, Z, X, f, n = P['Phi'], P['e'], P['s2'], P['Z'], P['X'], P['f'], P['n']
    h = len(rows) // 2
    r1, r2 = rows[:h], rows[h:]
    n2 = len(r2)
    m0 = pick_m0(n2)
    MR = Ridge(Phi, e, r1, 'ridge', s2)
    G1 = {md: group(r1, m0, md, Z, X, f, rg) for md in MODES}
    G2 = {md: group(r2, m0, md, Z, X, f, rg) for md in MODES}
    looR = np.zeros(n)
    looR[r1] = MR.loo

    def tau2(md):
        v_ = [looR[g].mean() ** 2 - MR.kappa * s2[g].mean() / m0 for g in G1[md]]
        return max(0.0, float(np.mean(v_))) if v_ else 0.0

    tauR = {md: tau2(md) for md in MODES}
    F = dict(MR=MR, m0=m0, G1=G1, G2=G2, tauR=tauR, r1=r1, r2=r2, n2=n2)
    F['cal'] = {md: micro_scores(P, F, G2[md], md, m0) for md in STRUCT}
    F['q'] = {md: qstrict(F['cal'][md]) for md in STRUCT}
    return F


def micro_scores(P, F, groups, md, size):
    if not groups:
        return np.array([])
    e, Phi, s2, MR = P['e'], P['Phi'], P['s2'], F['MR']
    eb = np.array([e[g].mean() for g in groups])
    Pb_ = np.array([Phi[g].mean(0) for g in groups])
    sb = np.array([s2[g].mean() for g in groups])
    return np.abs(eb - MR.bhat(Pb_)) / np.sqrt(MR.kappa * sb / size + MR.var(Pb_) + F['tauR'][md])


# Per (mode, m): (coverage, Winkler / O1, width ratio to O1, Winkler score, infinite fraction).
def eval_cells(P, F, cell, mult=1.0):
    MR, tauR, cap = F['MR'], F['tauR'], P['cap']
    out = {}
    for (md, m), c in cell.items():
        T = c['T']
        fb, yb = T['fb'], T['yb']
        ctr = fb + MR.bhat(c['Pb'])
        core = np.sqrt(MR.kappa * c['s2b'] / m + MR.var(c['Pb']) + tauR[md])
        hw0 = np.asarray(F['q'][md] * mult * core, float) * np.ones_like(yb)
        inf = float(np.mean(~np.isfinite(hw0)))
        hwc = np.minimum(hw0, cap)
        L, U = ctr - hwc, ctr + hwc
        out[md, m] = float(np.mean((yb >= L) & (yb <= U))), float(
            np.mean(winkler(L, U, yb)) / c['o1']
        ), float(np.mean(2 * hwc) / (2 * c['q_o1'])), float(np.mean(winkler(L, U, yb))), inf
    return out


def pooled(res):
    a = np.array(list(res.values()))
    return a.mean(0)


def draws_perms(P):
    rng_d = np.random.RandomState(P['seed'] * 100 + 7)
    return [rng_d.permutation(len(P['tc'])) for _ in range(NREP)]


# kappa, tau^2, penalty, effective degrees of freedom and the share of non-averaging variance at m
# = 100.
def diagnostics(P, F, cell):
    MR = F['MR']
    tmean = float(np.mean([F['tauR'][md] for md in STRUCT]))
    out = dict(kappa=float(MR.kappa), tau=tmean, lam=float(MR.lam), sig2=float(MR.sig2))
    sh = []
    for (md, m), c in cell.items():
        if m != 100:
            continue
        rown = MR.kappa * c['s2b'] / m
        ne = MR.var(c['Pb']) + F['tauR'][md]
        sh.append(float(np.mean(ne / (rown + ne))))
    out['nonavg_share_m100'] = float(np.mean(sh))
    return out


# Held-out micro-scale coverage of q_hat on size-m0 cohorts from te: 'greedy' built as in the
# calibration (group() on random pools), 'compact' built with cells_for.
def micro_eval(P, F, tag, npools=4):
    X, Z, f, y, e, Phi, s2, te, m0, n2 = P['X'], P['Z'], P['f'], P['y'], P['e'], P['Phi'], P['s2'], P['te'], F['m0'], F['n2']
    key = '_mc', m0
    if key not in P:
        P[key] = cells_for(
            X[te], Z[te], f[te], y[te], P['ones'][te], P['seed'] + 7, 200, sizes=(m0,), modes=STRUCT
        )
    cm = P[key]
    MR = F['MR']
    out = {}
    for md in STRUCT:
        q = F['q'][md]
        sc = []
        for p_ in range(npools):
            r = np.random.RandomState(tag * 10 + p_)
            pool = r.choice(te, n2, replace=False)
            rg2 = np.random.RandomState(tag * 10 + p_ + 777)
            sc.append(micro_scores(P, F, group(pool, m0, md, Z, X, f, rg2), md, m0))
        sc = np.concatenate(sc)
        I = cm[md, m0]['I']
        rows = te[I]
        eb = e[rows].mean(1)
        Pb_ = Phi[rows].mean(1)
        sb = s2[rows].mean(1)
        sc2 = np.abs(eb - MR.bhat(Pb_)) / np.sqrt(MR.kappa * sb / m0 + MR.var(Pb_) + F['tauR'][md])
        out[md] = dict(
            q=float(q),
            cal=F['cal'][md],
            greedy=sc,
            compact=sc2,
            cov_greedy=float(np.mean(sc <= q)),
            cov_compact=float(np.mean(sc2 <= q))
        )
    return out
