# Auxiliary checks on seeds 3 to 5 -> results/AUX_REVISIONS.json: A plug-in thresholds (Gaussian,
# Student-t), B original-scale evaluation, C source-range cap activation, D drift-corrected
# oracles, E predictor families.

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
import json, time, hashlib, tempfile, platform
import numpy as np
from scipy import stats
import sklearn, scipy
import common, mst
from common import load, make_split, cells_for, winkler, workloads, HR, HC, NearestNeighbors, ALPHA, MODES
from mst import Ridge, group, qstrict, pick_m0, NTS, NREP, NRFF
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge as SkRidge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
STRUCT = 'knn', 'slice', 'decile'
MS = 25, 50, 100
SEEDS = 3, 4, 5
AUXM = 'M0_MST', 'A_gauss', 'A_t'
MAIN_B = 'S0_B3_source', 'T1_B3_target_disjoint', 'T2_B3_target_randsub', 'T3_global_debiased_B3', 'T4_pointwise_minkowski', 'C3_driftcorr_pointwise_mink', 'C1_nodrift_micro_cia', 'M0_MST'


# Checks A, C and D for one job. M0_MST is recomputed with the random-number order of mst.run_mst
# and must equal the stored arrays (ref_identical).
def run_aux_job(args):
    d, s, seed = args
    t0 = time.time()
    X, y, task, fn, order = load(d)
    sp = make_split(X, fn, order, d, s, seed)
    tr, cal, tc, te = sp['tr'], sp['cal'], sp['tc'], sp['te']
    n, p = X.shape
    mdl = (HR if task == 'reg' else HC)(
        max_iter=200, early_stopping=False, random_state=seed
    ).fit(X[tr], y[tr])
    f = mdl.predict(X) if task == 'reg' else mdl.predict_proba(X)[:, 1]
    e = y - f
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-09
    Z = np.clip((X - mu) / sd, -5, 5)
    ysrc = y[np.concatenate([tr, cal])]
    cap = float(ysrc.max() - ysrc.min())
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
    ones = np.ones(n)
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
            o1=float(np.mean(winkler(T['fb'] - q_o1, T['fb'] + q_o1, T['yb'])))
        )
    zq = float(stats.norm.ppf(1 - ALPHA / 2))
    shifted = s != 'IID'
    # DC_b: drift fitted on a random half of tc, cohorts calibrated on the other half (independent
    # of n_T and the audit draw).
    dcb = {}
    pooledDC = {
        str(nT): {k: dict(cov=[], wr=[], ratio=[]) for k in ('DC_a', 'DC_b')}
        for nT in NTS
    } if shifted else None
    if shifted:
        rb = np.random.RandomState(seed * 100 + 11)
        pb = rb.permutation(len(tc))
        h1, h2 = tc[pb[:len(tc) // 2]], tc[pb[len(tc) // 2:]]
        MRb = Ridge(Phi, e, h1, 'ridge', s2)
        cb = cells_for(X[h2], Z[h2], f[h2], y[h2], ones[h2], seed + 6, 300, modes=STRUCT)
        for (md, m), c in cell.items():
            Cb = cb[md, m]
            sc = np.abs(Cb['yb'] - Cb['fb'] - MRb.bhat(Phi[h2][Cb['I']].mean(1)))
            q = qconf_(sc)
            T = c['T']
            ctr = T['fb'] + MRb.bhat(c['Pb'])
            L, U = ctr - q, ctr + q
            dcb[md, m] = float(np.mean((T['yb'] >= L) & (T['yb'] <= U))), float(
                np.mean(winkler(L, U, T['yb'])) / c['o1']
            )
    pooled = {str(nT): {k: dict(cov=[], wr=[], inf=[]) for k in AUXM} for nT in NTS}
    clip = {}
    chk = {}
    nus = {str(nT): [] for nT in NTS}
    rng_d = np.random.RandomState(seed * 100 + 7)
    for rep in range(NREP):
        perm = rng_d.permutation(len(tc))
        for nT in NTS:
            if nT > len(tc):
                continue
            rows = tc[perm[:nT]]
            h = nT // 2
            r1, r2 = rows[:h], rows[h:]
            n2 = len(r2)
            m0 = pick_m0(n2)
            rg = np.random.RandomState(seed * 100000 + rep * 10 + nT // 100)
            MR = Ridge(Phi, e, r1, 'ridge', s2)
            G1 = {md: group(r1, m0, md, Z, X, f, rg) for md in MODES}
            G2 = {md: group(r2, m0, md, Z, X, f, rg) for md in MODES}
            looR = np.zeros(n)
            looR[r1] = MR.loo

            def tau2(md):
                v_ = [looR[g].mean() ** 2 - MR.kappa * s2[g].mean() / m0 for g in G1[md]]
                return max(0.0, float(np.mean(v_))) if v_ else 0.0

            tauR = {md: tau2(md) for md in MODES}

            def micro_scores(groups, tau, size):
                if not groups:
                    return np.array([])
                eb = np.array([e[g].mean() for g in groups])
                Pb_ = np.array([Phi[g].mean(0) for g in groups])
                sb = np.array([s2[g].mean() for g in groups])
                return np.abs(eb - MR.bhat(Pb_)) / np.sqrt(MR.kappa * sb / size + MR.var(Pb_) + tau)

            qR = {md: qstrict(micro_scores(G2[md], tauR[md], m0)) for md in STRUCT}
            P1 = Phi[r1]
            df = float(np.einsum('ij,jk,ik->i', P1, MR.Ai, P1).sum())
            nu = max(len(r1) - df, 1.0)
            tq = float(stats.t.ppf(1 - ALPHA / 2, nu))
            nus[str(nT)].append(nu)
            acc = {k: [] for k in AUXM}
            wrM = {}
            if shifted:
                pool = np.setdiff1d(tc, r1)
                ca = cells_for(X[pool], Z[pool], f[pool], y[pool], ones[pool], seed + 6, 300, modes=STRUCT)
                # DC_a: drift fitted on D_T1, cohorts calibrated on the rest of tc.
                dca = {}
            for (md, m), c in cell.items():
                T = c['T']
                fb, yb, Pb, s2b = T['fb'], T['yb'], c['Pb'], c['s2b']
                ctr = fb + MR.bhat(Pb)
                core = np.sqrt(MR.kappa * s2b / m + MR.var(Pb) + tauR[md])
                for k, q in (('M0_MST', qR[md]), ('A_gauss', zq), ('A_t', tq)):
                    hw0 = np.asarray(q * core, float) * np.ones_like(yb)
                    inf = float(np.mean(~np.isfinite(hw0)))
                    hwc = np.minimum(hw0, cap)
                    L, U = ctr - hwc, ctr + hwc
                    cv = float(np.mean((yb >= L) & (yb <= U)))
                    wr = float(np.mean(winkler(L, U, yb)) / c['o1'])
                    acc[k].append((cv, wr, inf))
                    cvu = float(np.mean((yb >= ctr - hw0) & (yb <= ctr + hw0)))
                    act = float(np.mean(hw0 > cap))
                    cl = clip.setdefault(str(nT), {}).setdefault(f'{md}|{m}', {}).setdefault(
                        k, dict(act=[], cov_cap=[], cov_uncap=[])
                    )
                    cl['act'].append(act)
                    cl['cov_cap'].append(cv)
                    cl['cov_uncap'].append(cvu)
                    if k == 'M0_MST':
                        wrM[md, m] = wr
                        r = chk.setdefault(str(nT), {}).setdefault(f'{md}|{m}', dict(cov=[], wr=[], inf=[]))
                        r['cov'].append(cv)
                        r['wr'].append(wr)
                        r['inf'].append(inf)
            if shifted:
                for (md, m), c in cell.items():
                    Ca = ca[md, m]
                    sc = np.abs(Ca['yb'] - Ca['fb'] - MR.bhat(Phi[pool][Ca['I']].mean(1)))
                    q = qconf_(sc)
                    T = c['T']
                    ctr = T['fb'] + MR.bhat(c['Pb'])
                    L, U = ctr - q, ctr + q
                    dca[md, m] = float(np.mean((T['yb'] >= L) & (T['yb'] <= U))), float(
                        np.mean(winkler(L, U, T['yb'])) / c['o1']
                    )
                for key, dd in (('DC_a', dca), ('DC_b', dcb)):
                    pooledDC[str(nT)][key]['cov'].append(float(np.mean([v[0] for v in dd.values()])))
                    pooledDC[str(nT)][key]['wr'].append(float(np.mean([v[1] for v in dd.values()])))
                    pooledDC[str(nT)][key]['ratio'].append(float(np.mean([wrM[c_] / dd[c_][1] for c_ in dd])))
            for k in AUXM:
                a = np.array(acc[k])
                pooled[str(nT)][k]['cov'].append(float(a[:, 0].mean()))
                pooled[str(nT)][k]['wr'].append(float(a[:, 1].mean()))
                pooled[str(nT)][k]['inf'].append(float(a[:, 2].mean()))
    ref = json.load(open(f'{HERE}/results/MSTLOCKA_{d}_{s}_{seed}.json'))['res']
    same = all(
        (
            np.array_equal(ref[nT][cc]['M0_MST'][key], chk[nT][cc][key])
            for nT in chk
            for cc in chk[nT]
            for key in ('cov', 'wr', 'inf')
        )
    )
    clipS = {
        nT: {
            cc: {
                k: dict(
                    act_mean=float(np.mean(v_['act'])),
                    act_max=float(np.max(v_['act'])),
                    cov_cap=float(np.mean(v_['cov_cap'])),
                    cov_uncap=float(np.mean(v_['cov_uncap']))
                )
                for k,
                v_ in kk.items()
            }
            for cc,
            kk in cc_.items()
        }
        for nT,
        cc_ in clip.items()
    }
    return dict(
        d=d,
        s=s,
        seed=seed,
        cap=cap,
        ref_identical=bool(same),
        pooled=pooled,
        pooledDC=pooledDC,
        clip=clipS,
        nu_mean={k: float(np.mean(v_)) for k, v_ in nus.items()},
        secs=time.time() - t0
    )


def _mk_reg(fam, seed):
    if fam == 'MLP':
        return TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed)
            ),
            transformer=StandardScaler()
        )
    if fam == 'RF':
        return RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1)
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), SkRidge(alpha=1.0)), transformer=StandardScaler()
    )


def _mk_clf(fam, seed):
    if fam == 'MLP':
        return make_pipeline(
            StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed)
        )
    if fam == 'RF':
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


FAMS = 'MLP', 'RF', 'LIN'


# Check E: swaps the base predictor in mst for the job. The local noise model (max_leaf_nodes)
# keeps gradient boosting.
def run_family(args):
    fam, d, s_, seed = args
    t0 = time.time()
    oHR, oHC, oN, oHere, oTag = mst.HR, mst.HC, mst.NTS, mst.HERE, os.environ.get('TAG')

    def hr(**kw):
        return oHR(**kw) if 'max_leaf_nodes' in kw else _mk_reg(fam, kw['random_state'])

    def hc(**kw):
        return _mk_clf(fam, kw['random_state'])

    try:
        with tempfile.TemporaryDirectory(prefix='aux_fam_') as tmp:
            os.makedirs(tmp + '/results')
            mst.HR, mst.HC, mst.NTS, mst.HERE = hr, hc, (200,), tmp
            os.environ['TAG'] = 'AUXFAM'
            mst.run_mst((d, s_, seed))
            o = json.load(open(f'{tmp}/results/MSTAUXFAM_{d}_{s_}_{seed}.json'))
    finally:
        mst.HR, mst.HC, mst.NTS, mst.HERE = oHR, oHC, oN, oHere
        if oTag is None:
            os.environ.pop('TAG', None)
        else:
            os.environ['TAG'] = oTag
    red = {}
    for me in FAM_METH:
        arrs = {
            key: np.stack(
                [np.array(o['res']['200'][f'{md}|{m}'][me][key]) for md in STRUCT for m in MS], 1
            ).mean(1).tolist()
            for key in ('cov', 'wr', 'inf')
        }
        red[me] = arrs
    return dict(fam=fam, d=d, s=s_, seed=seed, red=red, secs=time.time() - t0)


FAM_METH = 'S0_B3_source', 'T4_pointwise_minkowski', 'C1_nodrift_micro_cia', 'M0_MST'


def qconf_(s_, a=ALPHA):
    return common.qconf(s_, a)


# Check B: mst.run_mst on the untransformed target (exp, expm1), writing to a temporary directory.
def run_raw(jobs):
    orig = mst.load

    def load_raw(name):
        X, y, task, fn, order = orig(name)
        if name == 'house_sales':
            y = np.exp(y)
        elif name == 'bike':
            y = np.expm1(y)
        return X, y, task, fn, order

    out = {}
    old_home = mst.HERE
    old_tag = os.environ.get('TAG')
    with tempfile.TemporaryDirectory(prefix='aux_raw_') as tmp:
        os.makedirs(tmp + '/results')
        mst.load = load_raw
        mst.HERE = tmp
        os.environ['TAG'] = 'AUXRAW'
        try:
            for d, s, seed in jobs:
                mst.run_mst((d, s, seed))
                o = json.load(open(f'{tmp}/results/MSTAUXRAW_{d}_{s}_{seed}.json'))
                red = {}
                for nT in NTS:
                    red[str(nT)] = {}
                    for me in MAIN_B:
                        cv = np.stack(
                            [
                                np.array(o['res'][str(nT)][f'{md}|{m}'][me]['cov'])
                                for md in STRUCT
                                for m in MS
                            ],
                            1
                        )
                        wr = np.stack(
                            [np.array(o['res'][str(nT)][f'{md}|{m}'][me]['wr']) for md in STRUCT for m in MS],
                            1
                        )
                        inf = np.stack(
                            [
                                np.array(o['res'][str(nT)][f'{md}|{m}'][me]['inf'])
                                for md in STRUCT
                                for m in MS
                            ],
                            1
                        )
                        red[str(nT)][me] = dict(
                            cov=cv.mean(1).tolist(), wr=wr.mean(1).tolist(), inf=inf.mean(1).tolist()
                        )
                out[f'{d}_{s}_{seed}'] = red
        finally:
            mst.load = orig
            mst.HERE = old_home
            if old_tag is None:
                os.environ.pop('TAG', None)
            else:
                os.environ['TAG'] = old_tag
    return out


if __name__ == '__main__':
    from multiprocessing import Pool
    t0 = time.time()
    wl = workloads()
    jobs = [(d, s, sd) for d, s in wl for sd in SEEDS]
    with Pool(int(os.environ.get('NP', 7))) as pl:
        res = []
        for r in pl.imap_unordered(run_aux_job, jobs):
            res.append(r)
    res.sort(key=lambda r: (r['d'], r['s'], r['seed']))
    A, C, ident, nu = {}, {}, {}, {}
    for r in res:
        k = f"{r['d']}_{r['s']}"
        A.setdefault(k, {})[str(r['seed'])] = r['pooled']
        C.setdefault(k, {})[str(r['seed'])] = dict(cap=r['cap'], cells=r['clip'])
        ident[f"{k}_{r['seed']}"] = r['ref_identical']
        nu[f"{k}_{r['seed']}"] = r['nu_mean']
    jobsB = [(d, s, sd) for d, s in wl if d in ('house_sales', 'bike') and s != 'IID' for sd in SEEDS]
    Braw = run_raw(jobsB)
    B = {}
    for k, v in Braw.items():
        dd, ss, sd = k.rsplit('_', 2)
        B.setdefault(f'{dd}_{ss}', {})[sd] = v
    D = {}
    for r in res:
        if r['pooledDC'] is not None:
            D.setdefault(f"{r['d']}_{r['s']}", {})[str(r['seed'])] = r['pooledDC']
    shifted_jobs = [(d, s_, sd) for d, s_ in wl if s_ != 'IID' for sd in SEEDS]
    E = {}
    with Pool(int(os.environ.get('NP', 7))) as pl:
        for r in pl.imap_unordered(run_family, [(fam,) + j for fam in FAMS for j in shifted_jobs]):
            E.setdefault(r['fam'], {}).setdefault(f"{r['d']}_{r['s']}", {})[str(r['seed'])] = r['red']
    md5 = lambda fn: hashlib.md5(open(f'{HERE}/{fn}', 'rb').read()).hexdigest()
    meta = dict(
        seeds=list(SEEDS),
        n_label_draws=NREP,
        alpha=ALPHA,
        n_T=list(NTS),
        cohort_sizes=list(MS),
        mechanisms=list(STRUCT),
        z_gauss=float(stats.norm.ppf(1 - ALPHA / 2)),
        t_df='nu = n_1 - df_ridge (residual degrees of freedom of the D_T1 ridge fit)',
        env=PIN,
        python=platform.python_version(),
        numpy=np.__version__,
        sklearn=sklearn.__version__,
        scipy=scipy.__version__,
        md5=dict(common=md5('common.py'), mst=md5('mst.py'), mst2=md5('mst2.py')),
        raw_scale='house_sales: price = exp(log price); bike: count = expm1(log1p count); original mst.run_mst with common.load wrapped',
        cap_rule='half-width of the cohort-mean interval truncated at max(y) - min(y) over source-train + source-calibration labels (mst.run_mst: hwc = min(hw, cap))',
        total_secs=time.time() - t0
    )
    assert all(ident.values()), [k for k, v in ident.items() if not v]
    json.dump(
        dict(meta=meta, ref_identical=ident, nu_mean=nu, A=A, B=B, C=C, D=D, E=E),
        open(f'{HERE}/results/AUX_REVISIONS.json', 'w')
    )
