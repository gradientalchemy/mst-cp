# Predictor-sensitivity job (n_T = 200, 10 shifted workloads): macro metrics, per-cell and held-
# out micro-scale coverage, variance diagnostics.

import sbx, json, numpy as np, time, sys
from sbx import *
# (family, basis): HGB/base (deployed method), LIN/base (L0), LIN/quad (L2).
CFGS = ('HGB', 'base'), ('LIN', 'base'), ('LIN', 'quad')


def job(args):
    fam, basis, d, s_, seed = args
    t0 = time.time()
    P = prep(d, s_, seed, fam, basis)
    cell = make_cells(P)
    perms = draws_perms(P)
    M = dict(cov=[], wr=[], wdr=[], wk=[], inf=[])
    cc = {}
    micro = {
        md: {
            k: []
            for k in (
                'cov_greedy',
                'cov_compact',
                'ks_g',
                'ks_c',
                'q90_cal',
                'q90_g',
                'q90_c',
                'med_cal',
                'med_g',
                'med_c',
                'ter_first',
                'ter_last',
                'q',
                'n_g'
            )
        }
        for md in STRUCT
    }
    diag = []
    for rep in range(NREP):
        rows = P['tc'][perms[rep][:200]]
        rg = np.random.RandomState(seed * 100000 + rep * 10 + 2)
        F = fit_mst(P, rows, rg)
        r = eval_cells(P, F, cell)
        a = pooled(r)
        for k_, v_ in zip(('cov', 'wr', 'wdr', 'wk', 'inf'), a):
            M[k_].append(float(v_))
        for key, v_ in r.items():
            cc.setdefault(f'{key[0]}|{key[1]}', []).append(v_[0])
        diag.append(diagnostics(P, F, cell))
        mo = micro_eval(P, F, seed * 1000 + rep)
        for md in STRUCT:
            o = mo[md]
            mm = micro[md]
            cal = o['cal']
            g = o['greedy']
            c = o['compact']
            mm['cov_greedy'].append(o['cov_greedy'])
            mm['cov_compact'].append(o['cov_compact'])
            mm['q'].append(o['q'] if np.isfinite(o['q']) else None)
            mm['n_g'].append(len(g))
            mm['ks_g'].append(float(stats.ks_2samp(cal, g).statistic))
            mm['ks_c'].append(float(stats.ks_2samp(cal, c).statistic))
            mm['q90_cal'].append(float(np.quantile(cal, 0.9)))
            mm['q90_g'].append(float(np.quantile(g, 0.9)))
            mm['q90_c'].append(float(np.quantile(c, 0.9)))
            mm['med_cal'].append(float(np.median(cal)))
            mm['med_g'].append(float(np.median(g)))
            mm['med_c'].append(float(np.median(c)))
            k3 = len(cal) // 3
            mm['ter_first'].append(float(np.mean(cal[:k3])))
            mm['ter_last'].append(float(np.mean(cal[-k3:])))
    return dict(
        fam=fam,
        basis=basis,
        d=d,
        s=s_,
        seed=seed,
        macro=M,
        cell_cov={k: float(np.mean(v)) for k, v in cc.items()},
        micro=micro,
        diag={k: float(np.mean([x[k] for x in diag])) for k in diag[0]},
        K0=len(P and F['cal']['knn']),
        m0=F['m0'],
        secs=time.time() - t0
    )


if __name__ == '__main__':
    from multiprocessing import Pool
    t0 = time.time()
    jobs = [(f_, b_, d, s_, sd) for f_, b_ in CFGS for d, s_ in WL10 for sd in SEEDS]
    out = []
    with Pool(7) as pl:
        for r in pl.imap_unordered(job, jobs):
            out.append(r)
    json.dump(out, open('out/inv12_full.json', 'w'))
