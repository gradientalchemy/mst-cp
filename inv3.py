# Same-budget audit strategies on the temporal workloads (house_sales S2, bike S1). The ordered
# target is split into I (first 25%) and three windows R_1 to R_3; each window has an acquisition
# part Q_k (its first 25%) and test rows E_k (the rest).
# Every strategy uses 200 labels: EARLY200 (all from I), EARLY100 (the first 100), ROLL4x50 (50
# from I, then 50 from each Q_k; window k uses the two latest blocks), PER2x100 (100 from I, 100
# from Q_2). No later label is used for an earlier window.

import sbx, json, numpy as np, time, sys
from sbx import *
STRATS = 'EARLY200', 'EARLY100', 'ROLL4x50', 'PER2x100'


def blocks(P, I, Q, rep, seed):
    r = np.random.RandomState(seed * 7919 + rep)
    e200 = I[r.permutation(len(I))[:200]]
    rq = [np.random.RandomState(seed * 104729 + rep * 31 + k) for k in range(1, 4)]
    q50 = [Q[k][rq[k].permutation(len(Q[k]))[:50]] for k in range(3)]
    q100 = Q[1][np.random.RandomState(seed * 130363 + rep).permutation(len(Q[1]))[:100]]
    return e200, q50, q100


def labels_for(strat, w, e200, q50, q100):
    if strat == 'EARLY200':
        return e200
    if strat == 'EARLY100':
        return e200[:100]
    if strat == 'ROLL4x50':
        # b[0] comes from I and b[k] from Q_k; window w uses b[w] and b[w + 1].
        b = [e200[:50]] + q50
        return np.concatenate([b[w], b[w + 1]])
    if strat == 'PER2x100':
        return e200[:100] if w == 0 else q100


def job(args):
    d, s_, seed, nrep = args
    t0 = time.time()
    P = prep(d, s_, seed, 'HGB', 'base')
    sp = P['sp']
    tg = np.concatenate([sp['tc'], sp['te']])
    tg = tg[np.argsort(np.asarray(P['order'])[tg], kind='stable')]
    b = int(0.25 * len(tg))
    I, H = tg[:b], tg[b:]
    R = np.array_split(H, 3)
    Q = []
    E = []
    for r in R:
        q = int(0.25 * len(r))
        Q.append(r[:q])
        E.append(r[q:])
    X, Z, f, y, e, ones, Phi, s2 = P['X'], P['Z'], P['f'], P['y'], P['e'], P['ones'], P['Phi'], P['s2']
    cells = []
    for w in range(3):
        cte = cells_for(X[E[w]], Z[E[w]], f[E[w]], y[E[w]], ones[E[w]], seed + 3, 200, modes=STRUCT)
        cell = {}
        for (md, m), T in cte.items():
            cell[md, m] = dict(
                T=T, Pb=Phi[E[w]][T['I']].mean(1), s2b=s2[E[w]][T['I']].mean(1), o1=1.0, q_o1=0.5
            )
        cells.append(cell)
    out = {
        st_: {
            meth: dict(
                cov=np.zeros((nrep, 3)),
                wid=np.zeros((nrep, 3)),
                wk=np.zeros((nrep, 3)),
                inf=np.zeros((nrep, 3))
            )
            for meth in ('MST', 'MINK')
        }
        for st_ in STRATS
    }
    for rep in range(nrep):
        e200, q50, q100 = blocks(P, I, Q, rep, seed)
        for st_ in STRATS:
            for w in range(3):
                lab = labels_for(st_, w, e200, q50, q100)
                rr = np.random.RandomState(seed * 999331 + rep * 17 + w)
                rows = lab[rr.permutation(len(lab))]
                F = fit_mst(P, rows, np.random.RandomState(seed * 100000 + rep * 10 + w))
                res = eval_cells(P, F, cells[w])
                a = pooled(res)
                o = out[st_]['MST']
                o['cov'][rep, w], o['wk'][rep, w], o['wid'][rep, w], o['inf'][rep, w] = a[0], a[3], a[2], a[4]
                q_pt = qstrict(np.abs(e[rows]))
                cv = []
                wd = []
                wk = []
                for (md, m), c in cells[w].items():
                    T = c['T']
                    hw = min(q_pt, P['cap']) if np.isfinite(q_pt) else P['cap']
                    L, U = T['fb'] - hw, T['fb'] + hw
                    cv.append(np.mean((T['yb'] >= L) & (T['yb'] <= U)))
                    wd.append(2 * hw)
                    wk.append(np.mean(winkler(L, U, T['yb'])))
                o = out[st_]['MINK']
                o['cov'][rep, w], o['wid'][rep, w], o['wk'][rep, w] = np.mean(cv), np.mean(wd), np.mean(wk)
    ser = {
        st_: {m_: {k: v.tolist() for k, v in dd.items()} for m_, dd in dm.items()}
        for st_,
        dm in out.items()
    }
    return dict(
        d=d,
        s=s_,
        seed=seed,
        nrep=nrep,
        n_I=len(I),
        n_R=[len(r) for r in R],
        n_Q=[len(q) for q in Q],
        n_E=[len(x) for x in E],
        res=ser,
        secs=time.time() - t0
    )


if __name__ == '__main__':
    from multiprocessing import Pool
    jobs = [(d, s_, sd, 20) for d, s_ in (('house_sales', 'S2'), ('bike', 'S1')) for sd in SEEDS]
    out = []
    with Pool(6) as pl:
        for r in pl.imap_unordered(job, jobs):
            out.append(r)
    json.dump(out, open('out/inv3_full.json', 'w'))
