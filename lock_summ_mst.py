# Summary tables from the MSTLOCKA payloads: python lock_summ_mst.py <comma-separated seeds>
# <tag>.

import json, glob, sys, numpy as np
from collections import defaultdict
seeds = [int(x) for x in sys.argv[1].split(',')]
tag = sys.argv[2]
STRUCT = 'knn', 'slice', 'decile'
MS = 25, 50, 100
METH = [
    'S0_B3_source',
    'T1_B3_target_disjoint',
    'T2_B3_target_randsub',
    'T3_global_debiased_B3',
    'T4_pointwise_minkowski',
    'C1_nodrift_micro_cia',
    'C2_localdrift_source_width',
    'C3_driftcorr_pointwise_mink',
    'M0_MST',
    'M1_global_shift',
    'M2_no_tau',
    'M3_lite',
    'M4_naive_disjoint',
    'M5_naive_randsub',
    'M6_agnostic_mix',
    'M7_random_grouping'
]
R = defaultdict(list)
for f in sorted(glob.glob('results/MSTLOCKA_*.json')):
    o = json.load(open(f))
    if o['meta']['seed'] in seeds:
        R[o['meta']['d'], o['meta']['s']].append(o)


def arr(runs, nT, me, key, modes=STRUCT, ms=MS):
    out = []
    for o in runs:
        cols = [np.array(o['res'][nT][f'{md}|{m}'][me][key]) for md in modes for m in ms]
        out.append(np.stack(cols, 1))
    return np.concatenate(out, 0)


def stat(runs, nT, me, modes=STRUCT, ms=MS):
    c = arr(runs, nT, me, 'cov', modes, ms)
    w = arr(runs, nT, me, 'wr', modes, ms)
    i = arr(runs, nT, me, 'inf', modes, ms)
    per_draw = c.mean(1)
    return dict(
        cov=c.mean(), wr=w.mean(), inf=i.mean(), low=float(np.mean(per_draw < 0.8)), sd=float(per_draw.std())
    )


out = []
P = lambda *a: out.append(' '.join((str(x) for x in a)))
wl = [k for k in sorted(R) if k[1] != 'IID']
iid = [k for k in sorted(R) if k[1] == 'IID']
for nT in ('100', '200', '300'):
    P(
        f'\n######## {tag} seeds={seeds}  n_T={nT}: pooled structured modes x m in {{25,50,100}}  cell = cov / Winkler-over-oracle / frac-draws<0.80 / (inf frac if >0)'
    )
    P(f"{'workload':18s}" + ''.join((f'{m[:11]:>17s}' for m in METH)))
    for k in wl + iid:
        row = f"{k[0][:8] + ' ' + k[1]:18s}"
        for me in METH:
            s = stat(R[k], nT, me)
            row += f"{s['cov']:6.2f}/{s['wr']:5.2f}/{s['low']:.2f}" + ('*' if s['inf'] > 0 else ' ')
        P(row)
P('\n(* = some capped infinite intervals)')


def valid(s):
    return s['cov'] >= 0.85


P(f'\n######## SUCCESS BAR ({tag}, n_T=200, structured)')
nT = '200'
i_ok = 0
ii_ok = 0
low = []
v_iv = []
v_v = []
dram = []
best_rows = []
OTHERS = [
    'S0_B3_source',
    'T1_B3_target_disjoint',
    'T2_B3_target_randsub',
    'T3_global_debiased_B3',
    'T4_pointwise_minkowski',
    'C1_nodrift_micro_cia',
    'C2_localdrift_source_width',
    'C3_driftcorr_pointwise_mink'
]
for k in wl:
    s = stat(R[k], nT, 'M0_MST')
    s100 = stat(R[k], nT, 'M0_MST', ms=(100,))
    i_ok += valid(s)
    ii_ok += valid(s100)
    low.append(s['low'])
    if valid(s):
        v_iv.append(s['wr'] <= 1.5)
    vo = {
        me: stat(R[k], nT, me)
        for me in OTHERS
        if valid(stat(R[k], nT, me)) and stat(R[k], nT, me)['inf'] == 0
    }
    bestme = min(vo, key=lambda me: vo[me]['wr']) if vo else None
    win = s['wr'] < vo[bestme]['wr'] if valid(s) and bestme else None
    if win is not None:
        v_v.append(win)
    t4 = stat(R[k], nT, 'T4_pointwise_minkowski')
    if t4['wr'] >= 1.5 and valid(s):
        dram.append(s['wr'] <= 0.7 * t4['wr'])
    P(
        f"{k[0][:8] + ' ' + k[1]:18s} MST cov {s['cov']:.2f} (m=100: {s100['cov']:.2f}) W/O1 {s['wr']:.2f} low-draws {s['low']:.2f} | best valid other: {bestme} {(vo[bestme]['wr'] if bestme else float('nan')):.2f} cov {(vo[bestme]['cov'] if bestme else float('nan')):.2f} | T4 W/O1 {t4['wr']:.2f}"
    )
P(
    f'(i) cov>=0.85 on {i_ok}/10 (need 8); (ii) m=100 cov>=0.85 on {ii_ok}/10 (need 7); (iii) mean frac draws<0.80 = {np.mean(low):.2f} (need <=0.25); (iv) valid & W/O1<=1.5: {sum(v_iv)}/{len(v_iv)} valid; (v) MST lowest Winkler among valid others on {sum(v_v)}/{len(v_v)} comparable (need >=6); (vi) dramatic cut (<=0.7xT4 where T4>=1.5x): {sum(dram)}/{len(dram)}'
)
P(f'\n######## per-m (n_T=200), mean over 10 shifted workloads: cov / W/O1')
for m in MS:
    P(
        f'm={m:3d} ' + ' '.join(
            (
                f"{me[:9]}:{np.mean([stat(R[k], '200', me, ms=(m,))['cov'] for k in wl]):.2f}/{np.mean([stat(R[k], '200', me, ms=(m,))['wr'] for k in wl]):.2f}"
                for me in [
                    'S0_B3_source',
                    'T3_global_debiased_B3',
                    'T4_pointwise_minkowski',
                    'C1_nodrift_micro_cia',
                    'C3_driftcorr_pointwise_mink',
                    'M1_global_shift',
                    'M3_lite',
                    'M0_MST'
                ]
            )
        )
    )
P(
    f'\n######## by n_T (structured, pooled m), mean over 10 shifted workloads: cov / W/O1 / #workloads cov>=0.85'
)
for nT in ('100', '200', '300'):
    P(
        f'nT={nT} ' + ' '.join(
            (
                f"{me[:9]}:{np.mean([stat(R[k], nT, me)['cov'] for k in wl]):.2f}/{np.mean([stat(R[k], nT, me)['wr'] for k in wl]):.2f}/{sum((stat(R[k], nT, me)['cov'] >= 0.85 for k in wl))}"
                for me in [
                    'T4_pointwise_minkowski',
                    'C1_nodrift_micro_cia',
                    'C3_driftcorr_pointwise_mink',
                    'M1_global_shift',
                    'M2_no_tau',
                    'M3_lite',
                    'M6_agnostic_mix',
                    'M7_random_grouping',
                    'M0_MST'
                ]
            )
        )
    )
P(f'\n######## IID controls (n_T=200): MST cov / W/O1')
P(
    ' '.join(
        (
            f"{k[0][:6]}:{stat(R[k], '200', 'M0_MST')['cov']:.2f}/{stat(R[k], '200', 'M0_MST')['wr']:.2f}"
            for k in iid
        )
    )
)
P(f'\n######## random-mode reference (n_T=200, m pooled): M0 cov / W/O1 per workload')
P(
    ' '.join(
        (
            f"{k[0][:5]}{k[1]}:{stat(R[k], '200', 'M0_MST', modes=('random',))['cov']:.2f}/{stat(R[k], '200', 'M0_MST', modes=('random',))['wr']:.2f}"
            for k in wl
        )
    )
)
P(
    f'\n######## size-invariance diagnostic: 0.9-quantile of M0-normalised score by cohort size (mean over workloads x seeds), per mode'
)
D = defaultdict(list)
for k in wl:
    for o in R[k]:
        for kk, v in o['diag'].items():
            md, sz = kk.split('|')
            D[md, 'm0' if int(sz) < 25 else sz].append(v['q90'])
for md in MODES if False else ('knn', 'slice', 'decile', 'random'):
    P(md, {sz: round(float(np.mean(D[md, sz])), 2) for sz in ('m0', '25', '50', '100')})
open(f'MST_RESULTS_{tag}.txt', 'w').write('\n'.join(out))
