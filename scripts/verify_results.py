import ast, json, glob, os, re, sys, hashlib
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES = os.path.join(ROOT, 'results')
P = lambda *a: os.path.join(RES, *a)
ok = True
def report(tag, text, good):
    global ok; ok &= bool(good); print(f'[{tag}] {text}', '' if good else '  <-- FAILED', flush=True)

class Tracked(dict):
    used = None
    def __getitem__(self, k):
        if self.used is None: self.used = set()
        self.used.add(k); return super().__getitem__(k)
REP = json.load(open(P('reported_values.json'))); N = Tracked(REP['macros']); TAB = REP['tables']
README = open(os.path.join(ROOT, 'README.md')).read()

bad, n, H = [], 0, hashlib.sha256()
for line in open(P('LOCK_V10_FILE_HASHES.txt')):
    name, ref = line.rstrip('\n').split('\t'); n += 1
    o = json.load(open(P(f'MSTLOCKA_{name}')))
    h = hashlib.sha256(json.dumps(dict(res=o['res'], diag=o['diag']), sort_keys=True).encode()).hexdigest(); H.update(h.encode())
    if h != ref: bad.append(name)
report('0a', f'MSTLOCKA canonical payload hashes: {n - len(bad)}/{n} match LOCK_V10_FILE_HASHES.txt; combined SHA-256 {H.hexdigest()[:16]}...', not bad and n == 45)
man = {}
for line in open(P('CONFIRM2_MANIFEST.txt')):
    if line.startswith('#') or not line.strip(): continue
    fn_, md_ = line.split(); man[fn_] = md_
bad7 = [fn_ for fn_, md_ in man.items() if not fn_.endswith('.py') and hashlib.md5(open(os.path.join(ROOT, fn_), 'rb').read()).hexdigest() != md_]
sums = [l.split('  ', 1) for l in open(P('SHA256SUMS')).read().splitlines()]
badsum = [f for h, f in sums if hashlib.sha256(open(P(f), 'rb').read()).hexdigest() != h]
extra = sorted(set(os.listdir(RES)) - {f for _, f in sums} - {'SHA256SUMS', 'reported_values.json', 'LICENSE'})
report('0e', f'SHA-256 of the stored result files: {len(sums) - len(badsum)}/{len(sums)} match SHA256SUMS; mismatches: {badsum}; result files not listed: {extra}', not badsum and not extra and bool(sums))
report('0b', f'frozen confirmation artifacts match CONFIRM2_MANIFEST.txt (md5): {sum(not f.endswith(".py") for f in man) - len(bad7)}/{sum(not f.endswith(".py") for f in man)}; mismatches: {bad7}', not bad7 and len(man) >= 9)
AX = json.load(open(P('AUX_REVISIONS.json'))); SEEDS = ('3', '4', '5'); WL10 = [(d, s_) for d in ('adult', 'bike', 'california', 'house_sales', 'wine_quality') for s_ in ('S1', 'S2')]
FROZEN = dict(common='711c9d24a82baa501b8a84a29eaad4ae', mst='047d060c98cf5ae7f7e661ea424dd6c0', mst2='440433d0fa1937b7841bb45e3ab985ae')
FP = {'common.py': '3cdfa8944abf3caea7437b1637e249915d51eeae1c68510c84aa2bc8e115a1c6', 'mst.py': '8fa8ddb05dc1d3c3d7e020d29d8acdd82b2c4979e9910b199daa978d11f1e38e', 'mst2.py': '7b14d8abbea7b7e47e6431a84775669c37b6b9866934a2c2ea3e3a13b5c71cf2', 'aux_revisions.py': 'b85a3437be92052f35323db8e5340ca93cf615da28aaf06bece856469c688b46', 'sbx.py': '4486ea1550681db6c5a29d872493c32fdb1a6eea7ddf31418617cec133e20242', 'inv12_full.py': 'e481657f2f954282aed980b90644a660acfe893a8043f463cc8b2a0892dcafa9', 'inv3.py': 'b453fdff3bb419e6258682f9128cb6dc0fafce754c9a4076d067f06c3236d805', 'confirm12.py': 'd0502b19fd80c8a80c228d2a5c65ea217602b05ab4950f60d251121a0f7d1c77', 'confirm3.py': 'b7a1a93298e822bd919d451a3b2f8599ab8126baf9b5b5d69dccef849bc2ea0f', 'be_proxy_lock.py': 'c0ac430c4bac46db5057313ab460117666d3b7d1af37a0711161136f976b0ced', 'lock_verify.py': '94aa9f9f7a12b7680d66d9292590a31af6e48c4429aa7562e5cbbb525f3ab679', 'lock_summ_mst.py': 'd3ab2c8700335a6604b0bd555e716ed674abea98c807093fd7d19d58d16848d8'}
PYFP = sys.version_info[:2] == (3, 12)
fpbad = [f for f, h in FP.items() if hashlib.sha256(ast.dump(ast.parse(open(os.path.join(ROOT, f)).read())).encode()).hexdigest() != h] if PYFP else None
c4 = [AX['meta']['md5'] == FROZEN, len(AX['ref_identical']) == 45 and all(AX['ref_identical'].values())]
report('0c', f'md5 of the frozen harness scripts recorded by the aux run matches the expected values: {c4[0]}; aux M0_MST path reproduces all 45 locked files exactly: {c4[1]}', all(c4))
report('0d', 'code of the 12 scripts (syntax tree, independent of comments and formatting) matches the recorded fingerprints: ' + ('skipped, the fingerprints were computed with Python 3.12' if fpbad is None else f'{len(FP) - len(fpbad)}/{len(FP)}; mismatches: {fpbad}'), fpbad is None or not fpbad)

RAW_, RAW_ALL = {}, {}
for f_ in sorted(glob.glob(P('MSTLOCKA_*.json'))):
    o = json.load(open(f_)); RAW_[(o['meta']['d'], o['meta']['s'], o['meta']['seed'])] = o['res']['200']; RAW_ALL[(o['meta']['d'], o['meta']['s'], o['meta']['seed'])] = o['res']
STm = ('knn', 'slice', 'decile'); MSz = (25, 50, 100)

txt = open(P('LOCK_CONFIRMATION_TABLE_v10_runA.txt')).read()
METH = ['S0', 'T1', 'T2', 'T3', 'T4', 'C1', 'C2', 'C3', 'M0']; old = {}
for blk in re.split(r'\n(?=######## )', txt):
    mm = re.match(r'######## lockfinal seeds=\[3, 4, 5\]  n_T=(\d+):', blk)
    if not mm: continue
    nT = mm.group(1)
    for line in blk.splitlines():
        r = re.match(r'(adult|bike|californ|house_sa|wine_qua)\w*\s+(S1|S2|IID)\s+(.*)', line)
        if not r: continue
        old[(nT, r.group(1), r.group(2))] = re.findall(r'\s*([\d.]+)/\s*([\d.]+)/([\d.]+)(\*?)', r.group(3))
PRET = {'Adult': 'adult', 'Bike': 'bike', 'California': 'californ', 'House sales': 'house_sa', 'Wine': 'wine_qua'}
def parse_tab(name):
    rows = {}
    for cells in TAB[name]:
        m = re.match(r'^(Adult|Bike|California|House sales|Wine)( S1| S2)?$', cells[0])
        if m: rows[(PRET[m.group(1)], (m.group(2) or ' IID').strip())] = cells[1:]
    return rows
bad = 0; chk = 0
ORDER = [0, 1, 2, 3, 4, 7, 5, 8]
for nT in ('100', '200', '300'):
    for (d, s), cells in parse_tab(f'main{nT}').items():
        for ci, mi in enumerate(ORDER):
            c = cells[ci]
            if c == 'inf': continue
            cv, w = c.split('/'); o = old[(nT, d, s)][mi]; chk += 1
            if abs(float(cv) - float(o[0])) > 0.0051 or abs(float(w) - float(o[1])) > 0.0051 + (0.05 if float(o[1]) >= 10 else 0): bad += 1; print('  MISMATCH', nT, d, s, METH[mi], c, o)
for (d, s), cells in parse_tab('iid').items():
    if s != 'IID': continue
    for ci, mi in ((0, 0), (1, 8)):
        cv, w = cells[ci].split('/'); o = old[('200', d, 'IID')][mi]; chk += 1
        if abs(float(cv) - float(o[0])) > 0.0051 or abs(float(w) - float(o[1])) > 0.0051: bad += 1; print('  MISMATCH IID', d, cells[ci], o)
report('1a', f'reported table cells compared with the independent older summariser output (LOCK_CONFIRMATION_TABLE): {chk}; mismatches: {bad}', bad == 0 and chk == 230)

MSKEY = {'S0': 'S0_B3_source', 'T1': 'T1_B3_target_disjoint', 'T2': 'T2_B3_target_randsub', 'T3': 'T3_global_debiased_B3', 'T4': 'T4_pointwise_minkowski', 'C1': 'C1_nodrift_micro_cia', 'C2': 'C2_localdrift_source_width', 'C3': 'C3_driftcorr_pointwise_mink', 'M0': 'M0_MST'}
LONG = {'adult': 'adult', 'bike': 'bike', 'californ': 'california', 'house_sa': 'house_sales', 'wine_qua': 'wine_quality'}
def pool_cell(nT, d, s, me):
    cov = wr = inf = 0.0; k = 0
    for sd in (3, 4, 5):
        res = RAW_ALL[(d, s, sd)][nT]
        for i in range(20):
            for md in STm:
                for m in MSz:
                    x = res[f'{md}|{m}'][me]; cov += x['cov'][i]; wr += x['wr'][i]; inf += x['inf'][i]; k += 1
    return cov / k, wr / k, inf / k
bad1 = 0; chk1 = 0
for (nT, d, s), cells in sorted(old.items()):
    if (nT != '200' and s == 'IID'): continue
    for mi, me in enumerate(METH):
        c, w, i = pool_cell(nT, LONG[d], s, MSKEY[me]); o = cells[mi]
        if i >= 0.99: continue
        chk1 += 1
        if abs(c - float(o[0])) > 0.0051 or abs(w - float(o[1])) > 0.0051 + (0.05 if float(o[1]) >= 10 else 0): bad1 += 1; print('  MISMATCH raw', nT, d, s, me, (round(c, 3), round(w, 3)), o)
report('1b', f'cells recomputed from the raw MSTLOCKA payloads vs the older summariser output: {chk1}; mismatches: {bad1}', bad1 == 0 and chk1 > 0)
bad1c = 0; chk1c = 0
for nT in ('100', '200', '300'):
    for (d, s_), cells in parse_tab(f'main{nT}').items():
        for ci, mi in enumerate(ORDER):
            c_, w_, i_ = pool_cell(nT, LONG[d], s_, MSKEY[METH[mi]]); chk1c += 1; rep = cells[ci]
            if (rep == 'inf') != (i_ >= 0.99): bad1c += 1; print('  MISMATCH inf flag', nT, d, s_, METH[mi], rep, round(i_, 3)); continue
            if rep == 'inf': continue
            cv_, wv_ = rep.split('/')
            if abs(float(cv_) - c_) > 0.0051 or abs(float(wv_) - w_) > 0.0051 + (0.05 if w_ >= 10 else 0): bad1c += 1; print('  MISMATCH reported vs raw', nT, d, s_, METH[mi], rep, (round(c_, 3), round(w_, 3)))
report('1c', f'reported main-table cells (including the infinity cells) vs the raw MSTLOCKA payloads: {chk1c}; mismatches: {bad1c}', bad1c == 0 and chk1c == 240)

chk = 0; bad4 = []; VER = set()
def cmp(name, val, fmt='{:.2f}'):
    global chk; chk += 1; VER.add(name)
    if str(N[name]) != fmt.format(val): bad4.append((name, N[name], fmt.format(val)))
def draws(group, k, nT, me, key): return [x for sd in SEEDS for x in AX[group][f'{k[0]}_{k[1]}'][sd][nT][me][key]]
def wstat(group, k, nT, me):
    c, w = draws(group, k, nT, me, 'cov'), draws(group, k, nT, me, 'wr'); return st.fmean(c), st.fmean(w), sum(1 for x in c if x < 0.80) / len(c)
for me, sh in (('A_gauss', 'G'), ('A_t', 'T')):
    R_ = [wstat('A', k, '200', me) for k in WL10]
    cmp(f'ax{sh}cov', st.fmean(r[0] for r in R_)); cmp(f'ax{sh}W', st.fmean(r[1] for r in R_)); cmp(f'ax{sh}low', st.fmean(r[2] for r in R_)); cmp(f'ax{sh}ok', sum(r[0] >= 0.85 for r in R_), '{}')
    for nT in ('100', '200', '300'):
        R_ = [wstat('A', k, nT, me) for k in WL10]; cmp(f'axN{nT}{sh}cov', st.fmean(r[0] for r in R_)); cmp(f'axN{nT}{sh}W', st.fmean(r[1] for r in R_)); cmp(f'axN{nT}{sh}ok', sum(r[0] >= 0.85 for r in R_), '{}')
for nT in ('100', '200', '300'):
    R_ = [wstat('A', k, nT, 'M0_MST') for k in WL10]; cmp(f'axN{nT}Vcov', st.fmean(r[0] for r in R_)); cmp(f'axN{nT}VW', st.fmean(r[1] for r in R_)); cmp(f'axN{nT}Vok', sum(r[0] >= 0.85 for r in R_), '{}')
for me, sh in (('A_gauss', 'G'), ('A_t', 'T'), ('M0_MST', 'V')):
    for s_ in ('S1', 'S2'): cmp(f'ax{sh}adult{s_}'.replace('adult', 'ad'), wstat('A', ('adult', s_), '200', me)[0])
cmp('axNu200', st.fmean(AX['nu_mean'][f'{k[0]}_{k[1]}_{sd}']['200'] for k in WL10 for sd in SEEDS), '{:.0f}'); cmp('axNu100', st.fmean(AX['nu_mean'][f'{k[0]}_{k[1]}_{sd}']['100'] for k in WL10 for sd in SEEDS), '{:.0f}')
RAWW = [('house_sales', 'S1'), ('house_sales', 'S2'), ('bike', 'S1'), ('bike', 'S2')]
for me, sh in (('M0_MST', 'V'), ('T4_pointwise_minkowski', 'M'), ('S0_B3_source', 'B'), ('C1_nodrift_micro_cia', 'C'), ('C3_driftcorr_pointwise_mink', 'D')):
    R_ = [wstat('B', k, '200', me) for k in RAWW]; cmp(f'axRaw{sh}cov', st.fmean(r[0] for r in R_)); cmp(f'axRaw{sh}W', st.fmean(r[1] for r in R_)); cmp(f'axRaw{sh}min', min(r[0] for r in R_)); cmp(f'axRaw{sh}ok', sum(r[0] >= 0.85 for r in R_), '{}')
R_ = [wstat('B', k, '200', 'M0_MST') for k in RAWW]; cmp('axRawMstWmin', min(r[1] for r in R_)); cmp('axRawMstWmax', max(r[1] for r in R_))
acts = {m: [] for m in ('M0_MST', 'A_gauss', 'A_t')}; dc = {m: [] for m in acts}; n100 = nad = 0
for kk, bs in AX['C'].items():
    for sd, v in bs.items():
        for nT, cc in v['cells'].items():
            for c_, mm in cc.items():
                for me, x in mm.items(): acts[me].append((x['act_mean'], x['act_max'])); dc[me].append(abs(x['cov_cap'] - x['cov_uncap']))
                if mm['M0_MST']['act_mean'] > 0: n100 += nT == '100'; nad += kk.startswith('adult')
for me, sh in (('M0_MST', 'V'), ('A_gauss', 'G'), ('A_t', 'T')):
    a_ = acts[me]; cmp(f'axCap{sh}Mean', st.fmean(x[0] for x in a_), '{:.4f}'); cmp(f'axCap{sh}Max', max(x[1] for x in a_), '{:.4f}'); cmp(f'axCap{sh}Cells', sum(x[0] > 0 for x in a_), '{}'); cmp(f'axCap{sh}CellsTot', len(a_), '{}'); cmp(f'axCap{sh}DCov', max(dc[me]), '{:.4f}')
cmp('axCapVCellsN100', n100, '{}'); cmp('axCapVCellsAdult', nad, '{}')
def parse_rows(name):
    return {cells[0]: cells[1:] for cells in TAB[name]}
ab = parse_rows('abl')
for key, me in (('Gaussian plug-in', 'A_gauss'), ('Student-$t$ plug-in', 'A_t')):
    row = [v for k, v in ab.items() if k.startswith(key)][0]; R_ = [wstat('A', k, '200', me) for k in WL10]
    exp = [f'{st.fmean(r[0] for r in R_):.2f}', f'{st.fmean(r[1] for r in R_):.2f}', f'{st.fmean(r[2] for r in R_):.2f}', str(sum(r[0] >= 0.85 for r in R_))]; chk += 4
    if row[:4] != exp: bad4.append((key, row[:4], exp))
rw = parse_rows('raw'); order = [('S0_B3_source', 'B3-Source'), ('T4_pointwise_minkowski', 'Tgt Minkowski'), ('C3_driftcorr_pointwise_mink', 'Drift-Corr. Mink.'), ('C1_nodrift_micro_cia', 'No-Drift $\\mu$CIA'), ('M0_MST', 'MST-CP (original)')]
for me, nm in order:
    for k, cellv in zip(RAWW, rw[nm]):
        v = wstat('B', k, '200', me); chk += 1; e = f'{v[0]:.2f}/{v[1]:.1f}' if v[1] >= 10 else f'{v[0]:.2f}/{v[1]:.2f}'
        if cellv != e: bad4.append((nm, k, cellv, e))
report('2a', f'aux-derived macros and table cells recomputed independently: {chk}; mismatches: {len(bad4)} {bad4[:5]}', not bad4)
def pooled(res, me, modes, ms, key): return [st.fmean(res[f'{md}|{m}'][me][key][i] for md in modes for m in ms) for i in range(20)]
H = [[st.fmean(pooled(RAW_[(d, s_, int(sd))], 'M0_MST', STm, MSz, 'cov')[i] for (d, s_) in WL10) for i in range(20)] for sd in SEEDS]
sm = [st.fmean(h) for h in H]; sdseed = st.stdev(sm); chk2 = 0; bad5 = []
for name, val, fmt in (('dpCovVMean', st.fmean(sm), '{:.2f}'), ('dpCovVSDseed', sdseed, '{:.3f}'), ('dpCovVSE', sdseed / 3 ** 0.5, '{:.3f}'), ('dpCovVSDdraw', (st.fmean(st.variance(h) for h in H)) ** 0.5, '{:.3f}')):
    chk2 += 1; VER.add(name)
    if N[name] != fmt.format(val): bad5.append((name, N[name], fmt.format(val)))
cellmeans = {(md, m): st.fmean(st.fmean(RAW_[(d, s_, int(sd))][f'{md}|{m}']['M0_MST']['cov']) for (d, s_) in WL10 for sd in SEEDS) for md in STm for m in MSz}
wlcells = [st.fmean(st.fmean(RAW_[(d, s_, int(sd))][f'{md}|{m}']['M0_MST']['cov']) for sd in SEEDS) for (d, s_) in WL10 for md in STm for m in MSz]
for name, val, fmt in (('msCellMin', min(cellmeans.values()), '{:.2f}'), ('msWlMin', min(wlcells), '{:.2f}'), ('msWlBelow85', sum(c < 0.85 for c in wlcells), '{}'), ('msWlN', len(wlcells), '{}')):
    chk2 += 1; VER.add(name)
    if str(N[name]) != fmt.format(val): bad5.append((name, N[name], fmt.format(val)))
for name, key_ in (('axLog4cov', 'cov'), ('axLog4W', 'wr')):
    chk2 += 1; val = st.fmean(st.fmean(st.fmean(RAW_[(d, s_, int(sd))][f'{md}|{m}']['M0_MST'][key_]) for md in STm for m in MSz) for (d, s_) in RAWW for sd in SEEDS)
    VER.add(name)
    if N[name] != f'{val:.2f}': bad5.append((name, N[name], f'{val:.2f}'))
report('2b', f'dispersion and mechanism x size macros recomputed independently from the locked payloads: {chk2}; mismatches: {len(bad5)} {bad5}', not bad5)
chk3 = 0; bad6 = []
def cmp2(name, val, fmt='{:.2f}'):
    global chk3; chk3 += 1; VER.add(name)
    if str(N[name]) != fmt.format(val): bad6.append((name, N[name], fmt.format(val)))
def dcarr(k, nT, v, key): return [x for sd in SEEDS for x in AX['D'][f'{k[0]}_{k[1]}'][sd][nT][v][key]]
for nT in ('100', '200', '300'):
    for v, sh in (('DC_a', 'A'), ('DC_b', 'B')):
        cmp2(f'axDc{sh}{nT}Cov', st.fmean(st.fmean(dcarr(k, nT, v, 'cov')) for k in WL10)); cmp2(f'axDc{sh}{nT}W', st.fmean(st.fmean(dcarr(k, nT, v, 'wr')) for k in WL10))
        cmp2(f'axDc{sh}{nT}R', st.fmean(st.fmean(dcarr(k, nT, v, 'ratio')) for k in WL10)); cmp2(f'axDc{sh}{nT}Le1', sum(st.fmean(dcarr(k, nT, v, 'ratio')) <= 1.0 for k in WL10), '{}')
def lk(k, me, key):
    return [st.fmean(RAW_[(k[0], k[1], int(sd))][f'{md}|{m}'][me][key][i] for md in STm for m in MSz) for sd in SEEDS for i in range(20)]
def fam_draws(fk, k, me, key):
    if fk == 'Hgb': return lk(k, me, key)
    return [x for sd in SEEDS for x in AX['E'][{'Mlp': 'MLP', 'Rf': 'RF', 'Lin': 'LIN'}[fk]][f'{k[0]}_{k[1]}'][sd][me][key]]
FMS = (('S0_B3_source', 'B'), ('T4_pointwise_minkowski', 'M'), ('C1_nodrift_micro_cia', 'C'), ('M0_MST', 'V'))
for fk in ('Hgb', 'Mlp', 'Rf', 'Lin'):
    for me, sh in FMS: cmp2(f'axPf{fk}{sh}Cov', st.fmean(st.fmean(fam_draws(fk, k, me, 'cov')) for k in WL10)); cmp2(f'axPf{fk}{sh}W', st.fmean(st.fmean(fam_draws(fk, k, me, 'wr')) for k in WL10))
    cv = [st.fmean(fam_draws(fk, k, 'M0_MST', 'cov')) for k in WL10]; cmp2(f'axPf{fk}Vmin', min(cv)); cmp2(f'axPf{fk}Vok', sum(c >= 0.85 for c in cv), '{}')
    cmp2(f'axPf{fk}Vlow', st.fmean(sum(1 for x in fam_draws(fk, k, 'M0_MST', 'cov') if x < 0.80) / 60 for k in WL10)); cmp2(f'axPf{fk}Mmin', min(st.fmean(fam_draws(fk, k, 'T4_pointwise_minkowski', 'cov')) for k in WL10))
fw_ = lambda x: f'{x:.1f}' if x >= 10 else f'{x:.2f}'
tf = parse_rows('fam')
for fk, nm in (('Hgb', 'Boosting'), ('Mlp', 'MLP $(64,32)$'), ('Rf', 'Random forest'), ('Lin', 'Ridge/logistic')):
    row = tf[nm]; exp = [f"{st.fmean(st.fmean(fam_draws(fk, k, me, 'cov')) for k in WL10):.2f}/{fw_(st.fmean(st.fmean(fam_draws(fk, k, me, 'wr')) for k in WL10))}" for me, _ in FMS]
    cvs = [st.fmean(fam_draws(fk, k, 'M0_MST', 'cov')) for k in WL10]; exp += [f'{min(cvs):.2f}', str(sum(c >= 0.85 for c in cvs))]; chk3 += 1
    if row[:6] != exp: bad6.append((nm, row[:6], exp))
td = parse_rows('dc')
for nT in ('100', '200', '300'):
    row = td[nT]; mw = st.fmean(st.fmean(AX['A'][f'{k[0]}_{k[1]}'][sd][nT]['M0_MST']['wr'][i_] for sd in SEEDS for i_ in range(20)) for k in WL10); mc = st.fmean(st.fmean(AX['A'][f'{k[0]}_{k[1]}'][sd][nT]['M0_MST']['cov'][i_] for sd in SEEDS for i_ in range(20)) for k in WL10)
    exp_dc = [f"{st.fmean(st.fmean(dcarr(k, nT, 'DC_a', 'cov')) for k in WL10):.2f}", f"{st.fmean(st.fmean(dcarr(k, nT, 'DC_a', 'ratio')) for k in WL10):.2f}", f"{st.fmean(st.fmean(dcarr(k, nT, 'DC_b', 'cov')) for k in WL10):.2f}", f"{st.fmean(st.fmean(dcarr(k, nT, 'DC_b', 'ratio')) for k in WL10):.2f}"]
    chk3 += 1
    o1 = []
    for k in WL10:
        vals = []
        for sd in SEEDS:
            r_ = json.load(open(P(f'V11RLOCK_{k[0]}_{k[1]}_{sd}.json')))['res'][nT]
            vals += [st.fmean(r_[f'{md}|{m}']['O1_oracle']['cov']) for md in STm for m in MSz]
        o1.append(st.fmean(vals))
    o1m = f'{st.fmean(o1):.2f}'
    if [row[3], row[4], row[5], row[6]] != exp_dc or [row[0], row[1], row[2]] != [f'{mc:.2f}', f'{mw:.2f}', o1m]: bad6.append((nT, row, exp_dc))
report('2c', f'drift-corrected-oracle and predictor-family macros and table cells recomputed independently: {chk3}; mismatches: {len(bad6)} {bad6[:4]}', not bad6)

C12 = json.load(open(P('CONFIRM2_predictor_L0_L2_seeds6-15.json'))); C3 = json.load(open(P('CONFIRM2_rolling_audit_seeds6-15.json')))
CS = list(range(6, 16)); chk4 = 0; bad8 = []
def cmp3(name, val, fmt='{:.3f}'):
    global chk4; chk4 += 1; VER.add(name)
    if str(N[name]) != fmt.format(val): bad8.append((name, N[name], fmt.format(val)))
cmp3('cfNseeds', len(CS), '{}'); cmp3('cfSeedLo', CS[0], '{}'); cmp3('cfSeedHi', CS[-1], '{}')
assert sorted({r['seed'] for r in C12}) == CS and sorted({r['seed'] for r in C3}) == CS
def pm(fam, basis, k, key): return [st.fmean(r['macro'][key]) for sd in CS for r in C12 if (r['fam'], r['basis'], r['d'], r['s'], r['seed']) == (fam, basis, k[0], k[1], sd)]
for nm, fam, basis in (('Bz', 'HGB', 'base'), ('Lz', 'LIN', 'base'), ('Lt', 'LIN', 'quad')):
    cv = {k: pm(fam, basis, k, 'cov') for k in WL10}
    cmp3(f'cf{nm}Cov', st.fmean(st.fmean(v) for v in cv.values()))
    if nm != 'Bz': cmp3(f'cf{nm}Worst', min(st.fmean(v) for v in cv.values()))
rows7 = parse_rows('predsens'); PRN = {'adult': 'Adult', 'bike': 'Bike', 'california': 'California', 'house_sales': 'House sales', 'wine_quality': 'Wine'}
for k in WL10:
    b, l0, l2 = pm('HGB', 'base', k, 'cov'), pm('LIN', 'base', k, 'cov'), pm('LIN', 'quad', k, 'cov'); w0, w2 = pm('LIN', 'base', k, 'wr'), pm('LIN', 'quad', k, 'wr')
    exp = [f'{st.fmean(b):.3f}', f'{st.fmean(l0):.3f}', f'{st.fmean(l2):.3f}', f'{st.fmean(l2) - st.fmean(l0):+.3f}', str(sum(x > y for x, y in zip(l2, l0))), f'{st.fmean(w0):.2f}', f'{st.fmean(w2):.2f}']; chk4 += 1
    if rows7[f'{PRN[k[0]]} {k[1]}'][:7] != exp: bad8.append((k, rows7[f'{PRN[k[0]]} {k[1]}'][:7], exp))
RS = ('EARLY200', 'EARLY100', 'ROLL4x50', 'PER2x100'); TAGS = {('house_sales', 'S2'): 'Hs', ('bike', 'S1'): 'Bk'}; SK2 = {'EARLY200': 'Ez', 'EARLY100': 'Eh', 'ROLL4x50': 'Rz', 'PER2x100': 'Pt'}
for (d_, s_), tag in TAGS.items():
    rs = sorted([r for r in C3 if (r['d'], r['s']) == (d_, s_)], key=lambda r: r['seed']); r0 = rs[0]
    ev = 100 * sum(r0['n_E']) / (r0['n_I'] + sum(r0['n_R'])); cmp3('cfEvalShare', ev, '{:.1f}')
    def seedmean(st_, m_, key): return [st.fmean(x for row in r['res'][st_][m_][key] for x in row) for r in rs]
    def winmean(st_, m_, key, w): return st.fmean(row[w] for r in rs for row in r['res'][st_][m_][key])
    for st_ in RS:
        ov = st.fmean(seedmean(st_, 'MST', 'cov')); wn = [winmean(st_, 'MST', 'cov', w) for w in range(3)]
        if st_ != 'PER2x100':
            cmp3(f'cf{tag}{SK2[st_]}Cov', ov)
            if not (tag == 'Bk' and st_ == 'EARLY100'): cmp3(f'cf{tag}{SK2[st_]}Worst', min(wn))
        elif tag == 'Bk': cmp3('cfBkPtCov', ov)
    d2 = [a_ - b_ for a_, b_ in zip(seedmean('ROLL4x50', 'MST', 'cov'), seedmean('EARLY100', 'MST', 'cov'))]; cmp3(f'cf{tag}RzEhDiff', st.fmean(d2)); cmp3(f'cf{tag}RzEhPos', sum(x > 0 for x in d2), '{}')
    if tag == 'Hs':
        cmp3('cfHsRzEhWidR', st.fmean(a_ / b_ for a_, b_ in zip(seedmean('ROLL4x50', 'MST', 'wid'), seedmean('EARLY100', 'MST', 'wid'))), '{:.2f}'); cmp3('cfHsRzEhWkR', st.fmean(a_ / b_ for a_, b_ in zip(seedmean('ROLL4x50', 'MST', 'wk'), seedmean('EARLY100', 'MST', 'wk'))), '{:.2f}')
lines8 = [cells for cells in TAB['roll'] if len(cells) == 8]
for gi, ((d_, s_), tag) in enumerate(TAGS.items()):
    rs = sorted([r for r in C3 if (r['d'], r['s']) == (d_, s_)], key=lambda r: r['seed'])
    for si, st_ in enumerate(RS):
        row = lines8[gi * 4 + si][2:]
        g = lambda m_, key, w=None: st.fmean((r_[w] if w is not None else st.fmean(r_)) for r in rs for r_ in r['res'][st_][m_][key])
        exp = [f"{g('MST','cov'):.3f}", f"{g('MST','cov',0):.2f} / {g('MST','cov',1):.2f} / {g('MST','cov',2):.2f}", f"{g('MST','wid'):.3f}", f"{g('MST','wk'):.3f}", f"{g('MINK','cov'):.3f}", f"{g('MINK','wk'):.3f}"]; chk4 += 1
        if row != exp: bad8.append((d_, st_, row, exp))
report('2d', f'confirmation-round (seeds 6-15) macros and table cells recomputed independently from the frozen artifacts: {chk4}; mismatches: {len(bad8)} {bad8[:4]}', not bad8)
rec = sorted(k for k in N if k.startswith(('ax', 'cf', 'dp', 'ms')))
report('2e', f'reported macros with prefix ax/cf/dp/ms: {len(rec)}; recomputed above: {sum(k in VER for k in rec)}', True)


EXPECT_ROWS = dict(main100=10, main200=10, main300=10, iid=5, abl=2, raw=5, fam=4, dc=3, predsens=10, roll=8)
EXPECT_COLS = dict(main100=9, main200=9, main300=9, iid=3, abl=5, raw=5, fam=7, dc=8, predsens=8, roll=8)
shape = [t for t in TAB if len(TAB[t]) != EXPECT_ROWS[t] or any(len(r) != EXPECT_COLS[t] for r in TAB[t])]
unused = sorted(set(N) - (N.used or set()))
report('2f', f'every value in reported_values.json was compared: {len(N) - len(unused)}/{len(N)} macros; tables with unexpected shape: {shape}; unused macros: {unused[:5]}', not unused and not shape and set(TAB) == set(EXPECT_ROWS))

# README claims recomputed from the stored artifacts
def seedpool(prefix, d, s_, nT, me, key, ms=MSz):
    out = []
    for sd in (3, 4, 5):
        res = json.load(open(P(f'{prefix}_{d}_{s_}_{sd}.json')))['res'][nT]
        for i in range(20): out.append(st.fmean(res[f'{md}|{m}'][me][key][i] for md in STm for m in ms))
    return st.fmean(out)
SHIFTED = [(d, s_) for d in ('adult', 'bike', 'california', 'house_sales', 'wine_quality') for s_ in ('S1', 'S2')]
IIDW = [(d, 'IID') for d in ('adult', 'bike', 'california', 'house_sales', 'wine_quality')]
mcov = {k: pool_cell('200', k[0], k[1], 'M0_MST')[0] for k in SHIFTED}; mwr = {k: pool_cell('200', k[0], k[1], 'M0_MST')[1] for k in SHIFTED}
iid_m0 = st.fmean(pool_cell('200', d, s_, 'M0_MST')[1] for d, s_ in IIDW); iid_s0 = st.fmean(pool_cell('200', d, s_, 'S0_B3_source')[1] for d, s_ in IIDW)
def dcm(v, key): return st.fmean(st.fmean(dcarr(k, '200', v, key)) for k in WL10)
dc_all = all(st.fmean(dcarr(k, '200', v, 'ratio')) > 1.0 for k in WL10 for v in ('DC_a', 'DC_b'))
ordc = {nT: seedpool('V11OLOCK', 'house_sales', 'S2', nT, 'V10', 'cov') for nT in ('100', '200', '300')}; ordbk = seedpool('V11OLOCK', 'bike', 'S1', '200', 'V10', 'cov')
ov = [seedpool('V11OLOCK', d, s_, '200', 'V10', 'cov') for d, s_ in SHIFTED if (d, s_) not in (('house_sales', 'S2'), ('bike', 'S1'))]
def rolling(d_, s_, sg): return st.fmean(st.fmean(x) for r in C3 if (r['d'], r['s']) == (d_, s_) for x in r['res'][sg]['MST']['cov'])
lin35 = [st.fmean(x for sd in SEEDS for x in AX['E']['LIN'][f'{k[0]}_{k[1]}'][sd]['M0_MST']['cov']) for k in WL10]
def predmean(fam, basis): return {k: st.fmean(st.fmean(r['macro']['cov']) for r in C12 if (r['fam'], r['basis'], r['d'], r['s']) == (fam, basis) + k) for k in WL10}
l0 = predmean('LIN', 'base'); l2 = predmean('LIN', 'quad')
claims = [
    f"mean coverage over the ten shifted workloads is {st.fmean(mcov.values()):.2f} (worst workload {min(mcov.values()):.2f}), and the mean Winkler score is {st.fmean(mwr.values()):.2f} times that of an uncorrected labelled-target oracle",
    f"on the five IID workloads at `n_T = 200` the mean Winkler ratio is {iid_m0:.2f} against {iid_s0:.2f}",
    f"At `n_T = 200` the Winkler ratio is {dcm('DC_a', 'ratio'):.2f} against the oracle DC_a and {dcm('DC_b', 'ratio'):.2f} against DC_b, on all ten shifted workloads for both, and those oracles cover {dcm('DC_a', 'cov'):.2f} and {dcm('DC_b', 'cov'):.2f}",
    f"MST-CP v1.0 covers {ordc['200']:.2f} on house sales S2 at `n_T = 200` ({ordc['100']:.2f} at `n_T = 100` and {ordc['300']:.2f} at `n_T = 300`) and {ordbk:.2f} on bike S1",
    f"mean coverage is {st.fmean(ov):.2f}, the lowest workload is {min(ov):.2f}, and {sum(c >= 0.85 for c in ov)} of {len(ov)} reach 0.85",
    f"coverage with all labels taken from the initial window is {rolling('house_sales', 'S2', 'EARLY200'):.2f} on house sales S2 and {rolling('bike', 'S1', 'EARLY200'):.2f} on bike S1. With four blocks of 50 labels it is {rolling('house_sales', 'S2', 'ROLL4x50'):.2f} and {rolling('bike', 'S1', 'ROLL4x50'):.2f}, and with two blocks of 100 labels it is {rolling('house_sales', 'S2', 'PER2x100'):.2f} and {rolling('bike', 'S1', 'PER2x100'):.2f}",
    f"mean coverage reaches 0.85 on only {sum(c >= 0.85 for c in lin35)} of the 10 workloads on seeds 3 to 5 (lowest {min(lin35):.2f}) and on {sum(c >= 0.85 for c in l0.values())} of 10 on seeds 6 to 15 (lowest {min(l0.values()):.2f}). On seeds 6 to 15 a quadratic drift basis, a post hoc variant that is not the submitted method, gives mean coverage {st.fmean(l2.values()):.2f} against {st.fmean(l0.values()):.2f} without it",
    f"{len(SHIFTED)} shifted workloads",
]
claims = claims[:-1]
nfiles = (len(glob.glob(P('MSTLOCKA_*.json'))), len(glob.glob(P('V11RLOCK_*.json'))), len(glob.glob(P('V11OLOCK_*.json'))))
missing = [c for c in claims if c not in README]
report('3', f'README numerical claims recomputed from the stored artifacts and found verbatim in README.md: {len(claims) - len(missing)}/{len(claims)}; workloads: {len(SHIFTED)} shifted and {len(IIDW)} IID, payload files per run {nfiles}; not reproduced: {missing}', not missing and len(SHIFTED) == 10 and dc_all and nfiles == (45, 45, 30))

print('VERIFICATION', 'PASSED' if ok else 'FAILED'); sys.exit(0 if ok else 1)
