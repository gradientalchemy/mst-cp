# Shared harness: datasets, source and target workloads, cohort drawing and interval utilities.

import os, sys, json, time, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.ensemble import HistGradientBoostingRegressor as HR, HistGradientBoostingClassifier as HC
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder
DATA = HERE + '/data'
ALPHA = 0.1
Z90 = 1.6449
SIZES = 25, 50, 100
MODES = 'knn', 'slice', 'decile', 'random'
WCLIP = 20.0


# Returns X, y, task ('reg' or 'clf'), feature names and an ordering key (None without time
# order). OpenML datasets are pinned by id: wine_quality 287, house_sales 42079,
# Bike_Sharing_Demand 42712, adult 1590.
def load(name):
    if name == 'california':
        c = fetch_california_housing(data_home=DATA)
        return c.data.astype(float), c.target.astype(float), 'reg', list(c.feature_names), None
    if name == 'wine_quality':
        a = fetch_openml(data_id=287, as_frame=True, data_home=DATA)
        return a.data.astype(float).values, a.target.astype(float).values, 'reg', list(a.data.columns), None
    if name == 'house_sales':
        a = fetch_openml(data_id=42079, as_frame=True, data_home=DATA)
        df = a.data.copy()
        order = df['date'].astype(str).values
        df = df.drop(columns=['date']).astype(float)
        X = df.drop(columns=['price'])
        return X.values, np.log(df['price'].values), 'reg', list(X.columns), order
    if name == 'bike':
        a = fetch_openml(data_id=42712, as_frame=True, data_home=DATA)
        df = a.data.copy()
        cat = [c for c in df.columns if str(df[c].dtype) == 'category']
        df[cat] = OrdinalEncoder().fit_transform(df[cat])
        return df.astype(float).values, np.log1p(
            a.target.astype(float).values
        ), 'reg', list(df.columns), np.arange(len(df))
    a = fetch_openml(data_id=1590, as_frame=True, data_home=DATA)
    df = a.data.copy()
    cat = [c for c in df.columns if str(df[c].dtype) in ('category', 'object')]
    df[cat] = OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1
    ).fit_transform(df[cat])
    return df.astype(float).values, a.target.astype(str).str.contains('>').astype(float).values, 'clf', list(df.columns), None


# Shift definitions. 'var': source and target come from opposite tails of one feature (see
# make_split); 'time': time-ordered 55% / 20% / 25% split into source training, source calibration
# and target. excl: features hidden from the density-ratio classifier.
SPLITS = {
    ('california', 'S1'): dict(kind='var', var='Latitude', dirn='lo2hi', excl=['Latitude']),
    ('california', 'S2'): dict(kind='var', var='Longitude', dirn='hi2lo', excl=['Longitude']),
    ('house_sales', 'S1'): dict(kind='var', var='lat', dirn='lo2hi', excl=['lat']),
    ('house_sales', 'S2'): dict(kind='time', excl=[]),
    ('bike', 'S1'): dict(kind='time', excl=['year']),
    ('bike', 'S2'): dict(kind='var', var='temp', dirn='lo2hi', excl=['temp', 'feel_temp']),
    ('adult', 'S1'): dict(kind='var', var='age', dirn='lo2hi', excl=['age']),
    ('adult', 'S2'): dict(kind='var', var='education-num', dirn='lo2hi', excl=['education-num', 'education']),
    ('wine_quality', 'S1'): dict(kind='var', var='alcohol', dirn='lo2hi', excl=['alcohol']),
    ('wine_quality', 'S2'): dict(
        kind='var', var='total.sulfur.dioxide', dirn='hi2lo', excl=['total.sulfur.dioxide']
    )
}


def workloads():
    return [
        (d, s)
        for d in ('california', 'house_sales', 'bike', 'adult', 'wine_quality')
        for s in ('S1', 'S2', 'IID')
    ]


# tc: target rows available for labelling (first 25% of the shuffled target); te: the remaining
# target rows, used for testing.
def make_split(X, fn, order, d, s, seed):
    n = len(X)
    rng = np.random.RandomState(seed)
    if s == 'IID':
        perm = rng.permutation(n)
        src = perm[:int(0.6 * n)]
        tgt = perm[int(0.6 * n):int(0.85 * n)]
        spec = dict(kind='iid', excl=[])
        src = rng.permutation(src)
        a = int(0.65 * len(src))
        tr, cal = src[:a], src[a:]
    else:
        spec = SPLITS[d, s]
        if spec['kind'] == 'time':
            o = np.argsort(order, kind='stable')
            tr, cal, tgt = o[:int(0.55 * n)], o[int(0.55 * n):int(0.75 * n)], o[int(0.75 * n):]
        else:
            v = X[:, fn.index(spec['var'])]
            if spec['dirn'] == 'lo2hi':
                src = np.where(v <= np.quantile(v, 0.6))[0]
                tgt = np.where(v >= np.quantile(v, 0.75))[0]
            else:
                src = np.where(v >= np.quantile(v, 0.4))[0]
                tgt = np.where(v <= np.quantile(v, 0.25))[0]
            src = rng.permutation(src)
            a = int(0.65 * len(src))
            tr, cal = src[:a], src[a:]
    tgt = rng.permutation(tgt)
    b = int(0.25 * len(tgt))
    tc, te = tgt[:b], tgt[b:]
    return dict(tr=tr, cal=cal, tc=tc, te=te, spec=spec)


# Cross-fitted density-ratio weights p_T(x) / p_S(x) from a domain classifier on features only,
# clipped to [0.05 / WCLIP, WCLIP].
def domain_weights(X, fn, sp, seed):
    keep = [j for j, f in enumerate(fn) if f not in sp['spec']['excl']]
    src = np.concatenate([sp['tr'], sp['cal']])
    tgt = np.concatenate([sp['tc'], sp['te']])
    idx = np.concatenate([src, tgt])
    dl = np.r_[np.zeros(len(src)), np.ones(len(tgt))]
    rng = np.random.RandomState(seed + 7)
    fold = rng.randint(0, 3, len(idx))
    p = np.zeros(len(idx))
    Xd = X[idx][:, keep]
    for k in range(3):
        m = HC(
            max_iter=100, max_depth=3, class_weight='balanced', early_stopping=False, random_state=seed
        ).fit(Xd[fold != k], dl[fold != k])
        p[fold == k] = m.predict_proba(Xd[fold == k])[:, 1]
    p = np.clip(p, 1e-06, 1 - 1e-06)
    w = np.clip(p / (1 - p), 1.0 / WCLIP * 0.05, WCLIP)
    W = np.zeros(len(X))
    W[idx] = w
    return W


# Split-conformal quantile: the ceil((n + 1)(1 - a))-th smallest score, capped at the maximum;
# +inf for an empty set.
def qconf(s, a=ALPHA):
    n = len(s)
    if n == 0:
        return np.inf
    return np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - a)) / n), method='higher')


def winkler(L, U, y, a=ALPHA):
    return U - L + 2 / a * np.maximum(L - y, 0) + 2 / a * np.maximum(y - U, 0)


# Weighted conformal quantile; the test point's weight wt is a point mass at +inf.
def wquant(scores, w, wt, a=ALPHA):
    o = np.argsort(scores)
    s = scores[o]
    c = np.cumsum(w[o])
    W = c[-1]
    thr = (1 - a) * (W + wt)
    k = np.searchsorted(c, thr, side='left')
    out = np.where(k >= len(s), np.inf, s[np.minimum(k, len(s) - 1)])
    return out


# Cohorts of size m: random subsets, kNN balls around random centres, blocks after sorting by a
# random feature ('slice'), or blocks in prediction order ('decile').
def draw(Zs, Xs, f, mode, m, nco, rng, nbr_cache):
    n = len(f)
    if mode == 'random':
        return np.array([rng.choice(n, m, replace=False) for _ in range(nco)])
    if mode == 'knn':
        if 'knn' not in nbr_cache:
            c = rng.choice(n, nco, replace=n < nco)
            nbr_cache['knn'] = NearestNeighbors(n_neighbors=min(100, n)).fit(Zs).kneighbors(
                Zs[c], return_distance=False
            )
        return nbr_cache['knn'][:, :m]
    if mode == 'slice':
        out = []
        for _ in range(nco):
            j = rng.randint(Xs.shape[1])
            o = np.lexsort((rng.rand(n), Xs[:, j]))
            s0 = rng.randint(0, n - m + 1)
            out.append(o[s0:s0 + m])
        return np.array(out)
    o = np.argsort(f + 1e-09 * rng.rand(n))
    return np.array([o[s0:s0 + m] for s0 in rng.randint(0, n - m + 1, nco)])


# The generator depends only on (seed, mode, m), so feature-based cohorts coincide across models.
def cells_for(Xs, Zs, f, y, w, seed, nco, sizes=SIZES, modes=MODES):
    out = {}
    for mi, mode in enumerate(modes):
        cache = {}
        for m in sizes:
            r = np.random.RandomState(seed * 1000 + mi * 10 + m % 7)
            I = draw(Zs, Xs, f, mode, m, nco, r, cache)
            lw = np.log(w)[I]
            out[mode, m] = dict(
                I=I, yb=y[I].mean(1), fb=f[I].mean(1), wm=w[I].mean(1), lw=lw.sum(1), m=m, e=(y - f)[I]
            )
    return out
