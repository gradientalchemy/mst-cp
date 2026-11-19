# Berry-Esseen proxy: Lyapunov ratio Lambda of the row noise (seed 0, descriptive) and the bound
# 0.56 * Lambda / sqrt(k).

import numpy as np, json
from mst2 import *


if __name__ == '__main__':
    out = {}
    for d, s in workloads():
        P = prep(d, s, 0, False)
        rows = P['tc']
        M = Ridge(P['Phi'], P['e'], rows, 'ridge', P['s2'], LAMS_NEW, None, P['c_s'])
        u = P['e'][rows] - P['Phi'][rows] @ M.w
        lam = float(np.mean(np.abs(u) ** 3) / np.mean(u ** 2) ** 1.5)
        kurt = float(np.mean(u ** 4) / np.mean(u ** 2) ** 2)
        out[f'{d}|{s}'] = dict(
            Lambda=lam, kurt=kurt, be={k: 0.56 * lam / np.sqrt(k) for k in (5, 8, 25, 50, 100)}
        )
    json.dump(out, open('results/be_proxy_lock.json', 'w'))
