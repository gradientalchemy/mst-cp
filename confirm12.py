# Final confirmation A: HGB, LIN (L0) and LIN with quadratic basis (L2) on seeds 6 to 15; writes
# out/confirm12.json.

import json, time
from inv12_full import job, CFGS
from sbx import WL10
SEEDS_C = tuple(range(6, 16))


if __name__ == '__main__':
    from multiprocessing import Pool
    t0 = time.time()
    jobs = [(f_, b_, d, s_, sd) for f_, b_ in CFGS for d, s_ in WL10 for sd in SEEDS_C]
    out = []
    with Pool(7) as pl:
        for r in pl.imap_unordered(job, jobs):
            out.append(r)
    json.dump(out, open('out/confirm12.json', 'w'))
