# Final confirmation B: rolling-audit strategies on seeds 6 to 15; writes out/confirm3.json.

import json, time
from inv3 import job
SEEDS_C = tuple(range(6, 16))


if __name__ == '__main__':
    from multiprocessing import Pool
    t0 = time.time()
    jobs = [(d, s_, sd, 20) for d, s_ in (('house_sales', 'S2'), ('bike', 'S1')) for sd in SEEDS_C]
    out = []
    with Pool(7) as pl:
        for r in pl.imap_unordered(job, jobs):
            out.append(r)
    json.dump(out, open('out/confirm3.json', 'w'))
