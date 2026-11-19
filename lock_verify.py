# Compares MSTLOCKA with MSTLOCKB (res and diag, wall-clock excluded) and writes
# LOCK_V10_FILE_HASHES.txt.

import json, glob, hashlib, sys, os


def canon(p):
    o = json.load(open(p))
    return json.dumps(dict(res=o['res'], diag=o['diag']), sort_keys=True)


A = sorted(glob.glob('results/MSTLOCKA_*.json'))
B = sorted(glob.glob('results/MSTLOCKB_*.json'))
na = [os.path.basename(p).replace('MSTLOCKA_', '') for p in A]
nb = [os.path.basename(p).replace('MSTLOCKB_', '') for p in B]
assert na == nb and len(na) == 45, (len(na), len(nb))
bad = []
H = hashlib.sha256()
lines = []
for a, b, n in zip(A, B, na):
    ca, cb = canon(a), canon(b)
    ha, hb = hashlib.sha256(ca.encode()).hexdigest(), hashlib.sha256(cb.encode()).hexdigest()
    lines.append(f'{n}\t{ha}')
    H.update(ha.encode())
    if ha != hb:
        bad.append(n)
open('LOCK_V10_FILE_HASHES.txt', 'w').write('\n'.join(lines) + '\n')
print(f'files compared: {len(na)}; non-identical: {len(bad)} {bad}')
print('combined SHA-256 of the 45 canonical result payloads (run A):', H.hexdigest())
sys.exit(1 if bad else 0)
