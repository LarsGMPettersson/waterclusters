#!/usr/bin/env python3
"""
cage_volume.py
==============
For each centred cluster (.cen file, Ar at origin) compute the volume
available to Argon inside the cage of nearest oxygen atoms.

Two volumes are reported for each cluster:

  V_hull   — convex hull volume of the O atoms within R_max (Å³).
             This is the geometric cage volume treating O atoms as points.

  V_avail  — volume inside the hull not occupied by O van der Waals spheres.
             Computed by Monte Carlo integration: random points are accepted
             if they lie inside the convex hull AND are further than r_vdW_O
             from every O atom.
             V_avail = V_hull × (fraction of interior points not in any O sphere)

Effective radii are derived as R = (3V/4π)^(1/3), i.e. the radius of a
sphere with the same volume.

Output
------
  --out FILE         per-cluster table (sorted by V_hull)
  --hist_out FILE    two-column histogram data for plotting (V_hull and V_avail)

Parameters
----------
  --basenames    FILE   BASE_NAMES file                  (default: BASE_NAMES)
  --weights_file FILE   Two-column weight file (REQUIRED)
  --ext_in       STR    File extension                   (default: .cen)
  --R_max        FLOAT  O-Ar cutoff radius (Å)           (default: 5.4)
  --r_vdW_O      FLOAT  O van der Waals radius (Å)       (default: 1.52)
  --N_mc         INT    Monte Carlo points per cluster   (default: 50000)
  --n_bins       INT    Histogram bins                   (default: 40)
  --nthreads     INT    Worker processes                 (default: all CPUs)
  --seed         INT    RNG seed for reproducibility     (default: 42)

Requirements: numpy, scipy
"""

import argparse
import multiprocessing
import time
from multiprocessing import Pool

import numpy as np
from scipy.spatial import ConvexHull
try:
    from scipy.spatial import QhullError
except ImportError:
    from scipy.spatial.qhull import QhullError   # scipy < 1.8


# ────────────────────────────────────────────────────────────────────────────
# I/O
# ────────────────────────────────────────────────────────────────────────────

def read_cen(path):
    with open(path) as fh:
        n = int(fh.readline()); fh.readline()
        labels, coords = [], []
        for _ in range(n):
            p = fh.readline().split()
            labels.append(p[0]); coords.append([float(x) for x in p[1:4]])
    return labels, np.array(coords, dtype=np.float64)


def read_weights(path):
    weights = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                weights[parts[1]] = float(parts[0])
    return weights


# ────────────────────────────────────────────────────────────────────────────
# Worker
# ────────────────────────────────────────────────────────────────────────────

def _cage_volume(args):
    """
    Compute convex hull volume and Monte Carlo available volume for one cluster.

    Returns (bn, V_hull, V_avail, R_hull, R_avail, n_O, error_or_None).
    """
    bn, ext_in, R_max, r_vdW_O, N_mc, seed = args

    try:
        labels, coords = read_cen(bn + ext_in)
    except Exception as e:
        return bn, None, None, None, None, 0, str(e)

    o_idx    = [i for i, l in enumerate(labels) if l.startswith('O')]
    o_coords = coords[o_idx]
    dists    = np.linalg.norm(o_coords, axis=1)
    in_shell = o_coords[dists <= R_max]
    n_O      = len(in_shell)

    if n_O < 4:
        return bn, None, None, None, None, n_O, \
            f'only {n_O} O atoms within R_max (need ≥ 4 for convex hull)'

    # ── Convex hull ──────────────────────────────────────────────────────
    try:
        hull    = ConvexHull(in_shell)
        V_hull  = hull.volume
        R_hull  = (3.0 * V_hull / (4.0 * np.pi)) ** (1.0/3.0)
    except QhullError as e:
        return bn, None, None, None, None, n_O, f'ConvexHull error: {e}'

    # ── Monte Carlo available volume ──────────────────────────────────────
    rng    = np.random.default_rng(seed)
    lo, hi = in_shell.min(axis=0), in_shell.max(axis=0)
    V_box  = float(np.prod(hi - lo))

    pts      = rng.uniform(lo, hi, (N_mc, 3))

    # Point-in-hull: all half-space inequalities satisfied
    # hull.equations[:, :3] are outward normals, hull.equations[:, 3] are offsets
    # A point x is inside if: equations[:,:3] @ x + equations[:,3] <= 0 for all rows
    inside   = np.all(
        pts @ hull.equations[:, :3].T + hull.equations[:, 3] <= 1e-10,
        axis=1)
    pts_in   = pts[inside]
    n_inside = len(pts_in)

    if n_inside == 0:
        return bn, V_hull, 0.0, R_hull, 0.0, n_O, None

    # Distance from each interior point to every O atom
    # Shape: (n_inside, n_O)
    d_to_O   = np.linalg.norm(
        pts_in[:, None, :] - in_shell[None, :, :], axis=2)
    not_in_O = np.all(d_to_O > r_vdW_O, axis=1)

    f_avail  = not_in_O.sum() / n_inside
    V_avail  = V_hull * f_avail
    R_avail  = (3.0 * V_avail / (4.0 * np.pi)) ** (1.0/3.0) if V_avail > 0 else 0.0

    return bn, V_hull, V_avail, R_hull, R_avail, n_O, None


def _batch(batch):
    """Process a batch, return list of per-cluster results."""
    results = []
    for args in batch:
        results.append(_cage_volume(args))
    return results


# ────────────────────────────────────────────────────────────────────────────
# Threshold reporting helpers
# ────────────────────────────────────────────────────────────────────────────

def _report_threshold(bns, V_hull, V_avail, R_hull, R_avail, n_Os, ws, thresh):
    """Print clusters with V_hull > thresh, sorted by V_hull descending."""
    mask   = V_hull > thresh
    n_over = int(mask.sum())
    w_over = float(ws[mask].sum())

    print(f'\n{"─"*70}')
    print(f'Clusters with V_hull > {thresh:.0f} Å³:  '
          f'{n_over} clusters  ({100*w_over:.3f}% of total weight)')
    print(f'{"─"*70}')
    if n_over == 0:
        print('  None.')
        return

    order = np.argsort(V_hull[mask])[::-1]
    idx   = np.where(mask)[0][order]
    print(f'  {"Basename":<55s}  {"V_hull":>9s}  {"V_avail":>9s}  '
          f'{"R_hull":>8s}  {"weight%":>8s}')
    print(f'  {"─"*55}  {"─"*9}  {"─"*9}  {"─"*8}  {"─"*8}')
    for i in idx:
        print(f'  {bns[i]:<55s}  {V_hull[i]:9.2f}  {V_avail[i]:9.2f}  '
              f'{R_hull[i]:8.4f}  {100*ws[i]:8.4f}%')
    print(f'\n  Cumulative weight of outliers: {100*w_over:.4f}%')
    print(f'  ({n_over} clusters out of {len(bns)} valid, '
          f'{100*n_over/len(bns):.2f}% by count)')


def _report_from_file(path, thresh):
    """
    Read an existing cage_volumes.txt and report clusters above threshold.
    Parses the columns: Basename n_O V_hull V_avail R_hull R_avail weight
    """
    bns, V_hull, V_avail, R_hull, R_avail, n_Os, ws = [], [], [], [], [], [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                bns.append(parts[0])
                n_Os.append(int(parts[1]))
                V_hull.append(float(parts[2]))
                V_avail.append(float(parts[3]))
                R_hull.append(float(parts[4]))
                R_avail.append(float(parts[5]))
                ws.append(float(parts[6]))
            except ValueError:
                continue

    if not bns:
        print(f'No data rows found in {path}'); return

    bns    = np.array(bns)
    V_hull = np.array(V_hull)
    V_avail= np.array(V_avail)
    R_hull = np.array(R_hull)
    R_avail= np.array(R_avail)
    n_Os   = np.array(n_Os)
    ws     = np.array(ws)
    ws    /= ws.sum()   # renormalise (file may have been filtered)

    print(f'Read {len(bns)} clusters from {path}')
    _report_threshold(bns, V_hull, V_avail, R_hull, R_avail, n_Os, ws, thresh)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    parser.add_argument('--basenames',    default='BASE_NAMES')
    parser.add_argument('--weights_file', default=None,
                        help='Two-column weight file: weight  basename. '
                             'Required unless --from_file is used.')
    parser.add_argument('--ext_in',       default='.cen')
    parser.add_argument('--R_max',        type=float, default=5.4)
    parser.add_argument('--r_vdW_O',      type=float, default=1.52,
                        help='O van der Waals radius in Å (default: 1.52)')
    parser.add_argument('--N_mc',         type=int,   default=50000,
                        help='Monte Carlo points per cluster (default: 50000)')
    parser.add_argument('--n_bins',       type=int,   default=40,
                        help='Histogram bins (default: 40)')
    parser.add_argument('--nthreads',     type=int,   default=None)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--out',          default='cage_volumes.txt',
                        help='Per-cluster results table')
    parser.add_argument('--hist_out',     default='cage_volume_hist.txt',
                        help='Histogram data for plotting')
    parser.add_argument('--report_thresh', type=float, default=None,
                        help='Report clusters with V_hull above this threshold '
                             '(Å³) and their cumulative weight fraction.  '
                             'If --basenames / --weights_file are not needed '
                             'you can run in report-only mode by also passing '
                             '--from_file to read an existing cage_volumes.txt.')
    parser.add_argument('--from_file',   default=None,
                        help='Read an existing cage_volumes.txt directly and '
                             'apply --report_thresh without recomputing.  '
                             'Example:  python cage_volume.py '
                             '--from_file cage_volumes.txt --report_thresh 300')
    args = parser.parse_args()

    # ── Report-only mode: read existing table and filter ──────────────────
    if args.from_file:
        _report_from_file(args.from_file,
                          args.report_thresh if args.report_thresh else 0.0)
        return

    if not args.weights_file:
        parser.error('--weights_file is required (unless --from_file is used)')

    nworkers = args.nthreads or multiprocessing.cpu_count()
    t0 = time.time()

    print(f'R_max           : {args.R_max} Å')
    print(f'r_vdW(O)        : {args.r_vdW_O} Å  (Bondi 1964)')
    print(f'Monte Carlo pts : {args.N_mc:,} per cluster')
    print(f'Workers         : {nworkers}')
    print()

    with open(args.basenames) as fh:
        basenames = [ln.strip() for ln in fh if ln.strip()]
    print(f'Total clusters : {len(basenames)}')

    w_raw = read_weights(args.weights_file)
    print(f'Weights from   : {args.weights_file}  ({len(w_raw)} entries)')

    weights_raw = np.array([w_raw.get(bn, 0.0) for bn in basenames])
    total = weights_raw.sum()
    if total < 1e-30:
        raise ValueError('All weights are zero')
    weights = weights_raw / total

    worker_args = [
        (bn, args.ext_in, args.R_max, args.r_vdW_O, args.N_mc,
         args.seed + i)     # unique seed per cluster → reproducible
        for i, (bn, w) in enumerate(zip(basenames, weights))
        if w > 1e-10
    ]
    n = len(worker_args)
    # Build weight lookup for histogram accumulation
    w_lookup = {bn: float(w) for bn, w in zip(basenames, weights)}

    print(f'Active clusters: {n}')
    print()

    # Batch across workers
    splits  = np.array_split(np.arange(n), nworkers)
    batches = [[worker_args[i] for i in idx] for idx in splits if len(idx)]

    print('Computing convex hull and available volumes ...', flush=True)
    all_results = []
    done        = 0
    n_errors    = 0
    first_errors_shown = 0

    with Pool(processes=nworkers) as pool:
        for batch_results in pool.imap_unordered(_batch, batches):
            all_results.extend(batch_results)
            done += len(batch_results)
            # Print first few errors immediately so failures are visible
            for bn, *_, err in batch_results:
                if err:
                    n_errors += 1
                    if first_errors_shown < 5:
                        print(f'  ERROR: {bn}: {err}', flush=True)
                        first_errors_shown += 1
                    elif first_errors_shown == 5:
                        print(f'  (further errors suppressed until summary)',
                              flush=True)
                        first_errors_shown += 1
            print(f'  {done}/{n}  ({n_errors} errors so far)', flush=True)
            # Abort early if nearly everything is failing — likely wrong path
            if done >= min(200, n//4) and n_errors / done > 0.9:
                print(f'\nABORTING: {100*n_errors/done:.0f}% error rate after '
                      f'{done} clusters. Check --ext_in and file paths.',
                      flush=True)
                pool.terminate()
                return

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed:.1f} s  ({n/elapsed:.1f} clusters/s)')

    # ── Collect valid results ─────────────────────────────────────────────
    errors   = [(bn, err) for bn, *_, err in all_results if err]
    valid    = [(bn, Vh, Va, Rh, Ra, nO)
                for bn, Vh, Va, Rh, Ra, nO, err in all_results
                if err is None]

    if errors:
        print(f'\n{len(errors)} error(s):')
        for bn, e in errors[:10]:
            print(f'  {bn}: {e}')

    if not valid:
        print('No valid results.'); return

    bns    = [v[0] for v in valid]
    V_hull = np.array([v[1] for v in valid])
    V_avail= np.array([v[2] for v in valid])
    R_hull = np.array([v[3] for v in valid])
    R_avail= np.array([v[4] for v in valid])
    n_Os   = np.array([v[5] for v in valid])
    ws     = np.array([w_lookup[bn] for bn in bns])
    ws    /= ws.sum()   # renormalise in case some clusters failed

    # ── Weighted statistics ───────────────────────────────────────────────
    def wstats(x, w):
        mean = np.dot(w, x)
        std  = np.sqrt(np.dot(w, (x - mean)**2))
        return mean, std

    Vh_mean, Vh_std   = wstats(V_hull,  ws)
    Va_mean, Va_std   = wstats(V_avail, ws)
    Rh_mean, Rh_std   = wstats(R_hull,  ws)
    Ra_mean, Ra_std   = wstats(R_avail, ws)
    nO_mean, nO_std   = wstats(n_Os.astype(float), ws)

    print(f'\n{"─"*60}')
    print(f'Weighted statistics over {len(valid)} clusters')
    print(f'{"─"*60}')
    print(f'\n  {"":35s}  {"Mean":>10s}  {"Std":>10s}')
    print(f'  {"Convex hull volume (Å³)":35s}  {Vh_mean:10.2f}  {Vh_std:10.2f}')
    print(f'  {"Available volume, excl. O vdW (Å³)":35s}  {Va_mean:10.2f}  {Va_std:10.2f}')
    print(f'  {"Hull effective radius (Å)":35s}  {Rh_mean:10.4f}  {Rh_std:10.4f}')
    print(f'  {"Available effective radius (Å)":35s}  {Ra_mean:10.4f}  {Ra_std:10.4f}')
    print(f'  {"O atoms in shell":35s}  {nO_mean:10.2f}  {nO_std:10.2f}')
    print()
    print(f'  vdW volume reduction: '
          f'{100*(Vh_mean-Va_mean)/Vh_mean:.1f}% of hull volume')
    print(f'  (O vdW radius {args.r_vdW_O} Å, '
          f'Ar vdW radius 1.88 Å for reference)')

    # ── Per-cluster table ─────────────────────────────────────────────────
    order = np.argsort(V_hull)
    with open(args.out, 'w') as fh:
        fh.write('# Cage volume analysis\n')
        fh.write(f'# R_max={args.R_max} Å  r_vdW_O={args.r_vdW_O} Å  '
                 f'N_mc={args.N_mc}\n')
        fh.write(f'# Sorted by V_hull (ascending)\n')
        fh.write(f'# {"Basename":<60s}  {"n_O":>5s}  {"V_hull(Å³)":>12s}  '
                 f'{"V_avail(Å³)":>12s}  {"R_hull(Å)":>10s}  '
                 f'{"R_avail(Å)":>10s}  {"weight":>12s}\n')
        for i in order:
            fh.write(f'  {bns[i]:<60s}  {n_Os[i]:>5d}  {V_hull[i]:>12.3f}  '
                     f'{V_avail[i]:>12.3f}  {R_hull[i]:>10.4f}  '
                     f'{R_avail[i]:>10.4f}  {ws[i]:>12.8f}\n')
    print(f'\nPer-cluster table written to: {args.out}')

    # ── Threshold report ──────────────────────────────────────────────────
    if args.report_thresh is not None:
        _report_threshold(bns, V_hull, V_avail, R_hull, R_avail, n_Os, ws,
                          args.report_thresh)

    # ── Weighted histograms ───────────────────────────────────────────────
    # Auto-range: 1st and 99th weighted percentile
    def wpercentile(x, w, p):
        order = np.argsort(x)
        cdf   = np.cumsum(w[order])
        return np.interp(p/100, cdf, x[order])

    Vh_lo  = wpercentile(V_hull,  ws, 0.5)
    Vh_hi  = wpercentile(V_hull,  ws, 99.5)
    Va_lo  = wpercentile(V_avail, ws, 0.5)
    Va_hi  = wpercentile(V_avail, ws, 99.5)

    # Use same range for both so they overlay cleanly
    lo_all = min(Vh_lo, Va_lo)
    hi_all = max(Vh_hi, Va_hi)

    edges = np.linspace(lo_all, hi_all, args.n_bins + 1)
    ctrs  = 0.5 * (edges[:-1] + edges[1:])
    bw    = edges[1] - edges[0]

    hist_hull  = np.zeros(args.n_bins)
    hist_avail = np.zeros(args.n_bins)
    for i in range(len(valid)):
        bi_h = int(np.clip(np.floor((V_hull[i]  - lo_all) / bw), 0, args.n_bins-1))
        bi_a = int(np.clip(np.floor((V_avail[i] - lo_all) / bw), 0, args.n_bins-1))
        hist_hull[bi_h]  += ws[i]
        hist_avail[bi_a] += ws[i]
    # Normalise: probability per bin
    hist_hull  /= hist_hull.sum()
    hist_avail /= hist_avail.sum()

    with open(args.hist_out, 'w') as fh:
        fh.write('# Cage volume histograms (weighted, probability per bin)\n')
        fh.write(f'# R_max={args.R_max} Å  r_vdW_O={args.r_vdW_O} Å  '
                 f'bin_width={bw:.2f} Å³\n')
        fh.write(f'# {"bin_centre(Å³)":>16s}  {"P(V_hull)":>14s}  '
                 f'{"P(V_avail)":>14s}\n')
        for c, ph, pa in zip(ctrs, hist_hull, hist_avail):
            fh.write(f'  {c:18.3f}  {ph:16.6f}  {pa:16.6f}\n')
    print(f'Histogram data written to:    {args.hist_out}')
    print(f'  Volume range: {lo_all:.1f}–{hi_all:.1f} Å³  '
          f'(bin width {bw:.2f} Å³)')


if __name__ == '__main__':
    main()
