#!/usr/bin/env python3
"""
angle_distribution.py
=====================
Compute the weighted distribution of the angle between the nearest
O-H bond vector of each water molecule and the direction from that
oxygen toward the central Argon atom.

Geometry
--------
For each water molecule (O atom) within R_max of Ar:
  - r_O  = O position  (Ar is at origin in .cen files)
  - r_H1, r_H2 = the two H atoms belonging to that O (paired by nearest-O)
  - Nearest H  = whichever of H1, H2 is closer to Ar
  - Direction from O toward Ar: u_Ar = -r_O / |r_O|
  - O-H bond vector of nearest H: v = r_H_near - r_O
  - Angle θ = arccos( v̂ · û_Ar )

  θ = 0°  : O-H bond points directly at Ar  (H between O and Ar)
  θ = 90° : O-H bond perpendicular to O-Ar axis
  θ = 180°: O-H bond points away from Ar

Normalization
-------------
Each cluster's histogram is normalised to integrate to 1 (sum × bin_width = 1).
The final distribution is the weighted sum of those normalised histograms.

Parameters
----------
--basenames     FILE   BASE_NAMES file              (default: BASE_NAMES)
--weights_file  FILE   weight file (REQUIRED): weight  basename
--ext_in        STR    file extension               (default: .cen)
--R_max         FLOAT  O-Ar distance cutoff (Å)     (default: 5.4)
--n_bins        INT    number of bins 0-180°        (default: 36 = 5° bins)
--out           STR    output text file             (default: angle_dist.txt)
--nthreads      INT    worker processes             (default: all CPUs)

Requirements: numpy
"""

import argparse
import multiprocessing
import time
from multiprocessing import Pool

import numpy as np


def read_cen(path):
    with open(path) as fh:
        n = int(fh.readline()); fh.readline()
        labels, coords = [], []
        for _ in range(n):
            p = fh.readline().split()
            labels.append(p[0])
            coords.append([float(x) for x in p[1:4]])
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


def _angle_histogram(args):
    """
    Compute the normalised angle histogram for one cluster.
    Returns (bn, weighted_hist_or_None, n_angles, error_or_None).
    """
    bn, ext_in, R_max, n_bins, weight = args

    try:
        labels, coords = read_cen(bn + ext_in)
    except Exception as e:
        return bn, None, 0, str(e)

    o_idx = np.array([i for i, l in enumerate(labels) if l.startswith('O')])
    h_idx = np.array([i for i, l in enumerate(labels) if l.startswith('H')])

    if len(o_idx) == 0 or len(h_idx) == 0:
        return bn, None, 0, 'no O or H atoms'

    o_coords = coords[o_idx]   # (nO, 3)
    h_coords = coords[h_idx]   # (nH, 3)

    # Pair each H to its nearest O
    diff     = h_coords[:, None, :] - o_coords[None, :, :]  # (nH, nO, 3)
    h_to_O   = np.sum(diff**2, axis=2)                       # (nH, nO)
    h_owner  = np.argmin(h_to_O, axis=1)                     # (nH,)

    h_to_Ar  = np.linalg.norm(h_coords, axis=1)              # (nH,)
    o_to_Ar  = np.linalg.norm(o_coords, axis=1)              # (nO,)
    in_shell = np.where(o_to_Ar <= R_max)[0]

    if len(in_shell) == 0:
        return bn, None, 0, 'no O within R_max'

    angles_deg = []
    for oi in in_shell:
        h_mine = np.where(h_owner == oi)[0]
        if len(h_mine) == 0:
            continue
        nearest_hi = h_mine[np.argmin(h_to_Ar[h_mine])]

        r_O      = o_coords[oi]
        r_H_near = h_coords[nearest_hi]
        OH       = r_H_near - r_O
        OH_norm  = np.linalg.norm(OH)
        OAr_norm = np.linalg.norm(r_O)
        if OH_norm < 1e-10 or OAr_norm < 1e-10:
            continue

        # Direction O → Ar is -r_O/|r_O|
        cos_t = np.dot(OH, -r_O) / (OH_norm * OAr_norm)
        angles_deg.append(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))

    if not angles_deg:
        return bn, None, 0, 'no valid angles'

    hist, _ = np.histogram(angles_deg, bins=n_bins, range=(0.0, 180.0))
    hist    = hist.astype(np.float64)
    # Normalise to probability per bin: sum(hist) = 1
    total   = hist.sum()
    if total > 0:
        hist /= total

    return bn, weight * hist, len(angles_deg), None


def _batch_histograms(batch):
    """Accumulate angle histograms over a batch of clusters."""
    if not batch:
        return np.zeros(batch[0][3] if batch else 1), 0, []
    n_bins = batch[0][3]
    acc    = np.zeros(n_bins)
    errors = []
    n_ok   = 0
    for job_args in batch:
        bn, hist, _, err = _angle_histogram(job_args)
        if err:
            errors.append(f'{bn}: {err}')
        elif hist is not None:
            acc  += hist
            n_ok += 1
    return acc, n_ok, errors


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    parser.add_argument('--basenames',    default='BASE_NAMES')
    parser.add_argument('--weights_file', required=True)
    parser.add_argument('--ext_in',       default='.cen')
    parser.add_argument('--R_max',        type=float, default=5.4)
    parser.add_argument('--n_bins',       type=int,   default=36)
    parser.add_argument('--out',          default='angle_dist.txt')
    parser.add_argument('--nthreads',     type=int,   default=None)
    args = parser.parse_args()

    nworkers  = args.nthreads or multiprocessing.cpu_count()
    bin_width = 180.0 / args.n_bins
    bin_edges = np.linspace(0, 180, args.n_bins + 1)
    bin_ctrs  = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    t0        = time.time()

    print(f'R_max      : {args.R_max} Å')
    print(f'Bins       : {args.n_bins}  ({bin_width:.1f}° each, 0–180°)')
    print(f'Workers    : {nworkers}')
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
        (bn, args.ext_in, args.R_max, args.n_bins, float(w))
        for bn, w in zip(basenames, weights) if w > 1e-10
    ]
    n = len(worker_args)
    print(f'Active clusters: {n}')
    print()

    splits  = np.array_split(np.arange(n), nworkers)
    batches = [[worker_args[i] for i in idx] for idx in splits if len(idx)]

    print(f'Computing ...', flush=True)
    Dhist  = np.zeros(args.n_bins)
    errors = []
    done   = 0

    with Pool(processes=nworkers) as pool:
        for acc, n_ok, errs in pool.imap_unordered(_batch_histograms, batches):
            done  += n_ok
            Dhist += acc
            errors.extend(errs)
            print(f'  {done}/{n}', flush=True)

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed:.1f} s')

    if errors:
        print(f'\n{len(errors)} error(s):')
        for e in errors[:10]:
            print(f'  {e}')

    integral = Dhist.sum()
    print(f'\nDistribution sum  : {integral:.4f}  (expect ≈ 1.0)')
    print(f'Peak at           : {bin_ctrs[np.argmax(Dhist)]:.1f}°  '
          f'(probability = {Dhist.max():.4f} per {bin_width:.0f}° bin)')

    with open(args.out, 'w') as fh:
        fh.write('# Nearest-H angle distribution\n')
        fh.write('# θ: angle between O-H bond and direction from O toward Ar\n')
        fh.write('# 0°=H pointing at Ar, 90°=perpendicular, 180°=H away from Ar\n')
        fh.write(f'# Normalisation: probability per bin; sum over all bins = 1\n')
        fh.write(f'# R_max={args.R_max} Å  n_bins={args.n_bins}  '
                 f'bin_width={bin_width:.2f}°\n')
        fh.write(f'# weights: {args.weights_file}\n')
        fh.write(f'# {"bin_centre_deg":>16s}  {"probability":>16s}\n')
        for theta, d in zip(bin_ctrs, Dhist):
            fh.write(f'  {theta:18.4f}  {d:18.6f}\n')

    print(f'Written to: {args.out}')


if __name__ == '__main__':
    main()
