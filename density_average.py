#!/usr/bin/env python3
"""
density_average.py
==================
Compute a weighted-average electron density map from Gaussian-smeared
atomic positions across an ensemble of aligned clusters.  Output is a
cubic grid in VESTA .ped format, directly equivalent to the Fortran code
(Density_for_VESTA_Average.f).

Algorithm
---------
For each cluster (weight > thresh):
  1. Read the aligned .rot file (all atoms, Ar at origin).
  2. For each non-Ar atom within R_max:
       Place a Gaussian: ρᵢ(r) = (α/π)^(3/2) · exp(-α·|r - rᵢ|²)
       where α = alpha_O for O atoms, alpha_H for H atoms.
  3. Integrate → normalise the per-cluster density to
         ∫ρ dV = 8·n_O + |Do_H|·n_H   electrons
     (O has 8 valence electrons, H has 1; Do_H = +1 adds H positive,
     Do_H = -1 subtracts H, Do_H = 0 excludes H entirely).
  4. Accumulate: Dsum += weight · Dmat

Key improvements over the Fortran
----------------------------------
Separable Gaussians
    exp(-α|r-r₀|²) = exp(-α(x-x₀)²)·exp(-α(y-y₀)²)·exp(-α(z-z₀)²)
    Each atom is deposited via an outer product of three 1-D profiles.
    Cost: O(3·n_steps) instead of O(n_steps³) — 500–4000× fewer operations
    per atom depending on α and step_R.

Batched parallel processing
    The cluster list is split into nthreads contiguous batches.  Each
    worker accumulates its own partial Dsum over its entire batch and
    returns one grid at the end.  The main process receives nthreads
    grids and sums them.

    This is critical for performance: the naive approach of returning
    one 10 MB grid per cluster forces the main process to handle 18k
    pickle round-trips serially, stalling the workers and limiting total
    CPU utilisation to ~20%.  With batching, pipe traffic is reduced
    by a factor of ~800 (= clusters/workers), workers run continuously
    at full CPU, and the main process is idle except for the final sum.

Memory budget
    nthreads × 10 MB for partial Dsums (one per worker, in worker memory)
    + one 10 MB Dsum in the main process.
    Peak ≈ (nthreads + 1) × 10 MB — identical to the per-cluster design
    but without any intermediate pipe traffic.

Exact Fortran grid
    Voxel centres: x_i = -R_max + step_R/2 + i·step_R  (i = 0 … nR-1)
    Nearest-voxel lookup uses int() truncation toward zero, matching
    Fortran's behaviour for negative coordinates exactly.

Input files
-----------
BASE_NAMES
    One cluster basename per line (extension stripped; .rot appended).

weights file  (e.g. 4C_weights.dat, supplied via --weights_file)
    Two-column ASCII:  weight  basename
    Weights are re-normalised internally so they need not sum to 1.

Parameters (command line — see --help)
---------------------------------------
--basenames     FILE   BASE_NAMES file                   (default: BASE_NAMES)
--weights_file  FILE   Two-column weight file            (REQUIRED)
--ext_in        STR    Extension of aligned files        (default: .rot)
--R_max         FLOAT  Half-edge of cubic grid (Å)       (default: 5.4)
--step_R        FLOAT  Voxel size (Å)                    (default: 0.10)
--alpha_O       FLOAT  Gaussian exponent for O (Å⁻²)    (default: 1.75)
--alpha_H       FLOAT  Gaussian exponent for H (Å⁻²)    (default: 6.77)
--Do_H          INT    H contribution: +1, -1, or 0      (default: 1)
--thresh        FLOAT  Gaussian exponent cutoff           (default: 14.0)
--weight_thresh FLOAT  Skip clusters below this weight   (default: 1e-6)
--dmin_file     FILE   alignment_dmin.txt from align_clusters.py
                       Used for topology-based exclusion  (default: alignment_dmin.txt)
--exclude_topo  STR    Comma-separated labels to exclude. Each is matched
                       against the Topology column (DODEC, CAGE, MIXED, LIQUID
                       — per-cluster ring topology) AND the Group prefix (CS1,
                       CS2, DODEC, LIQUID — simulation box source, the part
                       before "/" in e.g. CS1/DODEC).  "CS1,CS2,DODEC" thus
                       excludes all clusters from CS1 and CS2 boxes plus any
                       cluster whose own ring topology is DODEC.
--out           STR    Output VESTA .ped file             (default: GRID.ped)
--nthreads      INT    Worker processes                   (default: all CPUs)
--subtract      FILE   Subtract a previously computed .ped grid from the
                       result before writing.  Grids must share the same
                       R_max and step_R.  Useful for difference maps.
--equal_weights        Use uniform weights (1/N) for all active clusters,
                       ignoring the values in --weights_file.  The weight
                       file is still required for the active-cluster list
                       (clusters with weight > weight_thresh are included;
                       all others are excluded as usual).

Requirements: numpy, Python >= 3.6
"""

import argparse
import multiprocessing
import time
from multiprocessing import Pool

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ────────────────────────────────────────────────────────────────────────────

def read_rot(path):
    """Read a .rot (xyz) file.  Returns (labels list, coords array (N,3))."""
    with open(path) as fh:
        n = int(fh.readline())
        fh.readline()
        labels, coords = [], []
        for _ in range(n):
            parts = fh.readline().split()
            labels.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
    return labels, np.array(coords, dtype=np.float64)


def read_weights_from_file(path):
    """
    Read a two-column weight file:  weight  basename
    Blank lines and lines starting with '#' are ignored.
    Weights need not be pre-normalised.
    """
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


def read_labels_from_dmin(path):
    """
    Parse alignment_dmin.txt written by align_clusters.py.
    Returns {basename: {'topo': topology_label, 'group': group_label}}.

    Column layout (the header line starting with '#' is used to locate
    columns, so the function is robust to variable numbers of weight cols):
      Basename  Group  SND(Å²)  SOAP  [weight_T1 ...]  Topology  Rings(3-7)

    The Topology column holds the per-cluster ring topology label
    (DODEC, CAGE, MIXED, LIQUID).

    The Group column holds the sub-group the cluster was assigned to during
    alignment (e.g. CS1/DODEC, CS2/MIXED, LIQUID/LIQUID).  Its prefix
    (before '/') reflects the force-field / trajectory source (CS1, CS2,
    DODEC, LIQUID); its suffix reflects the per-cluster ring topology as
    seen at the time of the topology pre-scan.
    """
    topo_col  = None
    group_col = None
    labels    = {}

    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith('#'):
                parts = stripped.lstrip('#').split()
                for i, p in enumerate(parts):
                    if p.startswith('Topology'):
                        topo_col = i
                    if p == 'Group':
                        group_col = i
                continue

            if topo_col is None:
                continue

            parts = stripped.split()
            bn    = parts[0]
            topo  = parts[topo_col]  if len(parts) > topo_col  else ''
            group = parts[group_col] if (group_col is not None and
                                         len(parts) > group_col) else ''
            labels[bn] = {'topo': topo, 'group': group}

    return labels


def write_ped(path, Dsum, R_max, nR):
    """
    Write VESTA .ped format (cubic grid).
    Loop order matches Fortran: i outer, j middle, k inner, 10 values/line.
    """
    with open(path, 'w') as fh:
        fh.write(' Cubic grid for VESTA\n')
        fh.write(f'{R_max:12.6f}{R_max:12.6f}{R_max:12.6f} 90. 90. 90.\n')
        fh.write(f'{nR:5d}{nR:5d}{nR:5d}\n')
        for i in range(nR):
            for j in range(nR):
                row = Dsum[i, j, :]
                for start in range(0, nR, 10):
                    fh.write(
                        ''.join(f'{v:10.6f}' for v in row[start:start+10])
                        + '\n')


def read_ped(path, expected_nR=None):
    """
    Read a VESTA .ped file written by write_ped().
    Returns (Dsum array (nR,nR,nR), R_max, nR).

    Raises ValueError if the grid dimensions do not match expected_nR
    (when supplied), so that the caller can verify grid compatibility
    before subtracting.
    """
    with open(path) as fh:
        fh.readline()                          # title line
        cell_line = fh.readline().split()
        r_max = float(cell_line[0])            # first cell parameter = R_max
        dim_line = fh.readline().split()
        nR = int(dim_line[0])
        if expected_nR is not None and nR != expected_nR:
            raise ValueError(
                f'Grid in {path!r} has nR={nR} but current grid has '
                f'nR={expected_nR}.  R_max and step_R must match.')
        Dsum = np.zeros((nR, nR, nR), dtype=np.float64)
        values = []
        for line in fh:
            values.extend(float(v) for v in line.split())
    flat = np.array(values, dtype=np.float64)
    if flat.size != nR ** 3:
        raise ValueError(
            f'Expected {nR**3} values in {path!r}, got {flat.size}.')
    Dsum = flat.reshape(nR, nR, nR)
    return Dsum, r_max, nR


# ────────────────────────────────────────────────────────────────────────────
# Worker: process a batch of clusters, return one partial Dsum
# ────────────────────────────────────────────────────────────────────────────

def _process_batch(args):
    """
    Process a contiguous batch of clusters and return the accumulated
    partial Dsum for that batch.

    Each worker handles ~(n_clusters / nthreads) clusters, accumulates
    its own Dsum locally, and returns one grid.  The main process then
    sums nthreads grids — reducing inter-process pipe traffic by a factor
    of ~(clusters / nthreads) compared to returning one grid per cluster.

    Returns (partial_Dsum, n_ok, n_errors, error_list).
    """
    (batch,            # list of (bn, weight) pairs
     ext_in,
     R_max, step_R, nR,
     alpha_O, alpha_H, norm_O, norm_H,
     Do_H, thresh,
     x_vox,
     worker_id) = args

    steps_O = int(np.sqrt(thresh / alpha_O) / step_R) + 1
    steps_H = int(np.sqrt(thresh / alpha_H) / step_R) + 1 if alpha_H > 0 else 0

    Dsum_local = np.zeros((nR, nR, nR), dtype=np.float64)
    errors     = []
    n_ok       = 0

    for bn, weight in batch:
        try:
            labels, coords = read_rot(bn + ext_in)
        except Exception as e:
            errors.append(f'{bn}: {e}')
            continue

        grid  = np.zeros((nR, nR, nR), dtype=np.float64)
        n_O   = 0
        n_H   = 0

        for idx in range(len(labels)):
            if idx == 0:
                continue                    # skip Ar
            lbl      = labels[idx]
            x0, y0, z0 = coords[idx]
            is_O = lbl[0] == 'O'
            is_H = lbl[0] == 'H'
            if is_H and Do_H == 0:
                continue
            if not (is_O or is_H):
                continue
            if abs(x0) > R_max or abs(y0) > R_max or abs(z0) > R_max:
                continue
            if x0*x0 + y0*y0 + z0*z0 > R_max*R_max:
                continue

            if is_O:
                alpha, fact, steps = alpha_O, norm_O, steps_O
                n_O += 1
            else:
                alpha, fact, steps = alpha_H, float(Do_H) * norm_H, steps_H
                n_H += 1

            # Nearest voxel — int() truncates toward zero (matches Fortran)
            ic = nR // 2 + int(x0 / step_R)
            jc = nR // 2 + int(y0 / step_R)
            kc = nR // 2 + int(z0 / step_R)

            i0, i1 = max(0, ic - steps - 1), min(nR, ic + steps + 2)
            j0, j1 = max(0, jc - steps - 1), min(nR, jc + steps + 2)
            k0, k1 = max(0, kc - steps - 1), min(nR, kc + steps + 2)

            # Separable outer product of three 1-D Gaussian profiles
            ex = np.exp(-alpha * (x_vox[i0:i1] - x0) ** 2)
            ey = np.exp(-alpha * (x_vox[j0:j1] - y0) ** 2)
            ez = np.exp(-alpha * (x_vox[k0:k1] - z0) ** 2)

            grid[i0:i1, j0:j1, k0:k1] += (
                fact
                * ex[:, None, None]
                * ey[None, :, None]
                * ez[None, None, :])

        # Per-cluster normalisation
        electron_count = 8 * n_O + abs(Do_H) * n_H
        if electron_count == 0 or grid.sum() < 1e-30:
            errors.append(f'{bn}: no atoms within R_max or zero grid')
            continue

        grid *= electron_count / (grid.sum() * step_R**3)
        Dsum_local += weight * grid
        n_ok += 1

    return Dsum_local, n_ok, len(errors), errors


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    parser.add_argument('--basenames',     default='BASE_NAMES',
                        help='File listing cluster basenames (default: BASE_NAMES)')
    parser.add_argument('--weights_file',  default=None,
                        help='Two-column weight file: weight  basename '
                             '(e.g. 4C_weights.dat). '
                             'Required unless --equal_weights is used.')
    parser.add_argument('--ext_in',        default='.rot',
                        help='Extension of aligned cluster files (default: .rot)')
    parser.add_argument('--R_max',         type=float, default=5.4,
                        help='Half-edge of cubic grid in Å (default: 5.4)')
    parser.add_argument('--step_R',        type=float, default=0.10,
                        help='Voxel size in Å (default: 0.10)')
    parser.add_argument('--alpha_O',       type=float, default=1.75,
                        help='Gaussian exponent for O (Å⁻²) (default: 1.75)')
    parser.add_argument('--alpha_H',       type=float, default=6.77,
                        help='Gaussian exponent for H (Å⁻²) (default: 6.77)')
    parser.add_argument('--Do_H',          type=int,   default=1,
                        help='H contribution: +1, -1, or 0 (default: 1)')
    parser.add_argument('--thresh',        type=float, default=14.0,
                        help='Gaussian exponent cutoff (default: 14.0)')
    parser.add_argument('--weight_thresh', type=float, default=1e-6,
                        help='Skip clusters with weight below this (default: 1e-6)')
    parser.add_argument('--dmin_file',     default='alignment_dmin.txt',
                        help='alignment_dmin.txt from align_clusters.py, used '
                             'for topology-based exclusion '
                             '(default: alignment_dmin.txt)')
    parser.add_argument('--exclude_topo',  default=None,
                        help='Comma-separated labels to exclude. Each label is '
                             'matched against both the Topology column (DODEC, '
                             'CAGE, MIXED, LIQUID) and the Group column prefix '
                             '(the part before "/", e.g. CS1, CS2, DODEC). '
                             'Example: "CS1,CS2,DODEC" excludes all clusters '
                             'from CS1 and CS2 simulation boxes AND any cluster '
                             'whose own ring topology is DODEC. '
                             'Requires --dmin_file.')
    parser.add_argument('--out',           default='GRID.ped',
                        help='Output VESTA .ped file (default: GRID.ped)')
    parser.add_argument('--nthreads',      type=int,   default=None,
                        help='Worker processes (default: all CPUs)')
    parser.add_argument('--subtract',      default=None, metavar='FILE.ped',
                        help='Subtract this .ped grid from the result before '
                             'writing.  Must have the same R_max and step_R '
                             'as the current run.  Produces a difference map.')
    parser.add_argument('--equal_weights', action='store_true',
                        help='Replace all weights with 1/N (uniform) before '
                             'accumulating.  The weight file still determines '
                             'which clusters are active (weight > weight_thresh); '
                             'only the values are replaced.')
    args = parser.parse_args()

    nworkers = args.nthreads or multiprocessing.cpu_count()
    t0 = time.time()

    # ── Grid parameters ───────────────────────────────────────────────────
    R_max  = args.R_max
    step_R = args.step_R
    nR     = 2 * int(R_max / step_R) + 1
    nR3    = nR ** 3

    norm_O = (args.alpha_O / np.pi) ** 1.5
    norm_H = (args.alpha_H / np.pi) ** 1.5 if args.alpha_H > 0 else 0.0
    x_vox  = -R_max + step_R / 2.0 + np.arange(nR) * step_R

    grid_MB = nR3 * 8 / 1e6

    print(f'R_max          : {R_max} Å')
    print(f'step_R         : {step_R} Å')
    print(f'Grid           : {nR}³ = {nR3:,} voxels  ({grid_MB:.1f} MB each)')
    print(f'alpha_O        : {args.alpha_O} Å⁻²'
          f'  (r_cut = {np.sqrt(args.thresh/args.alpha_O):.2f} Å)')
    print(f'alpha_H        : {args.alpha_H} Å⁻²'
          f'  (r_cut = {np.sqrt(args.thresh/args.alpha_H):.2f} Å)')
    print(f'Do_H           : {args.Do_H}  '
          f'({"excluded" if args.Do_H==0 else "positive" if args.Do_H>0 else "negative"})')
    print(f'Workers        : {nworkers}'
          f'  (peak RAM ≈ {(nworkers+1)*grid_MB:.0f} MB)')
    print(f'Equal weights  : {"yes (--equal_weights)" if args.equal_weights else "no (from weights file)"}')
    if args.subtract:
        print(f'Subtract grid  : {args.subtract}')
    print()

    # ── Basenames ─────────────────────────────────────────────────────────
    with open(args.basenames) as fh:
        basenames = [ln.strip() for ln in fh if ln.strip()]
    print(f'Total clusters : {len(basenames)}')

    # ── Weights and topology exclusion (skipped with --equal_weights) ─────
    if not args.equal_weights and args.weights_file is None:
        parser.error('--weights_file is required unless --equal_weights is used.')
    if args.equal_weights and args.weights_file is not None:
        parser.error('--equal_weights and --weights_file are mutually exclusive: '
                     'supply one or the other, not both.')

    # ── Topology-based exclusion (applies to both weighted and equal-weight runs)
    excluded_basenames = set()
    if args.exclude_topo:
        exclude_set = {t.strip().upper() for t in args.exclude_topo.split(',')}
        print(f'Excluding      : {", ".join(sorted(exclude_set))}')
        print(f'  (matched against Topology column AND Group prefix)')
        print(f'Topology from  : {args.dmin_file}')

        try:
            labels_map = read_labels_from_dmin(args.dmin_file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f'--dmin_file {args.dmin_file!r} not found. '
                f'Run align_clusters.py first, or set --dmin_file.')

        n_excluded = 0
        match_counts = {}   # matched_label → count

        for bn in basenames:
            entry        = labels_map.get(bn, {})
            topo         = entry.get('topo',  '').upper()
            group        = entry.get('group', '').upper()
            group_prefix = group.split('/')[0] if '/' in group else group

            matched = None
            if topo in exclude_set:
                matched = topo
            elif group_prefix in exclude_set:
                matched = group_prefix

            if matched:
                excluded_basenames.add(bn)
                n_t = match_counts.get(matched, 0)
                match_counts[matched] = n_t + 1
                n_excluded += 1

        print(f'\n  Excluded {n_excluded} clusters '
              f'({100*n_excluded/len(basenames):.1f}% of total):')
        for label in sorted(match_counts):
            n_t = match_counts[label]
            src = ('Topology' if label in {t.upper() for t in
                   [e.get('topo','') for e in labels_map.values()]}
                   else 'Group prefix')
            print(f'    {label:<8s}: {n_t:6d} clusters  [{src}]')

        n_missing = sum(1 for bn in basenames
                        if bn not in labels_map)
        if n_missing:
            print(f'  WARNING: {n_missing} clusters have no entry '
                  f'in {args.dmin_file} — they are KEPT.')
        print()

    # ── Weights and active-cluster list ───────────────────────────────────
    if args.equal_weights:
        active_basenames = [bn for bn in basenames if bn not in excluded_basenames]
        n_active = len(active_basenames)
        if n_active == 0:
            raise ValueError('No clusters remain after topology exclusion.')
        ew = 1.0 / n_active
        active = [(bn, ew) for bn in active_basenames]
        excl_note = (f'after excluding {len(excluded_basenames)} topology-matched, '
                     if excluded_basenames else '')
        print(f'Active clusters: {n_active}  ({excl_note}--equal_weights)')
        print(f'Equal weights  : yes  (each cluster weighted {ew:.6e})')
    else:
        w_raw = read_weights_from_file(args.weights_file)
        print(f'Weights from   : {args.weights_file}  ({len(w_raw)} entries)')
        weights_raw = np.array([w_raw.get(bn, 0.0) for bn in basenames])

        # Zero out topology-excluded clusters before normalising
        for i, bn in enumerate(basenames):
            if bn in excluded_basenames:
                weights_raw[i] = 0.0

        total = weights_raw.sum()
        if total < 1e-30:
            raise ValueError('All weights are zero after exclusion')
        weights = weights_raw / total

        active = [(bn, float(w))
                  for bn, w in zip(basenames, weights)
                  if w > args.weight_thresh]
        label = 'after exclusion' if args.exclude_topo else f'weight > {args.weight_thresh}'
        print(f'Active clusters: {len(active)}  ({label})')
    print()

    # ── Split into batches, one per worker ────────────────────────────────
    # Contiguous split: each worker gets a roughly equal share of the list.
    # Because weights are sorted by descending weight (largest first in the
    # weight file), a contiguous split is also approximately balanced in
    # compute time (heavier-weighted clusters are not concentrated in one batch).
    n      = len(active)
    splits = np.array_split(np.arange(n), nworkers)
    batches = [[active[i] for i in idx] for idx in splits if len(idx)]

    worker_args = [
        (batch, args.ext_in, R_max, step_R, nR,
         args.alpha_O, args.alpha_H, norm_O, norm_H,
         args.Do_H, args.thresh, x_vox, wid)
        for wid, batch in enumerate(batches)
    ]

    n_batches = len(worker_args)
    print(f'Processing {n} clusters in {n_batches} batches '
          f'(~{n//n_batches} clusters/worker) ...', flush=True)

    # ── Parallel: each worker returns one partial Dsum ────────────────────
    Dsum      = np.zeros((nR, nR, nR), dtype=np.float64)
    all_errors = []
    total_ok   = 0

    with Pool(processes=nworkers) as pool:
        for partial, n_ok, n_err, errs in pool.imap_unordered(
                _process_batch, worker_args):
            Dsum      += partial
            total_ok  += n_ok
            all_errors.extend(errs)
            elapsed = time.time() - t0
            print(f'  batch done  {total_ok:6d}/{n} clusters  '
                  f'{elapsed:6.1f}s  '
                  f'{total_ok/elapsed:6.1f} clusters/s',
                  flush=True)

    elapsed = time.time() - t0
    print(f'\nCompleted in {elapsed:.1f} s  ({total_ok/elapsed:.1f} clusters/s)')

    if all_errors:
        print(f'\n{len(all_errors)} error(s):')
        for e in all_errors[:10]:
            print(f'  {e}')
        if len(all_errors) > 10:
            print(f'  ... and {len(all_errors)-10} more')

    # ── Subtract reference grid if requested ──────────────────────────────
    if args.subtract:
        print(f'\nReading subtraction grid: {args.subtract} ...')
        D_sub, r_sub, nR_sub = read_ped(args.subtract, expected_nR=nR)
        if not np.isclose(r_sub, R_max, rtol=1e-4):
            raise ValueError(
                f'R_max mismatch: current={R_max} Å, '
                f'subtract file={r_sub} Å.  Grids must match.')
        Dsum -= D_sub
        print(f'  Subtracted.  Difference range: '
              f'[{Dsum.min():.4f}, {Dsum.max():.4f}] e/Å³')

    # ── Diagnostics ───────────────────────────────────────────────────────
    print(f'\nIntegrated density : {Dsum.sum() * step_R**3:.4f} e')
    print(f'Peak density       : {Dsum.max():.4f} e/Å³')
    if args.subtract:
        print(f'Min density        : {Dsum.min():.4f} e/Å³'
              f'  (difference map — negative values expected)')

    # ── Write output ──────────────────────────────────────────────────────
    print(f'\nWriting {args.out} ...')
    write_ped(args.out, Dsum, R_max, nR)
    print(f'Written: {nR}×{nR}×{nR} grid, cell {step_R:.3f} Å, '
          f'box ±{R_max:.2f} Å'
          + (f'  [difference map: subtracted {args.subtract}]'
             if args.subtract else ''))


if __name__ == '__main__':
    main()
