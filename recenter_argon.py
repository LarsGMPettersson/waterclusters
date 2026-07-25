#!/usr/bin/env python3
"""
recenter_argon.py
==================
Re-centre Ar at the physical centre of its surrounding water cavity.

Algorithm
---------
A faithful Python port of the Fortran centering cascade (Center_Cluster_Fixed3.f).
The cascade handles four distinct physical situations in order:

  Stage 1 — Mass-weighted CoM then O-centroid (center_cluster)
    Computes the mass-weighted centre of all O and H atoms (not Ar), finds
    all O within R_max of that CoM, takes their centroid as the new Ar
    position.  Accepted only if no oxygen is closer to the new centre than
    to the original Ar position (the "dist1 guard").

  Stage 2 — Iterative relative-variance minimisation (center_cluster2/3)
    Entered if Stage 1 is rejected.  Takes the 20 nearest O atoms to the
    current Ar position and performs a gradient-free 3D walk minimising the
    RELATIVE variance of O-Ar distances:
        σ_rel = sqrt( Σᵢ ((d̄ - dᵢ)/dᵢ)² / N )
    This weights nearby oxygens more than distant ones.  Dynamic NMAX:
    if the outermost O in the working set is farther from d̄ than the
    nearest O, NMAX is reduced by 1 (handles asymmetric cages).
    The dist1 guard applies throughout: any step that would bring any O
    closer than the original nearest O-Ar distance is rejected.
    Stage 2b (center_cluster3) additionally reduces NMAX when the nearest
    O is below 3.0 Å.

  Stage 3 — Repulsion from a single close O (center_cluster4)
    Entered if exactly ONE O remains within 3.0 Å after Stage 2.
    Moves Ar in the direction AWAY from that oxygen until the nearest
    O-Ar distance reaches 3.2 Å.

  Stage 4 — Repulsion from multiple close Os (center_cluster5/6)
    Entered if more than ONE O remains within 3.0 Å after Stage 2.
    center_cluster5: for each close O, finds the most anti-parallel distant
    O and moves in that direction (weighted by 1/dist_close).
    center_cluster6: moves away from close Os weighted by (3.0-dist_close).

Key design choices (from the Fortran, preserved here):
  - Relative variance objective (not absolute std or sphere fit)
  - dist1 guard: never allow any O closer than the original nearest O-Ar
  - Dynamic NMAX: adapt to asymmetric cages
  - Mass-weighted CoM (O at 16, H at 1) as starting point for Stage 1
  - Hard minimum-distance threshold: flag clusters with any O < 2.9 Å

Output
------
One .cen file per cluster: all atoms shifted so the found cavity centre
becomes (0,0,0), which is also Ar's stored position.

Usage
-----
    python recenter_argon.py [options]

    --basenames FILE   File listing cluster base names      (default: BASE_NAMES)
    --ext_in    STR    Input file extension                 (default: .xyz)
    --ext_out   STR    Output file extension                (default: .cen)
    --R_max     FLOAT  O-Ar distance cutoff for Stage 1    (default: 5.5 Å)
    --nthreads  INT    Worker processes                     (default: all CPUs)
    --out       STR    Diagnostic summary file              (default: recentering.txt)
    --short_thresh FLOAT  Flag clusters with any O < this  (default: 2.9 Å)

Requirements: numpy
"""

import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# Physical constants
M_O = 16.0
M_H = 1.0

# Algorithm constants (matching Fortran)
NMAX_DEFAULT  = 20
D_REF         = 3.2    # target minimum O-Ar distance for Stages 3/4
THRESH_CLOSE  = 3.0    # "too close" threshold for Stages 3/4
THRESH5       = 3.0    # center_cluster5 threshold
THRESH6       = 3.1    # center_cluster6 threshold
STEP_FINE     = 0.01   # step size for Stages 1/2
STEP_COARSE   = 0.05   # step size for Stages 3/4
MAX_ITER_2    = 250
MAX_ITER_4    = 100
MAX_ITER_56   = 500


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_xyz(path):
    with open(path) as fh:
        n = int(fh.readline())
        fh.readline()
        labels, coords = [], []
        for _ in range(n):
            parts = fh.readline().split()
            labels.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
    return labels, np.array(coords, dtype=np.float64)


def write_cen(path, labels, coords):
    with open(path, 'w') as fh:
        fh.write(f'{len(labels):5d}\n')
        fh.write('   \n')
        for lbl, xyz in zip(labels, coords):
            fh.write(f'{lbl:<4s}{xyz[0]:12.6f}{xyz[1]:12.6f}{xyz[2]:12.6f}\n')


# ---------------------------------------------------------------------------
# Helper: relative variance of distances
# ---------------------------------------------------------------------------

def rel_variance(dists):
    """Relative variance: sqrt(Σ((mean-d)/d)² / N). Matches Fortran std."""
    mean = dists.mean()
    return float(np.sqrt(np.sum(((mean - dists) / dists) ** 2) / len(dists)))


# ---------------------------------------------------------------------------
# Stage 1: mass-weighted CoM → O-centroid  (center_cluster)
# ---------------------------------------------------------------------------

def stage1_com_centroid(coords, labels, R_max):
    """
    Compute mass-weighted CoM of O+H, find all O within R_max of it,
    take their centroid as the candidate new Ar position.

    Returns (candidate_centre, accepted) where accepted=False if any O
    would be closer to the candidate than to the original Ar.
    """
    ar_pos = coords[0].copy()

    # Mass-weighted CoM of O and H (not Ar)
    R_cm = np.zeros(3)
    M_tot = 0.0
    for i, lbl in enumerate(labels):
        if lbl.startswith('O'):
            R_cm += M_O * coords[i]
            M_tot += M_O
        elif lbl.startswith('H'):
            R_cm += M_H * coords[i]
            M_tot += M_H
    if M_tot < 1e-10:
        return ar_pos.copy(), False
    R_cm /= M_tot

    # Find O atoms within R_max of CoM
    o_idx = [i for i, l in enumerate(labels) if l.startswith('O')]
    o_coords = coords[o_idx]
    dists_to_com = np.linalg.norm(o_coords - R_cm, axis=1)
    in_shell = o_coords[dists_to_com <= R_max]

    if len(in_shell) == 0:
        return ar_pos.copy(), False

    # Centroid of the first-shell O atoms → candidate new centre
    candidate = in_shell.mean(axis=0)

    # dist1 guard: check if any O is closer to candidate than to original Ar
    orig_min = float(np.linalg.norm(o_coords - ar_pos, axis=1).min())
    new_dists = np.linalg.norm(o_coords - candidate, axis=1)
    if new_dists.min() < orig_min:
        return candidate, False

    return candidate, True


# ---------------------------------------------------------------------------
# Stage 2: iterative relative-variance walk  (center_cluster2/3)
# ---------------------------------------------------------------------------

def stage2_rel_var_walk(o_coords_all, R_start, dist1,
                        reduce_if_close=False,
                        step=STEP_FINE, max_iter=MAX_ITER_2,
                        nmax=NMAX_DEFAULT):
    """
    Gradient-free 3D walk minimising relative variance of O-Ar distances.
    Exact port of Fortran center_cluster2/3.

    Key detail: the O atoms are re-sorted by distance to the CURRENT centre
    at the start of EACH axis step (k=1,2,3), not just once per outer
    iteration.  This matches the Fortran exactly and ensures that after an
    accepted move along axis k, the next axis uses the updated O ordering
    from the new position.

    Parameters
    ----------
    o_coords_all   : (M, 3) — all O positions
    R_start        : (3,)   — starting centre (current Ar position)
    dist1          : float  — original nearest O-Ar dist (hard floor)
    reduce_if_close: if True (center_cluster3), also reduce NMAX when
                    nearest < 3.0 Å before iterating
    """
    R = R_start.copy()
    n_O_all = len(o_coords_all)

    def sorted_nearest(centre):
        d = np.linalg.norm(o_coords_all - centre, axis=1)
        order = np.argsort(d)
        return o_coords_all[order], d[order]

    # Initial sort to set up dynamic NMAX
    xyz_w, d_w = sorted_nearest(R)
    nmax_cur = nmax

    # Dynamic NMAX: reduce if outermost O is farther from mean than nearest
    mean_d = d_w[:nmax_cur].mean()
    while (nmax_cur > 1 and
           abs(mean_d - d_w[nmax_cur - 1]) > abs(mean_d - d_w[0])):
        nmax_cur -= 1
        mean_d = d_w[:nmax_cur].mean()

    # Also reduce NMAX when nearest O < 3.0 Å (center_cluster3 mode)
    if reduce_if_close and d_w[0] < THRESH_CLOSE:
        while nmax_cur > 1 and d_w[0] < THRESH_CLOSE:
            nmax_cur -= 1

    std_min = rel_variance(d_w[:nmax_cur])

    for iteration in range(max_iter):
        changes = 0

        for k in range(3):
            # --- Re-sort O by distance to CURRENT R (Fortran-exact) ---
            xyz_w, d_w = sorted_nearest(R)

            # Direction = sum of (O_i - R) for the NMAX nearest
            shift_k = float(np.sum(xyz_w[:nmax_cur, k] - R[k]))

            # Step: sign of shift, but flip if it moves TOWARD nearest O
            delta = np.sign(shift_k) * step
            if delta * (xyz_w[0, k] - R[k]) > 0:
                delta = -delta

            R_trial = R.copy()
            R_trial[k] += delta

            # Evaluate trial position
            d_trial = np.sort(
                np.linalg.norm(o_coords_all - R_trial, axis=1))

            # dist1 guard: reject if any O moves closer than original nearest
            if d_trial[0] < dist1:
                continue

            std_trial = rel_variance(d_trial[:nmax_cur])
            if std_trial < std_min:
                std_min = std_trial
                R = R_trial
                changes += 1

        if changes == 0:
            break

    return R


# ---------------------------------------------------------------------------
# Stage 3: repulsion from single close O  (center_cluster4)
# ---------------------------------------------------------------------------

def stage3_single_repulsion(o_coords_all, R_start, dist1):
    """
    Move Ar away from the nearest O until all O-Ar distances >= D_REF.
    Matches Fortran center_cluster4.

    Critically: dist1 is RESET to the current nearest O-Ar distance at
    entry (not the original pre-centering distance).  This matches the
    Fortran, which initialises dist1 = dist_O(1) from the current position.
    The guard then means "don't let any O get closer than it currently is"
    — a soft, incrementally-updated floor — rather than "never cross the
    original pre-centering distance", which would block Stage 3 entirely
    when the original Ar was already inside 2.9 Å of an oxygen.
    """
    R = R_start.copy()

    # Re-initialise dist1 from CURRENT position (Fortran-exact)
    dists_current = np.linalg.norm(o_coords_all - R, axis=1)
    dist1 = float(np.sort(dists_current)[0])

    for _ in range(MAX_ITER_4):
        dists = np.linalg.norm(o_coords_all - R, axis=1)
        if dists.min() >= D_REF:
            break

        # Direction away from nearest O
        nearest_idx = int(np.argmin(dists))
        direction = R - o_coords_all[nearest_idx]
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            break
        direction /= norm

        R_trial = R + STEP_FINE * direction

        # dist1 guard: don't let any O get closer than current nearest
        d_trial = np.linalg.norm(o_coords_all - R_trial, axis=1)
        if d_trial.min() < dist1:
            break

        R = R_trial
        # Update dist1 to current nearest (Fortran: dist1 stays fixed,
        # but the check is against the INITIAL dist_O(1) at entry which
        # is already the current minimum — so this is equivalent)

    return R


# ---------------------------------------------------------------------------
# Stage 4a: anti-parallel repulsion (center_cluster5)
# ---------------------------------------------------------------------------

def stage4a_antiparallel(o_coords_all, R_start, dist1):
    """
    For each close O, find the most anti-parallel distant O and move
    in that direction (weighted by 1/dist_close).
    Matches center_cluster5.
    dist1 re-initialised from current position on entry.
    """
    R = R_start.copy()
    # Re-initialise dist1 from current position (Fortran-exact)
    dist1 = float(np.sort(np.linalg.norm(o_coords_all - R, axis=1))[0])

    for _ in range(MAX_ITER_56):
        dists = np.linalg.norm(o_coords_all - R, axis=1)
        order = np.argsort(dists)
        dists_sorted = dists[order]
        xyz_sorted = o_coords_all[order]

        nmax_cur = NMAX_DEFAULT
        close_mask = dists_sorted[:nmax_cur] < THRESH5
        n_close = int(close_mask.sum())
        if n_close == 0:
            break

        # Normalised direction vectors
        dirs = (xyz_sorted[:nmax_cur] - R)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        dirs_norm = dirs / norms

        # For each close O, find most anti-parallel distant O
        shift = np.zeros(3)
        for i in range(n_close):
            # Overlap with all distant O (indices n_close..nmax_cur)
            overlaps = dirs_norm[n_close:] @ dirs_norm[i]
            if len(overlaps) == 0:
                continue
            pick = int(np.argmin(overlaps)) + n_close
            shift += dirs_norm[pick] / dists_sorted[i]

        norm = np.linalg.norm(shift)
        if norm < 1e-12:
            break
        shift /= norm

        R_trial = R + STEP_COARSE * shift
        d_trial = np.linalg.norm(o_coords_all - R_trial, axis=1)
        if d_trial.min() < dist1:
            break

        R = R_trial

    return R


# ---------------------------------------------------------------------------
# Stage 4b: weighted repulsion from close Os  (center_cluster6)
# ---------------------------------------------------------------------------

def stage4b_weighted_repulsion(o_coords_all, R_start, dist1):
    """
    Move Ar away from close O atoms, weighted by (thresh - dist_close).
    Matches center_cluster6.
    dist1 re-initialised from current position on entry.
    """
    R = R_start.copy()
    # Re-initialise dist1 from current position (Fortran-exact)
    dist1 = float(np.sort(np.linalg.norm(o_coords_all - R, axis=1))[0])

    for _ in range(MAX_ITER_56):
        dists = np.linalg.norm(o_coords_all - R, axis=1)
        order = np.argsort(dists)
        dists_sorted = dists[order]
        xyz_sorted = o_coords_all[order]

        nmax_cur = NMAX_DEFAULT
        close_mask = dists_sorted[:nmax_cur] < THRESH6
        n_close = int(close_mask.sum())
        if n_close == 0:
            break

        # Update dist1 if minimum has increased (Fortran line: if dd > dist1: dist1=dd)
        cur_min = dists_sorted[0]
        if cur_min > dist1:
            dist1 = cur_min

        # Normalised direction vectors
        dirs = (xyz_sorted[:nmax_cur] - R)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        dirs_norm = dirs / norms

        # Move away from close Os, weighted by (thresh - dist)
        shift = np.zeros(3)
        for i in range(n_close):
            shift -= dirs_norm[i] * (THRESH6 - dists_sorted[i])

        norm = np.linalg.norm(shift)
        if norm < 1e-12:
            break
        shift /= norm

        R_trial = R + STEP_COARSE * shift
        d_trial = np.linalg.norm(o_coords_all - R_trial, axis=1)
        if d_trial.min() < dist1:
            break

        R = R_trial

    return R


# ---------------------------------------------------------------------------
# Main centering function: full cascade
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Single cascade from one starting point (used by multi-start)
# ---------------------------------------------------------------------------

def _run_cascade(o_coords, R_start, dist1):
    """
    Run Stage 2 → 2b → 3/4 from a given starting position.
    Returns (final_pos, rel_var, min_dist, n_close_final, stage_label).
    """
    pos = stage2_rel_var_walk(o_coords, R_start, dist1, reduce_if_close=False)
    stage = 'S2'

    dists = np.linalg.norm(o_coords - pos, axis=1)
    n_close = int(np.sum(dists < THRESH_CLOSE))

    if n_close > 0:
        pos = stage2_rel_var_walk(o_coords, pos, dist1, reduce_if_close=True)
        stage = 'S2b'
        dists = np.linalg.norm(o_coords - pos, axis=1)
        n_close = int(np.sum(dists < THRESH_CLOSE))

        if n_close == 1:
            pos = stage3_single_repulsion(o_coords, pos, dist1)
            stage = 'S3'
        elif n_close > 1:
            pos = stage4a_antiparallel(o_coords, pos, dist1)
            pos = stage4b_weighted_repulsion(o_coords, pos, dist1)
            stage = 'S4'

    dists_final = np.sort(np.linalg.norm(o_coords - pos, axis=1))
    std    = rel_variance(dists_final[:NMAX_DEFAULT])
    min_d  = float(dists_final[0])
    n_cl   = int(np.sum(dists_final < THRESH_CLOSE))
    return pos, std, min_d, n_cl, stage


# ---------------------------------------------------------------------------
# Main centering function: Stage 1 + multi-start Stage 2 cascade
# ---------------------------------------------------------------------------

def centre_cavity(labels, coords, R_max, short_thresh, n_starts=20,
                  perturb_min=0.05, perturb_max=0.30, seed=42):
    """
    Full Fortran-equivalent centering cascade with multi-start Stage 2.

    Stage 2 is a gradient-free walk on a non-convex landscape and can
    settle in different local minima depending on the starting point.
    To find the best basin (minimum relative variance with no O < 3.0 Å),
    Stage 2+ is run from:
      - the Stage 1 result (or original Ar position if Stage 1 rejected)
      - n_starts random perturbations around the Stage 1 result

    The winner is selected by: (1) fewest O atoms within 3.0 Å, then
    (2) lowest relative variance.  This gives a well-defined, reproducible
    centre that is robust against the local-minimum sensitivity that causes
    small differences between the Fortran and Python implementations.

    Parameters
    ----------
    n_starts     : number of random perturbations for multi-start
    perturb_min  : minimum perturbation step (Å)
    perturb_max  : maximum perturbation step (Å)
    seed         : RNG seed for reproducibility
    """
    ar_pos = coords[0].copy()
    o_idx = [i for i, l in enumerate(labels) if l.startswith('O')]
    o_coords = coords[o_idx]

    dists_orig = np.linalg.norm(o_coords - ar_pos, axis=1)
    dist1 = float(np.sort(dists_orig)[0])

    # ---------------------------------------------------------------- Stage 1
    candidate, accepted = stage1_com_centroid(coords, labels, R_max)
    s1_pos = candidate if accepted else ar_pos.copy()
    stage = 'S1' if accepted else 'S0'

    # ---------------------------------------------------------------- Multi-start Stage 2+
    # Run cascade from Stage 1 result and n_starts perturbations around it.
    rng = np.random.default_rng(seed)
    starts = [s1_pos]
    for _ in range(n_starts):
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        step = perturb_min + (perturb_max - perturb_min) * rng.random()
        starts.append(s1_pos + direction * step)

    best_pos   = None
    best_std   = np.inf
    best_min   = 0.0
    best_nc    = 999
    best_stage = 'S2'

    for R_start in starts:
        pos, std, min_d, nc, stg = _run_cascade(o_coords, R_start, dist1)
        # Select: (1) fewer close contacts, (2) lower relative variance
        if (nc < best_nc) or (nc == best_nc and std < best_std):
            best_pos, best_std, best_min, best_nc, best_stage = (
                pos, std, min_d, nc, stg)

    new_pos = best_pos
    if best_stage != 'S0':
        stage = best_stage

    shift_mag = float(np.linalg.norm(new_pos - ar_pos))
    short     = best_min if best_min < short_thresh else None

    return new_pos, shift_mag, short, stage


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _process_one(args):
    bn, ext_in, ext_out, R_max, short_thresh, n_starts, p_min, p_max = args
    path_in = bn + ext_in
    try:
        labels, coords = read_xyz(path_in)
    except FileNotFoundError:
        return bn, None, None, None, f'File not found: {path_in}'
    except Exception as e:
        return bn, None, None, None, f'Read error: {e}'

    if not (labels[0].startswith('Ar') or labels[0].upper() == 'AR'):
        return bn, None, None, None, f'Atom 1 is not Ar (found "{labels[0]}")'

    o_count = sum(1 for l in labels if l.startswith('O'))
    if o_count < NMAX_DEFAULT:
        return bn, None, None, None, f'Only {o_count} O atoms, need {NMAX_DEFAULT}'

    try:
        new_ar_pos, shift_mag, short_dist, stage = centre_cavity(
            labels, coords, R_max, short_thresh,
            n_starts=n_starts, perturb_min=p_min, perturb_max=p_max)
    except Exception as e:
        return bn, None, None, None, f'Centering error: {e}'

    new_coords = coords - new_ar_pos
    new_coords[0] = np.zeros(3)
    write_cen(bn + ext_out, labels, new_coords)
    return bn, shift_mag, short_dist, stage, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    parser.add_argument('--basenames',    default='BASE_NAMES')
    parser.add_argument('--ext_in',       default='.xyz')
    parser.add_argument('--ext_out',      default='.cen')
    parser.add_argument('--R_max',        type=float, default=5.5,
                        help='O-Ar cutoff for Stage 1 CoM/centroid (Å) '
                             '(default: 5.5)')
    parser.add_argument('--nthreads',     type=int, default=None)
    parser.add_argument('--out',          default='recentering.txt')
    parser.add_argument('--short_thresh', type=float, default=2.9,
                        help='Flag clusters with any O-Ar < this (Å) '
                             '(default: 2.9)')
    parser.add_argument('--n_starts',     type=int, default=20,
                        help='Number of random starting points for multi-start '
                             'Stage 2 (in addition to the Stage 1 result). '
                             'Higher values reduce sensitivity to local minima '
                             'but increase runtime proportionally. '
                             '(default: 20)')
    parser.add_argument('--perturb_min',  type=float, default=0.05,
                        help='Minimum perturbation step for multi-start (Å) '
                             '(default: 0.05)')
    parser.add_argument('--perturb_max',  type=float, default=0.30,
                        help='Maximum perturbation step for multi-start (Å). '
                             'Should be smaller than the cavity radius to keep '
                             'starts physically sensible. (default: 0.30)')
    args = parser.parse_args()

    nworkers = args.nthreads or multiprocessing.cpu_count()
    t0 = time.time()

    with open(args.basenames) as fh:
        basenames = [ln.strip() for ln in fh if ln.strip()]
    nspec = len(basenames)

    print(f'Number of clusters : {nspec}')
    print(f'Worker processes   : {nworkers}')
    print(f'Input  extension   : {args.ext_in}')
    print(f'Output extension   : {args.ext_out}')
    print(f'R_max (Stage 1)    : {args.R_max} Å')
    print(f'Short-dist flag    : < {args.short_thresh} Å')
    print(f'Multi-start        : {args.n_starts} random starts + Stage 1 result')
    print(f'Perturbation range : {args.perturb_min}–{args.perturb_max} Å')
    print(f'Algorithm          : Stage 1 → multi-start Stage 2 → Stage 2b → 3/4')
    print()
    print('Centering all clusters ...', flush=True)

    worker_args = [(bn, args.ext_in, args.ext_out, args.R_max,
                    args.short_thresh, args.n_starts,
                    args.perturb_min, args.perturb_max)
                   for bn in basenames]

    results = {}
    errors  = []
    shorts  = []

    chunksize = max(1, nspec // (nworkers * 4))
    with ProcessPoolExecutor(max_workers=nworkers) as pool:
        done = 0
        for bn, shift, short_d, stage, err in pool.map(
                _process_one, worker_args, chunksize=chunksize):
            done += 1
            if err:
                errors.append((bn, err))
            else:
                results[bn] = (shift, stage)
                if short_d is not None:
                    shorts.append((bn, short_d))
            if done % max(1, nspec // 10) == 0 or done == nspec:
                print(f'  {done}/{nspec} processed', flush=True)

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed:.1f} s')

    if errors:
        print(f'\n{len(errors)} cluster(s) failed:')
        for bn, err in errors[:20]:
            print(f'  {bn}: {err}')

    if not results:
        print('\nNo valid results.')
        return

    shifts = np.array([v[0] for v in results.values()])
    stages = [v[1] for v in results.values()]
    from collections import Counter
    stage_counts = Counter(stages)

    print(f'\n=== Centering summary over {len(results)} clusters ===')
    print(f'\nShift from original Ar position:')
    print(f'  Average : {shifts.mean():.4f} Å   Std : {shifts.std():.4f} Å')
    print(f'  Min     : {shifts.min():.4f} Å   Max : {shifts.max():.4f} Å')
    for p in [50, 75, 90, 95, 99]:
        print(f'  P{p:2d}     : {np.percentile(shifts, p):.4f} Å')

    print(f'\nStage breakdown:')
    stage_desc = {'S1':'CoM+centroid', 'S2':'rel-var walk',
                  'S2b':'rel-var walk (close O)', 'S3':'single repulsion',
                  'S4':'multi-O repulsion'}
    for s in ['S1','S2','S2b','S3','S4']:
        cnt = stage_counts.get(s, 0)
        if cnt:
            print(f'  {s} ({stage_desc[s]:25s}): '
                  f'{cnt:6d} ({100*cnt/len(results):.1f}%)')

    print(f'\nShort O-Ar distances (< {args.short_thresh} Å): '
          f'{len(shorts)} ({100*len(shorts)/len(results):.2f}%)')
    if shorts:
        print(f'  (Shortest: {min(shorts, key=lambda x: x[1])[1]:.4f} Å)')

    sorted_bns = sorted(results.keys(), key=lambda bn: -results[bn][0])
    with open(args.out, 'w') as fh:
        fh.write('# Ar cavity centering (Fortran-cascade algorithm)\n')
        fh.write(f'# R_max={args.R_max}  short_thresh={args.short_thresh}\n')
        fh.write(f'# Sorted by shift magnitude (largest first)\n')
        fh.write(f'# {"Basename":<60s}  {"Shift(Å)":>10s}  {"Stage":>5s}\n')
        for bn in sorted_bns:
            shift, stage = results[bn]
            fh.write(f'  {bn:<60s}  {shift:10.4f}  {stage:>5s}\n')
        if shorts:
            fh.write(f'\n# Short O-Ar distances (< {args.short_thresh} Å):\n')
            for bn, d in sorted(shorts, key=lambda x: x[1]):
                fh.write(f'#   {d:.4f} Å  {bn}\n')
        if errors:
            fh.write(f'\n# {len(errors)} cluster(s) with errors:\n')
            for bn, err in errors:
                fh.write(f'#   {bn}: {err}\n')

    print(f'\nFull per-cluster results written to: {args.out}')
    print(f'Re-centred clusters written to *{args.ext_out} files')


if __name__ == '__main__':
    main()
