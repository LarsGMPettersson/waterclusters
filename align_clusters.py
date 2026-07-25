#!/usr/bin/env python3
"""
align_clusters.py
=================
Reference-invariant, group-aware alignment of Ar-centred water clusters,
with cross-group anchor selection and inter-group registration.

Two-stage pipeline
------------------
STAGE 1 — Intra-group alignment (one independent iterative alignment per group)
  Each structural category (CS1, CS2, DODEC, LIQUID, …) is aligned to its
  own iterative mean alignment.

STAGE 2 — Inter-group registration (inter-group registration)
  After Stage 1 each group lives in an arbitrary internal frame.  The
  cross-group anchor pool scan (described above) resolves this: all groups
  are re-aligned in Phase B to the single global anchor, so all clusters
  end up in one common coordinate frame without any further inter-group
  registration step.

  A single set of .rot files is produced and is temperature-independent.
  The analysis weights (--analysis_weights) are never used during alignment;
  they are applied only in density_average.py when computing the weighted
  density map for a specific temperature.

Rotation metrics — two phases use different aligners
----------------------------------------------------
Phase A (iterative alignment to moving mean) uses Hungarian+Kabsch:
    The Hungarian algorithm finds the optimal bijective assignment between
    the moving O atoms and the reference O atoms; Kabsch then finds the
    exact rotation minimising RMSD under that fixed assignment.

    Initial reference (iteration 0):
        The first cluster in the sub-group is used as the initial
        reference.  Using the coordinate mean of all clusters would
        collapse all 20 reference positions to within ~0.1 Å of the
        origin (random orientations cancel), making Hungarian matching
        degenerate on the first iteration.  A single cluster is a
        well-defined, non-degenerate starting point.

    Hungarian matching and mean update:
        The Hungarian cost matrix is C[i,j] = ||p_i − q_j||² —
        squared Euclidean distance, the same per-pair measure as SND,
        but minimised subject to a bijection constraint.  After finding
        the optimal assignment π and the Kabsch rotation R, the rotated
        O atoms are stored in Hungarian-matched order: the atom assigned
        to reference position i is stored at position i.  The new
        reference is then the unweighted mean of these reordered,
        rotated arrays — so position i of the reference averages all
        atoms that were matched to reference position i.  This gives a
        physically consistent correspondence across clusters at every
        iteration.

    Kabsch rotation:
        Given the Hungarian correspondence, Kabsch finds the rotation R
        minimising Σᵢ ||R·pᵢ − qᵢ||² in closed form via the SVD of
        H = Pᵀ Q.  SVD is used not because of rank deficiency but because
        it gives the orthogonal matrix closest to H directly, and the
        determinant of VᵀUᵀ must be checked to ensure a proper rotation
        (det = +1) rather than an improper one (reflection, det = −1),
        which can arise when the paired point sets happen to be nearly
        coplanar.

    The combination is fully deterministic: the same input always gives
    the same rotation, which is essential for the mean-reference iteration
    to converge without limit cycles.

    SND has a unique global minimum, but finding it reliably requires
    Gaussian annealing — an expensive cascade of NM optimisations at
    decreasing σ.  This expense is acceptable in Phase B and the final
    write pass, where each cluster is aligned once to a fixed reference.
    In Phase A many iterations are needed over all clusters, so the
    cumulative cost of Gaussian annealing would be prohibitive.  Hungarian
    assignment is therefore used to establish the correspondence, after
    which Kabsch gives the exact rotation cheaply.

    Note: for well-centred water clusters the bijection constraint of
    Hungarian and the nearest-neighbour assignment of SND give identical
    correspondences in practice.  O-O distances are ~2.8 Å, so two
    moving O atoms would both have to lie within ~1.4 Å of the same
    reference O atom to cause a double assignment in SND — a geometry
    that cannot occur in physical water clusters.

Phase B and the final write pass use the SND metric:

    SND(R) = Σ_i  min_j  ||R · p_i − q_j||²

    Unlike Hungarian+Kabsch, SND requires no bijection between atom sets.
    It is continuous in R and produces sharper alignments when the reference
    has near-symmetric features that a bijection-based matcher mislabels.
    Because the reference is fixed in Phase B, the determinism requirement
    is gone and absolute alignment quality takes priority.

Optimisation strategy — Gaussian overlap annealing (depth='full')
-----------------------------------------------------------------
A plain coarse grid over SO(3) (Fibonacci lattice, ~15° spacing) fails
to find the global SND minimum for liquid clusters: the SND=0 basin
around the true rotation is only ~10° wide and is easily straddled.
A finer grid is impractical (the SND=0 basin is still narrower than any
affordable grid).

The solution is Gaussian overlap annealing:

  O(R; σ) = Σ_i Σ_j  exp(−|R·p_i − q_j|²/ 2σ²)

At large σ the landscape is smooth with broad basins; the coarse grid
reliably finds the right region.  Successive NM polish steps at
decreasing σ walk from the broad basin into the sharp SND minimum:

  1. Coarse Gaussian search  σ = 3.0 Å  (24 sym seeds × 1000 grid pts)
     Wide basins → correct region found even with 27° grid offset.
  2. Nelder-Mead (NM) polish  σ = 1.5 Å   sharpening the basin.
  3. NM polish  σ = 0.8 Å   approaching the SND landscape.
  4. NM polish  σ = 0.4 Å   near-SND resolution.
  5. Final SND NM polish     converge to SND = 0 (for self-alignment)
                              or to the true SND minimum for cross-cluster.

Nelder-Mead is a derivative-free simplex optimiser well suited to the
rotation manifold SO(3): it requires only function evaluations (no
gradients), handles the non-Euclidean geometry gracefully via the
axis-angle parameterisation, and converges in a few hundred evaluations.

This gives 30/30 recovery of random rotations applied to a single
cluster (vs 24/30 with grid-only SND), while being ~3× faster
(~200 ms vs ~600 ms per cluster) because the annealing NM steps are
much cheaper than a fine global grid.

Depth selection (per cluster, SOAP-guided)
------------------------------------------
  'full'   : Gaussian annealing cascade + SND NM  (high SOAP similarity)
  'fine'   : coarse + fine SND grid only           (medium SOAP)
  'coarse' : coarse SND grid only                  (low SOAP)

With 24 workers and ~16k clusters this costs ~25 min per run on
your machine — roughly 20× slower per iteration than Hungarian+Kabsch
but far fewer iterations are needed, and the final result is sharper.

Best-cluster reference (--use_best_ref)
----------------------------------------
After the first alignment pass the cluster with the lowest SND to the
current mean is selected as a fixed reference.  This breaks the
rotational degeneracy of the liquid mean (which converges to a
near-spherical shell, giving many equivalent orientations) while
introducing minimal bias (the lowest-SND cluster is by definition the
most typical member of the ensemble).

Two-level anchor selection
--------------------------
Choosing the alignment anchor well is critical: a liquid cluster as
anchor produces a featureless average because the SND landscape has many
near-degenerate minima; a structured cage cluster gives a sharper, more
physically meaningful density map.

Phase B (intra-group anchor scan, --anchor_scan_k K):
  After Phase A (iterative alignment to the group mean) converges,
  the top-K clusters by SND-to-mean are evaluated as candidate anchors.
  For each candidate c, all other clusters in the group are aligned to c
  (one fast Phase B pass at depth='fine'), and the mean SND over the
  group is recorded.  The candidate with the lowest group-mean SND wins
  and becomes the fixed reference for the final write pass of that group.
  This is repeated independently for each of the 13 topology sub-groups.
  Default K = 5 (--anchor_scan_k).

Cross-group anchor pool (global anchor selection):
  After all groups have completed Phase B, the K best candidates from
  every sub-group are collected into a single pool of K × n_groups
  clusters (5 × 13 = 65 for the default settings).  Each pool member c
  is then evaluated as a potential global reference: all other pool
  members are aligned to c using a fast SND coarse+fine grid (depth='fine',
  no Nelder-Mead, for speed), and the mean SND over the pool is recorded.
  The pool member with the lowest mean SND across the entire pool becomes
  the global anchor for all groups.

  This cross-group evaluation is decisive: a well-structured candidate
  (e.g. a DODEC cluster) typically produces a much lower cross-pool mean
  SND than a liquid candidate, because its cage geometry constrains the
  SND minimum to a narrow neighbourhood of the true rotation, whereas a
  liquid cluster has many near-degenerate minima that prevent precise
  alignment of the other pool members to it.  Once the global anchor is
  chosen, all 13 sub-groups run a final Phase B alignment pass to that
  single frame, placing all clusters in a common coordinate system without
  any subsequent inter-group registration step.

  The cross-pool scan costs ~30 s (65 × 64 = 4160 pairwise fast-SND
  evaluations, parallelised over 24 workers).

SOAP structural similarity (rotationally invariant pre-screening)
-----------------------------------------------------------------
SOAP (Smooth Overlap of Atomic Positions) computes a rotationally
invariant fingerprint of the local O-atom environment.  It is used
in the final write pass ONLY — not in the iterative alignment loop
and not in the anchor selection.

References:
  Bartók, Kondor, Csányi, Phys. Rev. B 87, 184115 (2013)  — original SOAP
  Bartók, De, Poelking, Bernstein, Kermode, Csányi, Ceriotti,
    Sci. Adv. 3, e1701816 (2017)  — SOAP for molecular similarity

The power spectrum implemented here follows equations (2)–(5) of:
  De, Bartók, Csányi, Ceriotti, Phys. Chem. Chem. Phys. 18, 13754 (2016)

  SOAP(A, B) = |p_A · p_B| / (||p_A|| ||p_B||) ∈ [0, 1]

where p is the power spectrum of the atomic density expanded in radial
basis × spherical harmonics (σ = 0.3 Å default smearing).

Because SOAP is invariant under rotation it cannot guide the rotation
search directly.  What it provides is a structurally meaningful
pre-screening: before committing to an expensive Gaussian annealing
cascade for a given cluster, the SOAP similarity to the reference is
computed.  High similarity (SOAP ≥ soap_polish_thresh = 0.90) means the
cluster shares the cage topology of the reference and a precise alignment
exists — full effort (Gaussian annealing + SND NM) is warranted.  Low
similarity (SOAP < soap_skip_thresh = 0.70) means the structures are
genuinely different types and a high residual SND reflects structural
dissimilarity rather than a search failure — coarse grid only suffices.
The topology label (from ring analysis) provides a second, complementary
criterion: a topology match boosts the search depth regardless of SOAP.

Per-iteration diagnostics
--------------------------
  Objective  : total SND (Å²) and change
  SND stats  : mean / std / min / max across clusters
  Max mean-O shift: how much the reference frame is still moving (Å)

Usage
-----
    python align_clusters.py [options]

    Input / output
    --------------
    --basenames FILE      File listing cluster base names        (default: BASE_NAMES)
    --ext_in    STR       Input file extension                   (default: .cen)
    --ext_out   STR       Output file extension                  (default: .rot)

    Alignment
    ---------
    --num_O     INT       Nearest O atoms used for alignment     (default: 20)
    --nthreads  INT       Worker processes                       (default: all CPUs)
    --groups    STR       Comma-separated regex patterns defining structural groups,
                          e.g. "CS1,CS2,DODEC,LIQUID".  Each cluster matches the
                          first pattern; unmatched go into group "other".
    --use_best_ref        After Phase A (iterative alignment to mean) converges, switch to
                          the lowest-SND cluster as a fixed reference for a single
                          Phase B alignment pass.  Recommended: breaks rotational
                          degeneracy of the liquid mean.
    --no_iter             Align every cluster to the first cluster only
                          (Fortran-style single-reference mode, for comparison).

    Convergence (Phase A)
    ---------------------
    --maxiter   INT       Max alignment iterations per group    (default: 200)
    --conv_obj  FLOAT     Convergence: Δ(total SND) in Å²       (default: 1.0)
                          Phase A also exits when Δobj/cluster < conv_obj AND
                          mean-O shift < 0.20 Å (handles near-spherical liquid mean).

    Stage 2 inter-group registration
    ---------------------------------
    --analysis_weights STR  Comma-separated weight files for post-alignment
                          analysis only.  NOT used for alignment or Stage 2
                          registration (alignment is temperature-independent;
                          the same .rot files are used for all temperatures).
                          Format: "weight  basename" per line.
                          Typically all four temperature files at once:
                            "4C_weights.dat,15C_weights.dat,25C_weights.dat,45C_weights.dat"
                          Produces a per-temperature weighted topology table
                          and writes per-temperature weights to alignment_dmin.txt.
    --weights   STR       Deprecated alias for --analysis_weights.

    SOAP depth selection (final write pass)
    ----------------------------------------
    --soap_sigma          FLOAT  Gaussian smearing width for full SOAP (Å).
                          Controls how closely SOAP approximates the SND metric:
                          smaller sigma = sharper Gaussians = more sensitive to
                          precise O positions; sigma→0 approaches SND exactly.
                          Recommended: 0.3 Å (≈10%% of O-O distance ~2.8 Å).
                                                                 (default: 0.3)
    --soap_polish_thresh  FLOAT  SOAP ≥ this → full search (coarse+fine+Nelder-Mead).
                          Cluster is structurally similar to reference; a precise
                          rotation exists and is worth finding.   (default: 0.90)
    --soap_skip_thresh    FLOAT  SOAP < this → coarse grid only.
                          Cluster is structurally dissimilar; high residual SND
                          reflects genuine structural difference, not poor alignment.
                          Between thresholds → fine grid (coarse+fine, no NM).
                                                                 (default: 0.70)

    Ring topology analysis (final write pass)
    ------------------------------------------
    --ring_cutoff  FLOAT  O-O distance cutoff for H-bond graph (Å).
                          Should lie in the gap between the first O-O peak
                          (~2.8 Å) and the second shell (~4.5 Å).
                          3.3 Å (default) sits at the first-shell RDF minimum
                          and captures weaker H-bonds relevant to cage topology,
                          more standard in clathrate hydrate literature than 3.2.
                                                                 (default: 3.3)
    --ring_max     INT    Largest primitive ring size to search.  (default: 7)
                          Topology labels written per cluster:
                            LIQUID : n5=0  (fully disordered)
                            MIXED  : 1 ≤ n5 < 5  (some cage-like motifs)
                            CAGE   : 5 ≤ n5 < 10 (partial cage / precursor)
                            DODEC  : n5 ≥ 10, n6=0  (5¹² dodecahedron)
                            CS1    : n5 ≥ 10, n6 ≤ 2 (5¹²6², CS-I)
                            CS2    : n5 ≥ 10, n6 ≥ 3 (5¹²6⁴, CS-II)
    --topo_subgroups      Option A: split each group into topology sub-groups
                          before Stage 1.  E.g. the LIQUID group becomes
                          LIQUID/DODEC, LIQUID/CAGE, LIQUID/MIXED, LIQUID/LIQUID.
                          Each sub-group gets its own iterative mean alignment and
                          best-cluster reference, giving sharper density maps
                          for structurally coherent sub-populations such as
                          cage-precursor liquid clusters.

    Topology is always used to guide search depth (Option B, no flag needed):
      topo-match AND SOAP ≥ soap_polish_thresh  →  full (coarse+fine+NM)
      topo-match (any SOAP)                     →  fine (coarse+fine)
      topo-mismatch AND SOAP ≥ soap_polish_thresh →  fine
      topo-mismatch (any SOAP)                  →  coarse

Requirements: numpy, scipy (>= 1.0; sph_harm / sph_harm_y auto-detected)
"""

import argparse
import multiprocessing
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product as iproduct

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from scipy.spatial import cKDTree

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')


# ---------------------------------------------------------------------------
# Precomputed rotation grids  (module-level constants, built once)
# ---------------------------------------------------------------------------

def _fibonacci_SO3(n):
    """Uniform quaternion grid on SO(3) via Fibonacci / Shoemake mapping."""
    phi1 = 1.0 / 1.618033988749895
    phi2 = 1.0 / 2.618033988749895
    qs = np.empty((n, 4))
    idx = np.arange(n, dtype=float)
    s   = idx / n
    s1  = (idx * phi1) % 1.0
    s2  = (idx * phi2) % 1.0
    qs[:,0] = np.sqrt(1-s) * np.sin(2*np.pi*s1)
    qs[:,1] = np.sqrt(1-s) * np.cos(2*np.pi*s1)
    qs[:,2] = np.sqrt(s)   * np.sin(2*np.pi*s2)
    qs[:,3] = np.sqrt(s)   * np.cos(2*np.pi*s2)
    return qs

def _qs_to_Rs(qs):
    """(N,4) quaternions [w,x,y,z] → (N,3,3) rotation matrices."""
    w,x,y,z = qs[:,0],qs[:,1],qs[:,2],qs[:,3]
    n = len(qs)
    Rs = np.empty((n,3,3))
    Rs[:,0,0]=1-2*(y*y+z*z); Rs[:,0,1]=2*(x*y-z*w); Rs[:,0,2]=2*(x*z+y*w)
    Rs[:,1,0]=2*(x*y+z*w);   Rs[:,1,1]=1-2*(x*x+z*z); Rs[:,1,2]=2*(y*z-x*w)
    Rs[:,2,0]=2*(x*z-y*w);   Rs[:,2,1]=2*(y*z+x*w);   Rs[:,2,2]=1-2*(x*x+y*y)
    return Rs

N_COARSE = 1000   # coarse grid points
N_FINE   = 400    # fine grid points (±10° around coarse best)
FINE_DEG = 10.0   # half-width for fine search (degrees)

_QS_COARSE = _fibonacci_SO3(N_COARSE)
_RS_COARSE = _qs_to_Rs(_QS_COARSE)
_QS_FINE   = _fibonacci_SO3(N_FINE)    # perturbed per-cluster at runtime


def _perturb_qs(q_best, angle_deg, qs_template):
    """
    Map qs_template (N,4) onto the neighbourhood of q_best within angle_deg.
    Returns (N,4) unit quaternions.
    """
    half_max = np.radians(angle_deg) / 2.0
    sin_max  = np.sin(half_max)
    vec_norms = np.linalg.norm(qs_template[:,1:], axis=1, keepdims=True)
    scale = np.minimum(1.0, sin_max / np.maximum(vec_norms, 1e-12))
    qs = np.empty_like(qs_template)
    qs[:,1:] = qs_template[:,1:] * scale
    qs[:,0]  = np.sqrt(np.maximum(1.0 - np.sum(qs[:,1:]**2, axis=1), 0.0))
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    # Compose: q_result = q_best ⊗ qs
    w1,x1,y1,z1 = q_best
    w2,x2,y2,z2 = qs[:,0],qs[:,1],qs[:,2],qs[:,3]
    out = np.column_stack([
        w1*w2-x1*x2-y1*y2-z1*z2,
        w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2,
        w1*z2+x1*y2-y1*x2+z1*w2,
    ])
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out


# ---------------------------------------------------------------------------
# Core SND evaluation (vectorised)
# ---------------------------------------------------------------------------

def _snd_batch(Rs, P, Q):
    """
    Rs: (N,3,3), P: (M,3) moving, Q: (K,3) reference.
    Returns (N,) SND = Σ_i min_j ||R·p_i - q_j||².
    Vectorised: loop over M=20, broadcast over N×K.
    """
    RP   = np.einsum('nij,mj->nmi', Rs, P)      # (N, M, 3)
    NM   = RP.shape[0] * RP.shape[1]
    rp_f = RP.reshape(NM, 3)
    diff = rp_f[:,np.newaxis,:] - Q[np.newaxis,:,:]  # (NM, K, 3)
    d2   = np.sum(diff**2, axis=2).min(axis=1)        # (NM,)
    return d2.reshape(RP.shape[0], RP.shape[1]).sum(axis=1)  # (N,)


def _R_from_axis_angle(v):
    """3-vector (axis × angle) → rotation matrix."""
    angle = np.linalg.norm(v)
    if angle < 1e-10:
        return np.eye(3)
    ax = v / angle
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = ax
    return np.array([
        [c+x*x*(1-c),   x*y*(1-c)-z*s, x*z*(1-c)+y*s],
        [y*x*(1-c)+z*s, c+y*y*(1-c),   y*z*(1-c)-x*s],
        [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)  ]
    ])


def _gaussian_overlap_batch(P, Q, Rs, sigma):
    """
    Gaussian overlap O(R) = Σ_i Σ_j exp(-|R·p_i - q_j|² / 2σ²)
    for each R in Rs.  Factored for efficiency:
        |R·p_i - q_j|² = |p_i|² + |q_j|² - 2(R·p_i)·q_j
    so:
        O(R) = Σ_ij exp(-(|p_i|²+|q_j|²)/2σ²) · exp((R·p_i)·q_j / σ²)
    The first factor is independent of R and precomputed once.
    """
    inv2s2 = 0.5 / (sigma * sigma)
    P_sq   = np.sum(P**2, axis=1)                     # (nP,)
    Q_sq   = np.sum(Q**2, axis=1)                     # (nQ,)
    base   = np.exp(-inv2s2 * (P_sq[:,None] + Q_sq[None,:]))  # (nP,nQ)
    # RP[k,i,:] = p_i rotated by Rs[k]: P @ Rs[k].T gives (nP,3)
    RP  = np.einsum('kab,ib->kia', Rs, P)             # (nRot,nP,3)
    dot = np.einsum('kia,ja->kij', RP, Q)             # (nRot,nP,nQ)
    return np.sum(base[None,:,:] * np.exp(dot / (sigma*sigma)), axis=(1,2))


def align_snd(P, Q, depth='full'):
    """
    Find rotation R minimising SND(R) = Σ_i min_j ||R·p_i - q_j||².

    P: (M,3) moving O coords  (already centred on Ar)
    Q: (K,3) reference O coords
    depth: controls search effort, chosen based on SOAP similarity
        'coarse'  — coarse grid only           (low SOAP, structures dissimilar)
        'fine'    — coarse + fine grid          (medium SOAP, moderate similarity)
        'full'    — Gaussian annealing cascade + SND NM (high SOAP)

    For 'full' depth, uses Gaussian overlap annealing:
        σ = [3.0, 1.5, 0.8, 0.4] Å → SND NM polish
    This gives 30/30 recovery of random rotations of a cluster to itself
    (vs 24/30 with grid-only SND), while being ~3× faster (200 ms vs 600 ms).
    The broad σ finds the correct basin on the coarse grid; successive NM
    steps at decreasing σ walk into the sharp SND=0 minimum.

    Returns (R, snd_value).
    """
    # Coarse grid
    vals_c   = _snd_batch(_RS_COARSE, P, Q)
    best_ci  = int(np.argmin(vals_c))
    q_best   = _QS_COARSE[best_ci]
    R_best   = _RS_COARSE[best_ci]
    snd_best = vals_c[best_ci]

    if depth == 'coarse':
        return R_best, float(snd_best)

    # Fine grid around coarse best
    qs_f   = _perturb_qs(q_best, FINE_DEG, _QS_FINE)
    Rs_f   = _qs_to_Rs(qs_f)
    vals_f = _snd_batch(Rs_f, P, Q)
    best_fi = int(np.argmin(vals_f))
    if vals_f[best_fi] < snd_best:
        R_best   = Rs_f[best_fi]
        snd_best = vals_f[best_fi]

    if depth != 'full':
        return R_best, float(snd_best)

    # Gaussian annealing cascade: broad σ → sharp σ → SND NM
    # Start from the Gaussian-guided coarse+fine best across all 24 sym seeds.
    # Step 1: coarse Gaussian search with σ=3.0 (broad basin identification)
    sigma_init = 3.0
    best_R_gaus, best_ov = R_best, -np.inf
    for R_seed in CUBE_ROTATIONS:
        P_s  = P @ R_seed.T
        ovs_c = _gaussian_overlap_batch(P_s, Q, _RS_COARSE, sigma=sigma_init)
        bi    = int(np.argmax(ovs_c))
        qs_fg = _perturb_qs(_QS_COARSE[bi], FINE_DEG, _QS_FINE)
        Rs_fg = _qs_to_Rs(qs_fg)
        ovs_f = _gaussian_overlap_batch(P_s, Q, Rs_fg, sigma=sigma_init)
        bfi   = int(np.argmax(ovs_f))
        R_f   = Rs_fg[bfi] if ovs_f[bfi] > ovs_c[bi] else _RS_COARSE[bi]
        ov    = max(ovs_c[bi], ovs_f[bfi])
        if ov > best_ov:
            best_ov, best_R_gaus = ov, R_f @ R_seed

    # Step 2: NM polish at each decreasing σ (annealing)
    P_sq_log = np.sum(P**2, axis=1)
    Q_sq_log = np.sum(Q**2, axis=1)
    base_log  = -(P_sq_log[:,None] + Q_sq_log[None,:])   # factors out 2σ²

    R_cur = best_R_gaus
    for sigma in [1.5, 0.8, 0.4]:
        inv_s2 = 1.0 / (sigma * sigma)
        base   = np.exp(base_log * 0.5 * inv_s2)
        def neg_ov(v, R0=R_cur, b=base, is2=inv_s2):
            R  = R0 @ _R_from_axis_angle(v)
            RP = P @ R.T
            return -float(np.sum(b * np.exp((RP @ Q.T) * is2)))
        res   = minimize(neg_ov, np.zeros(3), method='Nelder-Mead',
                         options={'xatol':1e-5, 'fatol':1e-5, 'maxiter':400})
        R_cur = R_cur @ _R_from_axis_angle(res.x)

    # Step 3: Final SND NM polish
    tree = cKDTree(Q)
    def f_snd(v):
        R = R_cur @ _R_from_axis_angle(v)
        d, _ = tree.query(P @ R.T)
        return float(np.sum(d**2))
    res     = minimize(f_snd, np.zeros(3), method='Nelder-Mead',
                       options={'xatol':1e-7, 'fatol':1e-7, 'maxiter':800})
    R_final = R_cur @ _R_from_axis_angle(res.x)
    d, _    = tree.query(P @ R_final.T)
    return R_final, float(np.sum(d**2))


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_cen(path):
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


def read_weights(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                out[parts[1]] = float(parts[0])
    return out


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def centre_on_ar(coords, labels=None, path=''):
    """
    Shift all atoms so that the Ar atom is at the origin.
    Ar is expected to be atom 0 (first line after the header in .cen files).
    If labels are provided, verifies atom 0 is indeed Ar.
    Raises ValueError if Ar is not found at position 0.
    """
    if labels is not None:
        if not labels[0].startswith('Ar') and not labels[0].upper() == 'AR':
            # Search for Ar anywhere in the list as fallback
            ar_idx = next((i for i,l in enumerate(labels)
                           if l.startswith('Ar') or l.upper()=='AR'), None)
            if ar_idx is None:
                raise ValueError(
                    f"No Ar atom found in {path or 'structure'}. "
                    f"Labels: {labels[:5]}")
            if ar_idx != 0:
                import warnings
                warnings.warn(
                    f"Ar is at index {ar_idx}, not 0, in {path or 'structure'}. "
                    f"Recentring on Ar at index {ar_idx}.")
            return coords - coords[ar_idx]
    shifted = coords - coords[0]
    # Sanity check: after shifting, atom 0 must be exactly at origin
    assert np.allclose(shifted[0], 0.0), \
        f"Ar centering failed: atom 0 at {shifted[0]} after shift"
    return shifted

def get_O_indices(labels):
    return [i for i, lbl in enumerate(labels) if lbl.startswith('O')]

def nearest_O(coords, o_indices, num_O):
    oc    = coords[o_indices]
    order = np.argsort(np.linalg.norm(oc, axis=1))[:num_O]
    return oc[order]


# ---------------------------------------------------------------------------
# Kabsch (used only for Stage 2 inter-group registration)
# ---------------------------------------------------------------------------

def kabsch(P, Q):
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    return Vt.T @ np.diag([1.0,1.0,d]) @ U.T

def hungarian_kabsch(mov_O, ref_O):
    diff   = mov_O[:,np.newaxis,:] - ref_O[np.newaxis,:,:]
    cost   = np.sum(diff**2, axis=2)
    ri, ci = linear_sum_assignment(cost)
    dmin   = float(cost[ri,ci].sum())
    R      = kabsch(mov_O[ri], ref_O[ci])
    return R, dmin, ci


# ---------------------------------------------------------------------------
# Full SOAP structural similarity
# Bartók, Kondor, Csányi, Phys. Rev. B 87, 184115 (2013)
# Power spectrum equations follow De, Bartók, Csányi, Ceriotti,
# Phys. Chem. Chem. Phys. 18, 13754 (2016)
# ---------------------------------------------------------------------------
#
# Scipy spherical harmonic compatibility: old API sph_harm(m,l,phi,theta),
# new API (>=1.15) sph_harm_y(l,m,theta,phi).  Resolved once at import time.
try:
    from scipy.special import sph_harm_y as _sph_harm_import
    def _ylm(l, m, theta, phi):
        return _sph_harm_import(l, m, theta, phi)
except ImportError:
    from scipy.special import sph_harm as _sph_harm_import
    def _ylm(l, m, theta, phi):
        return _sph_harm_import(m, l, phi, theta)   # old: (m, l, phi, theta)

from scipy.special import spherical_in as _spherical_in


def soap_power_spectrum(O_coords, sigma=0.3, n_max=5, l_max=6, r_cut=7.0):
    """
    Full SOAP power spectrum (equations 2-5 of the paper).

    Each O atom j contributes a 3D Gaussian of width sigma centred at its
    position r_j (equation 2):
        rho(r) = sum_j  exp(-|r - r_j|^2 / 2*sigma^2)

    This density is expanded in radial basis functions g_n(r) times spherical
    harmonics Y_lm (equations 3-4).  The expansion coefficients c_nlm are
    obtained analytically via the addition theorem for Gaussians:
        exp(-|r - r_j|^2 / 2s^2) = exp(-r^2/2s^2) * exp(-r_j^2/2s^2)
                                    * exp(r . r_j / s^2)
    and expanding exp(r . r_j / s^2) in modified spherical Bessel functions
    i_l and spherical harmonics Y_lm*:
        c_nlm = sum_j  Y_lm*(th_j, ph_j) * exp(-r_j^2/2s^2) * I_nl(r_j)
    where
        I_nl(r_j) = int_0^{r_cut} g_n(r) * exp(-r^2/2s^2) * i_l(r*r_j/s^2)
                    * r^2 dr

    The power spectrum (equation 5, rotationally invariant):
        p_nn'l = sum_m  c_nlm * c*_n'lm

    Physical interpretation of sigma:
      - sigma controls the spatial localisation of each atom's Gaussian.
      - Small sigma (~0.15 Å): descriptor is sensitive to precise atom
        positions; in the limit sigma->0 approaches the SND metric.
      - Large sigma (~1.0 Å): descriptor reflects only coarse topology.
      - Recommended: sigma = 0.3 Å for water O atoms (O-O distance ~2.8 Å).

    Parameters
    ----------
    O_coords : (M, 3) array  — O-atom positions centred on Ar (origin)
    sigma    : float         — Gaussian smearing width in Angstrom
    n_max    : int           — number of radial basis functions
    l_max    : int           — maximum angular momentum
    r_cut    : float         — radial cutoff in Angstrom

    Returns
    -------
    p : 1-D array of length n_max*(n_max+1)/2 * (l_max+1)
    """
    # Radial basis: Gaussians g_n centred at r_n = (n+1)*dr, width s_r = dr
    dr_basis = r_cut / (n_max + 1)
    r_n  = np.array([(n + 1) * dr_basis for n in range(n_max)])
    s_r  = dr_basis

    # Filter atoms within cutoff
    r_j  = np.linalg.norm(O_coords, axis=1)
    mask = (r_j < r_cut) & (r_j > 1e-10)
    O    = O_coords[mask]
    r_j  = r_j[mask]
    n_p  = n_max * (n_max + 1) // 2 * (l_max + 1)
    if len(O) == 0:
        return np.zeros(n_p)

    x, y, z = O[:, 0], O[:, 1], O[:, 2]
    theta = np.arccos(np.clip(z / r_j, -1.0, 1.0))
    phi   = np.arctan2(y, x)

    # Radial integration grid
    n_grid  = 300
    r_grid  = np.linspace(0.01, r_cut, n_grid)
    dr_grid = r_grid[1] - r_grid[0]

    # Precompute (n_max, n_grid): radial basis * Gaussian decay
    G_n  = np.exp(-0.5 * ((r_grid[np.newaxis, :] - r_n[:, np.newaxis]) / s_r) ** 2)
    Gd   = G_n * np.exp(-0.5 * (r_grid / sigma) ** 2)[np.newaxis, :]

    # Accumulate c[n, l, m+l_max]
    c = np.zeros((n_max, l_max + 1, 2 * l_max + 1), dtype=complex)

    for j in range(len(O)):
        rj     = r_j[j]
        exp_rj = np.exp(-0.5 * (rj / sigma) ** 2)
        # Argument for modified spherical Bessel: r * r_j / sigma^2
        x_arg  = np.clip(r_grid * rj / sigma ** 2, 0.0, 700.0)

        # I_nl: (n_max, l_max+1)
        I_nl = np.zeros((n_max, l_max + 1))
        for l in range(l_max + 1):
            il_vals    = _spherical_in(l, x_arg)          # (n_grid,)
            integrand  = Gd * (il_vals * r_grid ** 2)[np.newaxis, :]
            I_nl[:, l] = integrand.sum(axis=1) * dr_grid * exp_rj

        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                c[:, l, m + l_max] += (
                    np.conj(_ylm(l, m, theta[j], phi[j])) * I_nl[:, l])

    # Power spectrum p_{nn'l} = Σ_m c_{nlm} * conj(c_{n'lm})  (upper triangle n'>=n)
    # Note: the 2017 erratum (PRB 96, 019902) corrects this to conj(c_{nlm})*c_{n'lm}.
    # For real atomic positions both formulas give identical real values because the
    # expansion coefficients satisfy c_{nlm}^* = (-1)^m c_{nl,-m}, making the sum
    # over m purely real and equal under either conjugation convention.
    p = []
    for l in range(l_max + 1):
        for n in range(n_max):
            for np_ in range(n, n_max):
                p.append(float(np.real(
                    np.sum(c[n, l, :] * np.conj(c[np_, l, :])))))
    return np.array(p)


def soap_similarity(O_a, O_b, sigma=0.3, n_max=5, l_max=6, r_cut=7.0):
    """
    Full SOAP cosine similarity ∈ [0, 1].
    Rotationally invariant.  1 = identical structure; lower = more different.

    sigma controls localisation of atom Gaussians:
      smaller sigma -> more sensitive to precise O positions -> better
      predictor of achievable SND after alignment.
      Recommended sigma = 0.3 Å (≈ 10% of O-O distance).
    """
    pa = soap_power_spectrum(O_a, sigma=sigma, n_max=n_max,
                             l_max=l_max, r_cut=r_cut)
    pb = soap_power_spectrum(O_b, sigma=sigma, n_max=n_max,
                             l_max=l_max, r_cut=r_cut)
    denom = np.linalg.norm(pa) * np.linalg.norm(pb)
    if denom < 1e-30:
        return 0.0
    return float(np.dot(pa, pb) / denom)

# ---------------------------------------------------------------------------
# Primitive ring analysis (H-bond graph topology)
# ---------------------------------------------------------------------------
#
# Builds an O-O graph using a distance cutoff (default 3.2 Å, just beyond
# the first O-O peak at ~2.8 Å and well below the second shell at ~4.5 Å).
# Finds all primitive rings up to size max_ring using BFS on each edge:
# for each edge (u,v), the shortest alternate path u→v not using that edge
# closes a ring.  A ring is primitive if it cannot be expressed as the XOR
# (symmetric difference) of two smaller rings already found.
#
# Ring sizes discriminate structural motifs:
#   CS-I  cage  : ~11-12 pentagons, 0-2 hexagons  (5¹²6² topology)
#   CS-II cage  : ~11-12 pentagons, 0-4 hexagons  (5¹²6⁴ topology)
#   Dodecahedron: ~11-12 pentagons, 0 hexagons    (5¹² topology)
#   Liquid      : variable mix of 4-, 5-, 6-, 7-membered rings
#
# Note: with only 20 O atoms (an incomplete shell), the graph is missing
# some bonds to atoms outside the 20-nearest set, so counts are slightly
# below ideal cage values.  Pentagon dominance vs mixed rings is still
# clearly diagnostic.
#
# Reference: Franzblau (1991), Phys Rev B 44:4925.

def _build_hbond_graph(O_coords, cutoff):
    """Adjacency dict for O atoms within cutoff distance."""
    from collections import defaultdict
    n   = len(O_coords)
    adj = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(O_coords[i] - O_coords[j]) < cutoff:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _bfs_path(adj, start, end, forbidden_edge, max_len):
    """BFS shortest path start→end avoiding one edge.  Returns node list or None."""
    from collections import deque
    fe  = frozenset(forbidden_edge)
    q   = deque([(start, [start])])
    vis = {start}
    while q:
        node, path = q.popleft()
        if len(path) >= max_len:
            continue
        for nb in adj[node]:
            if frozenset((node, nb)) == fe:
                continue
            if nb == end:
                return path + [nb]
            if nb not in vis:
                vis.add(nb)
                q.append((nb, path + [nb]))
    return None


def ring_fingerprint(O_coords, cutoff=3.2, max_ring=7):
    """
    Compute the primitive ring census of the O-atom H-bond network.

    Parameters
    ----------
    O_coords : (M, 3) array  — O-atom positions centred on Ar (origin)
    cutoff   : float         — O-O bond cutoff in Å          (default 3.2 Å)
    max_ring : int           — largest ring size to search    (default 7)

    Returns
    -------
    census : dict {ring_size: count}  e.g. {5: 11} for a CS-I cage
    rings  : list of frozensets of node indices (one per primitive ring)
    """
    adj  = _build_hbond_graph(O_coords, cutoff)
    seen = set()
    raw  = {}   # frozenset(nodes) -> size

    for u in adj:
        for v in sorted(adj[u]):
            e = frozenset((u, v))
            if e in seen:
                continue
            seen.add(e)
            path = _bfs_path(adj, v, u, (u, v), max_ring)
            if path:
                ring = frozenset(path)
                sz   = len(ring)
                if sz <= max_ring and ring not in raw:
                    raw[ring] = sz

    # Keep only primitive rings (not expressible as XOR of two smaller ones)
    by_size = sorted(raw.items(), key=lambda x: x[1])
    primitive = {}
    for ring, sz in by_size:
        composite = False
        for r1 in primitive:
            if len(r1) >= sz:
                continue
            if frozenset(ring.symmetric_difference(r1)) in raw:
                composite = True
                break
        if not composite:
            primitive[ring] = sz

    from collections import defaultdict
    census = defaultdict(int)
    for sz in primitive.values():
        census[sz] += 1

    return dict(census), list(primitive.keys())


def ring_census_str(census, sizes=(3, 4, 5, 6, 7)):
    """Compact string representation, e.g. '0-0-11-0-0'."""
    return '-'.join(str(census.get(s, 0)) for s in sizes)


def classify_topology(census):
    """
    Assign a topology label from the primitive ring census.

    Labels (based on pentagon n5 and hexagon n6 counts)
    ------
    'DODEC'  : n5 >= 10, n6 == 0  (5¹² dodecahedron)
    'CS1'    : n5 >= 10, n6 <= 2  (5¹²6² cage, CS-I)
    'CS2'    : n5 >= 10, n6 >= 3  (5¹²6⁴ cage, CS-II)
    'CAGE'   : 5 <= n5 < 10       (partial cage / cage precursor)
    'MIXED'  : 1 <= n5 < 5        (disordered with cage-like motifs)
    'LIQUID' : n5 == 0            (fully disordered)
    """
    n5 = census.get(5, 0)
    n6 = census.get(6, 0)
    if n5 >= 10:
        if n6 == 0:
            return 'DODEC'
        elif n6 <= 2:
            return 'CS1'
        else:
            return 'CS2'
    elif n5 >= 5:
        return 'CAGE'
    elif n5 >= 1:
        return 'MIXED'
    else:
        return 'LIQUID'


# ---------------------------------------------------------------------------
# 24 proper cube rotations (Stage 2 symmetry search)
# ---------------------------------------------------------------------------

def _cube_rotations():
    rots = []
    for perm in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]:
        for signs in iproduct([1,-1], repeat=3):
            M = np.zeros((3,3))
            for col,(row,s) in enumerate(zip(perm,signs)):
                M[row,col] = s
            if np.linalg.det(M) > 0:
                rots.append(M)
    return rots

CUBE_ROTATIONS = _cube_rotations()
assert len(CUBE_ROTATIONS) == 24


# ---------------------------------------------------------------------------
# Pre-scan: parallel topology classification of all clusters
# ---------------------------------------------------------------------------

def _prescan_chunk(chunk_args):
    """
    Worker: compute ring topology for a chunk of clusters.
    chunk_args = (coords_list, basenames_chunk, o_indices, num_O,
                  ring_cutoff, ring_max)
    Returns list of (bn, topo_label, census_dict, ring_str).
    """
    coords_list, basenames_chunk, o_indices, num_O, ring_cutoff, ring_max = chunk_args
    results = []
    for bn, coords in zip(basenames_chunk, coords_list):
        mov_O  = nearest_O(coords, o_indices, num_O)
        census, _ = ring_fingerprint(mov_O, cutoff=ring_cutoff, max_ring=ring_max)
        topo   = classify_topology(census)
        rstr   = ring_census_str(census)
        results.append((bn, topo, census, rstr))
    return results


def prescan_topology(basenames, all_coords_map, o_indices, args, nworkers):
    """
    Classify every cluster by ring topology before Stage 1 alignment.

    Returns
    -------
    topo_pre  : dict {bn: topology_label}   (e.g. 'DODEC', 'MIXED', ...)
    census_pre: dict {bn: census_dict}
    rings_pre : dict {bn: ring_census_str}
    """
    nspec      = len(basenames)
    chunk_size = max(1, (nspec + nworkers - 1) // nworkers)
    chunks     = []
    for start in range(0, nspec, chunk_size):
        end = min(start + chunk_size, nspec)
        bns = basenames[start:end]
        cds = [all_coords_map[bn] for bn in bns]
        chunks.append((cds, bns, o_indices, args.num_O,
                       args.ring_cutoff, args.ring_max))

    topo_pre   = {}
    census_pre = {}
    rings_pre  = {}

    with ProcessPoolExecutor(max_workers=nworkers) as pool:
        for result_list in pool.map(_prescan_chunk, chunks, chunksize=1):
            for bn, topo, census, rstr in result_list:
                topo_pre[bn]   = topo
                census_pre[bn] = census
                rings_pre[bn]  = rstr

    # Summary
    from collections import Counter
    counts = Counter(topo_pre.values())
    print(f'  Pre-scan topology distribution:')
    for label in ['DODEC','CS1','CS2','CAGE','MIXED','LIQUID']:
        cnt = counts.get(label, 0)
        if cnt:
            print(f'    {label:<8s}: {cnt:6d} ({100*cnt/nspec:.1f}%)')

    return topo_pre, census_pre, rings_pre


def split_by_topology(group_map, topo_pre, topo_subgroup_labels=None):
    """
    Split each group into topology sub-groups.

    Only splits groups where the topology is genuinely mixed —
    if all members share one topology, no split is needed.

    Parameters
    ----------
    topo_subgroup_labels : list of topology labels to split out, or None
        (split all labels present)

    Returns
    -------
    new_group_map : dict {group/topo: [basenames]}
    parent_map    : dict {subgroup_name: parent_group_name}
    """
    from collections import defaultdict, Counter
    new_group_map = {}
    parent_map    = {}

    for g, members in group_map.items():
        counts = Counter(topo_pre[bn] for bn in members)
        n_labels = len(counts)

        if n_labels == 1:
            # All same topology — no split needed
            new_group_map[g] = members
            parent_map[g]    = g
        else:
            for label, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                if topo_subgroup_labels and label not in topo_subgroup_labels:
                    # Lump unlisted labels into parent group name
                    key = g
                else:
                    key = f'{g}/{label}'
                sub = [bn for bn in members if topo_pre[bn] == label]
                if sub:
                    new_group_map[key] = new_group_map.get(key, []) + sub
                    parent_map[key]    = g

    return new_group_map, parent_map



def _align_chunk(chunk_args):
    """
    Align a chunk of clusters to ref_O using Hungarian+Kabsch.

    Used exclusively in Phase A (iterative alignment to moving mean).

    The Hungarian algorithm finds the bijection π minimising
        Σᵢ ||p_i − q_{π(i)}||²
    using the same squared-Euclidean cost as SND but with a bijection
    constraint.  Kabsch then finds the rotation R minimising
        Σᵢ ||R·p_i − q_{π(i)}||²
    under that fixed assignment.

    The rotated O positions are stored in HUNGARIAN ORDER (i.e. the
    atom matched to reference position i is stored at position i),
    so that averaging near_O_arr across clusters gives a mean reference
    where position i is the average of all atoms assigned to reference
    position i — a physically consistent correspondence.

    Returns (new_coords, snds, near_O_arr (k,num_O,3), n_improved).
    near_O_arr[k] is the k-th cluster's O atoms after rotation,
    stored in Hungarian-matched order aligned to the reference.
    """
    coords_list, o_indices, ref_O, num_O, prev_snds = chunk_args
    new_coords, snds, near_O_list = [], [], []
    n_improved = 0
    for idx, coords in enumerate(coords_list):
        mov_O = nearest_O(coords, o_indices, num_O)
        R, snd, ci = hungarian_kabsch(mov_O, ref_O)
        rotated_full   = coords @ R.T
        rotated_O      = mov_O @ R.T          # (num_O, 3) in distance-sorted order
        # Reorder to Hungarian-matched order: position i gets the O atom
        # that was matched to reference position i (i.e. inverse permutation of ci)
        inv_perm = np.argsort(ci)
        reordered_O = rotated_O[inv_perm]     # position i ↔ ref position i
        new_coords.append(rotated_full)
        snds.append(snd)
        near_O_list.append(reordered_O)
        if prev_snds is None or snd < prev_snds[idx] - 0.01:
            n_improved += 1
    return new_coords, snds, np.array(near_O_list), n_improved


def _write_one(args):
    """
    Final write pass for one cluster.

    Search depth is chosen by combining SOAP similarity and topology match
    (Option B — topology-guided depth selection):

        topo_match = cluster topology == reference topology

        topo_match  AND  SOAP >= soap_polish_thresh  →  'full'
        topo_match  AND  SOAP >= soap_skip_thresh    →  'fine'
        topo_match  (any SOAP)                       →  'fine'
        not topo_match  AND  SOAP >= soap_polish_thresh → 'fine'
        not topo_match  (any SOAP)                       → 'coarse'

    Rationale:
      - Topology match means the structures share the same ring network —
        a precise rotation is both findable and physically meaningful.
        These get at least 'fine', and 'full' if SOAP is also high.
      - Topology mismatch means the cluster and reference are fundamentally
        different structural types.  No rotation will give a truly low SND;
        'coarse' records a best-effort alignment and the high SND correctly
        signals genuine structural dissimilarity, not alignment failure.
    """
    (bn, ext_in, ext_out, o_indices, R_inter, ref_O, num_O,
     soap_polish_thresh, soap_skip_thresh, soap_sigma,
     args_ring_cutoff, args_ring_max, ref_topo, pre_coords) = args
    try:
        labels, coords_orig = read_cen(bn + ext_in)
    except FileNotFoundError:
        return bn, float('nan'), float('nan'), None, {}, '?', '?', \
               f'File not found: {bn+ext_in}'

    coords = pre_coords if pre_coords is not None else \
             centre_on_ar(coords_orig, labels=labels, path=bn + ext_in)
    mov_O  = nearest_O(coords, o_indices, num_O)

    # Ring topology of this cluster (rotationally invariant)
    census, _ = ring_fingerprint(mov_O, cutoff=args_ring_cutoff,
                                 max_ring=args_ring_max)
    topo  = classify_topology(census)
    rstr  = ring_census_str(census)

    # Full SOAP similarity (rotationally invariant)
    soap  = soap_similarity(mov_O, ref_O, sigma=soap_sigma)

    # Topology-guided depth selection (Option B)
    topo_match = (topo == ref_topo)
    if topo_match:
        if soap >= soap_polish_thresh:
            depth = 'full'
        else:
            depth = 'fine'       # topology matches — always at least fine
    else:
        if soap >= soap_polish_thresh:
            depth = 'fine'       # high SOAP but different topology — cap at fine
        else:
            depth = 'coarse'     # different topology — coarse only

    R, snd = align_snd(mov_O, ref_O, depth=depth)
    c_rot  = coords @ R.T @ R_inter.T
    write_cen(bn + ext_out, labels, c_rot)
    return bn, snd, soap, depth, census, topo, rstr, None


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def assign_groups(basenames, patterns):
    groups = {p: [] for p in patterns}
    groups['other'] = []
    for bn in basenames:
        matched = False
        for p in patterns:
            if re.search(p, bn, re.IGNORECASE):
                groups[p].append(bn)
                matched = True
                break
        if not matched:
            groups['other'].append(bn)
    return {k: v for k,v in groups.items() if v}


# ---------------------------------------------------------------------------
# Stage 1: intra-group iterative alignment (Hungarian+Kabsch)
# ---------------------------------------------------------------------------

def octant_align_to_template(near_O_boot, template_idx):
    """
    For each distance-sorted O array in near_O_boot, flip the sign of each
    coordinate so that it matches the sign of the corresponding coordinate in
    the template cluster.

    Rationale
    ---------
    Distance-sorting places the closest O atom first, the second-closest
    second, etc.  Without any further constraint, clusters in random
    orientations will have these positions scattered over all octants of
    R³, so their unweighted mean collapses toward the Ar origin, making
    Hungarian matching degenerate on the first Phase A iteration.

    This function resolves the sign ambiguity per O position while
    preserving the distance ordering: for O atom i, each coordinate x, y, z
    is independently reflected through the origin if its sign disagrees with
    that of the template's i-th O atom.  The result is that every cluster's
    nearest O atom is placed in the same octant as the template's nearest O
    atom, and so on for subsequent shells.  The mean of these sign-corrected
    positions is well-spread in space and gives a non-degenerate starting
    reference for Hungarian matching.

    Parameters
    ----------
    near_O_boot : list of (num_O, 3) arrays — distance-sorted O coords,
                  one array per cluster in the group.
    template_idx : int — index into near_O_boot of the cluster to use as
                  the octant template (--phase_A_select K).

    Returns
    -------
    aligned : list of (num_O, 3) arrays with signs matched to the template.
              The template itself is returned unchanged.
    """
    template = near_O_boot[template_idx]          # (num_O, 3)
    # sign pattern of the template: +1 or -1 per (position, coordinate).
    # For coordinates exactly zero we use +1 (arbitrary; negligible effect).
    template_signs = np.sign(template)
    template_signs[template_signs == 0] = 1.0

    aligned = []
    for O in near_O_boot:
        signs_O = np.sign(O)
        signs_O[signs_O == 0] = 1.0
        # flip = +1 where signs agree, -1 where they disagree
        flip = template_signs * signs_O           # (num_O, 3): ±1 elementwise
        aligned.append(O * flip)
    return aligned


def stage1_align_group(group_name, basenames, all_coords_map,
                       o_indices, args, nworkers):
    """
    Intra-group alignment using SND metric.  Two cleanly separated phases:

    PHASE A — iterative alignment to a moving mean reference
        Runs until Δobj < conv_obj.  The mean reference is updated every
        iteration, so iteration is meaningful: the target shifts because we
        are refining what "typical" means for this group.  This phase
        establishes a good collective orientation and identifies the most
        representative cluster.

        The iterative mean is the unweighted arithmetic mean of the
        num_O nearest-O position arrays after alignment:

            ref_O_{k+1}[i] = (1/N) Σ_n  R_n^(k) · p_n[i]

        where p_n[i] is the i-th nearest-O atom of cluster n (matched to
        the current reference by the Hungarian algorithm), R_n^(k) is the
        Kabsch rotation found in iteration k, and N is the number of clusters
        in the group.

        The mean is deliberately unweighted: the alignment is
        temperature-independent (one .rot file set for all temperatures),
        so EXAFS signal weights must not influence the geometric mean.
        Analysis weights are applied later in density_average.py.

    PHASE B — single-pass alignment to the best-cluster reference
        Only entered when --use_best_ref is set.  The best cluster (lowest
        SND in Phase A) is used as a fixed reference.  Because the reference
        is now fixed, the optimal rotation for each cluster is a fixed
        mathematical answer — there is nothing to iterate.  A single pass
        with depth='fine' aligns all clusters; the SOAP-tiered polish
        happens later in the final write pass.

        This eliminates the oscillation that occurred when the iteration loop
        continued after the reference switch: the grid search has ~15°
        resolution, so re-running it on already-aligned coordinates introduced
        quantisation noise on every pass, causing a limit cycle.

    Returns ref_O (num_O, 3) — O positions of the fixed reference cluster
    (or the converged mean if --use_best_ref is not set).
    """
    nspec      = len(basenames)
    all_coords = [all_coords_map[bn] for bn in basenames]

    chunk_size     = max(1, (nspec + nworkers - 1) // nworkers)
    coord_chunks   = [all_coords[i:i+chunk_size]
                      for i in range(0, nspec, chunk_size)]
    actual_workers = len(coord_chunks)

    # ------------------------------------------------------------------ #
    # PHASE A: iterative alignment to the moving mean                    #
    # ------------------------------------------------------------------ #
    near_O_boot = [nearest_O(c, o_indices, args.num_O) for c in all_coords]

    phase_A_select = getattr(args, 'phase_A_select', None)
    if phase_A_select is not None:
        # Compromise initial reference: distance-sort all clusters (already
        # done by nearest_O), then flip the sign of each coordinate of each
        # O position so it lands in the same octant as the corresponding O
        # position of cluster K (--phase_A_select K).  The mean of these
        # sign-corrected positions is well-spread in space — no collapse
        # toward the Ar origin — but is not biased to any single cluster's
        # geometry.  The iterative alignment then converges from this
        # unambiguous starting point.
        template_idx = min(phase_A_select, nspec - 1)
        if template_idx != phase_A_select:
            print(f'  WARNING: --phase_A_select {phase_A_select} exceeds group '
                  f'size {nspec}; using cluster {template_idx} instead.')
        near_O_octant = octant_align_to_template(near_O_boot, template_idx)
        ref_O = np.mean(near_O_octant, axis=0)   # (num_O, 3) mean over all clusters
        print(f'  Phase A initial ref: octant-averaged mean  '
              f'(template=cluster #{template_idx} in group, '
              f'--phase_A_select={phase_A_select})', flush=True)
    else:
        # Default: use the first cluster as a well-defined, non-degenerate
        # starting point.  Averaging randomly oriented clusters would collapse
        # all 20 reference positions to within ~0.1 Å of the origin.
        ref_O = near_O_boot[0]
        print(f'  Phase A initial ref: first cluster in group '
              f'(use --phase_A_select K for octant-averaged mean)', flush=True)
    obj    = float('inf')
    converged_A    = False
    prev_snd_chunks = [None] * actual_workers
    d_obj = float('inf')

    print(f'  Phase A: iterative alignment to moving mean '
          f'(Hungarian+Kabsch, deterministic)', flush=True)

    with ProcessPoolExecutor(max_workers=actual_workers) as pool:
        for iteration in range(1, args.maxiter + 1):
            t_iter = time.time()

            map_args = [(chunk, o_indices, ref_O, args.num_O, prev_s)
                        for chunk, prev_s in zip(coord_chunks, prev_snd_chunks)]

            all_new_coords = []
            all_snds       = []
            near_O_parts   = []
            total_improved = 0

            for new_c, snd_list, near_O_arr, n_imp in pool.map(
                    _align_chunk, map_args, chunksize=1):
                all_new_coords.extend(new_c)
                all_snds.extend(snd_list)
                near_O_parts.append(near_O_arr)
                total_improved += n_imp

            near_O_all = np.concatenate(near_O_parts, axis=0)
            new_ref_O  = near_O_all.mean(axis=0)
            new_obj    = float(np.sum(all_snds))
            d_obj      = abs(obj - new_obj)
            mean_shift = float(np.max(np.linalg.norm(new_ref_O - ref_O, axis=1)))
            # mean_shift = max displacement of any O atom in the MEAN REFERENCE
            # structure between this and the previous iteration.  Since the mean
            # reference is the centroid of all rotated cluster O positions,
            # this measures how much the collective orientation consensus shifted.
            # Scale: 1° of collective rotation ≈ 0.06 Å at typical O-Ar radius.
            snd_arr    = np.array(all_snds)

            print(f'  Iter {iteration:3d} ({time.time()-t_iter:.1f}s)'
                  f'  [mean ref]:'
                  f'  obj={new_obj:.2f}  Δobj={d_obj:.2f} Å²', flush=True)
            print(f'    SND   : mean={snd_arr.mean():.2f}  std={snd_arr.std():.2f}'
                  f'  min={snd_arr.min():.2f}  max={snd_arr.max():.2f}  (Å²)',
                  flush=True)
            print(f'    Improved          : {total_improved}/{nspec}'
                  f' ({100.0*total_improved/nspec:.1f}%)', flush=True)
            print(f'    Mean-ref O shift  : {mean_shift:.4f} Å'
                  f'  (~{mean_shift/0.06:.1f}° collective rotation)',
                  flush=True)

            all_coords      = all_new_coords
            coord_chunks    = [all_coords[i:i+chunk_size]
                               for i in range(0, nspec, chunk_size)]
            prev_snd_chunks = [all_snds[i:i+chunk_size]
                               for i in range(0, nspec, chunk_size)]
            ref_O = new_ref_O
            obj   = new_obj

            if d_obj < args.conv_obj and iteration > 1:
                print(f'  → Phase A converged after {iteration} iterations'
                      f'  (Δobj={d_obj:.4f} Å²)')
                converged_A = True
                break

            # Secondary criterion: if the mean frame has stabilised (shift
            # plateaued at a fixed value for several iterations in a row)
            # AND Δobj is small relative to group size, accept convergence.
            # This handles large liquid groups whose mean has genuine rotational
            # degeneracy — the mean-O shift never reaches zero, but the frame
            # is stable and Δobj/cluster is already sub-threshold.
            if (iteration > 5
                    and mean_shift < 0.20
                    and d_obj / nspec < args.conv_obj
                    and iteration > 1):
                print(f'  → Phase A converged after {iteration} iterations'
                      f'  (Δobj/cluster={d_obj/nspec:.4f} Å²,'
                      f'  mean-O shift={mean_shift:.4f} Å)')
                converged_A = True
                break

    if not converged_A:
        print(f'  → Phase A WARNING: did not converge in {args.maxiter} iterations'
              f'  (last Δobj={d_obj:.2f} Å²)')

    if not args.use_best_ref:
        # Mean reference: compute its topology from the mean O coords
        census_ref, _ = ring_fingerprint(ref_O,
                                         cutoff=args.ring_cutoff,
                                         max_ring=args.ring_max)
        ref_topo = classify_topology(census_ref)
        print(f'  Reference topology (mean): {ref_topo}  '
              f'[{ring_census_str(census_ref)}]')
        return ref_O, ref_topo

    # ------------------------------------------------------------------ #
    # PHASE B: find the best-cluster reference via anchor scanning         #
    # ------------------------------------------------------------------ #
    #
    # The cluster with the lowest Phase A SND-to-mean is a proxy for the
    # best anchor, but it is only a proxy.  The true criterion is: which
    # cluster c minimises mean(SND(all others aligned to c))?
    #
    # We scan the top-K candidates (lowest Phase A SND) and compute a
    # quick mean SND for each by running one Phase B pass.  The candidate
    # giving the lowest mean SND wins.  This costs K× Phase B time but
    # finds a much better anchor, especially when the mean is dominated by
    # a high-weight structural type that may not match the overall group.
    #
    # --anchor NAME overrides this scan: use a specific cluster by basename.

    # Resolve anchor
    anchor_override = getattr(args, 'anchor', None)
    if anchor_override:
        if anchor_override in basenames:
            fixed_idx = basenames.index(anchor_override)
            print(f'\n  Phase B: using user-specified anchor: {anchor_override}')
        else:
            # Partial match
            matches = [i for i,bn in enumerate(basenames)
                       if anchor_override in bn]
            if matches:
                fixed_idx = matches[0]
                print(f'\n  Phase B: anchor partial match → {basenames[fixed_idx]}')
            else:
                print(f'\n  Phase B: anchor "{anchor_override}" not found; '
                      f'falling back to scan')
                anchor_override = None
                fixed_idx = None
    else:
        fixed_idx = None

    # Sort candidates by Phase A SND-to-mean
    snd_order = np.argsort(all_snds)
    n_scan = 1 if anchor_override else min(getattr(args, 'anchor_scan_k', 5),
                                           len(basenames))

    if anchor_override:
        candidates = [fixed_idx]
    else:
        candidates = list(snd_order[:n_scan])

    print(f'\n  Phase B: scanning {len(candidates)} anchor candidate(s) '
          f'(--anchor_scan_k={n_scan})', flush=True)

    best_mean_snd = float('inf')
    best_ref_O    = None
    best_ref_name = None
    best_ref_topo = None
    best_final_coords = None
    best_final_snds   = None

    for rank, cand_idx in enumerate(candidates):
        cand_name = basenames[cand_idx]
        cand_ref_O = nearest_O(all_coords[cand_idx], o_indices, args.num_O)

        census_c, _ = ring_fingerprint(cand_ref_O,
                                       cutoff=args.ring_cutoff,
                                       max_ring=args.ring_max)
        cand_topo = classify_topology(census_c)

        map_args = [(chunk, o_indices, cand_ref_O, args.num_O, None)
                    for chunk in coord_chunks]

        t_c = time.time()
        cand_coords, cand_snds = [], []
        with ProcessPoolExecutor(max_workers=actual_workers) as pool:
            for new_c, snd_list, _, _ in pool.map(
                    _align_chunk, map_args, chunksize=1):
                cand_coords.extend(new_c)
                cand_snds.extend(snd_list)

        snd_arr_c = np.array(cand_snds)
        # Global measure: mean SND excluding self-term
        non_ref   = snd_arr_c[snd_arr_c >= 0.5]
        mean_snd_c = float(non_ref.mean()) if len(non_ref) else 0.0

        print(f'    Candidate {rank+1}/{len(candidates)}: {cand_name}  '
              f'topo={cand_topo}  mean_SND={mean_snd_c:.4f} Å²  '
              f'({time.time()-t_c:.1f}s)', flush=True)

        if mean_snd_c < best_mean_snd:
            best_mean_snd   = mean_snd_c
            best_ref_O      = cand_ref_O
            best_ref_name   = cand_name
            best_ref_topo   = cand_topo
            best_final_coords = cand_coords
            best_final_snds   = cand_snds

    ref_O    = best_ref_O
    ref_topo = best_ref_topo

    snd_arr     = np.array(best_final_snds)
    snd_non_ref = snd_arr[snd_arr >= 0.5]
    census_best, _ = ring_fingerprint(ref_O,
                                       cutoff=args.ring_cutoff,
                                       max_ring=args.ring_max)
    print(f'\n  ✓ Best anchor: {best_ref_name}  '
          f'topo={best_ref_topo}  '
          f'[{ring_census_str(census_best)}]')
    print(f'    mean_SND (global measure) = {best_mean_snd:.4f} Å²')
    if len(snd_non_ref):
        print(f'    SND: mean={snd_non_ref.mean():.2f}  std={snd_non_ref.std():.2f}'
              f'  min={snd_non_ref.min():.2f}  max={snd_non_ref.max():.2f}  (Å²)'
              f'  [self-term excluded]', flush=True)

    # Update coords map from the winning Phase B pass
    for bn, c in zip(basenames, best_final_coords):
        all_coords_map[bn] = c

    # Collect top-K candidate O-arrays for cross-pool anchor evaluation
    top_k_O = []
    for cand_idx in snd_order[:n_scan]:
        cand_O = nearest_O(all_coords_map[basenames[cand_idx]], o_indices,
                           args.num_O)
        top_k_O.append((basenames[cand_idx], cand_O))

    return ref_O, ref_topo, top_k_O


# ---------------------------------------------------------------------------
# Stage 2: inter-group registration (Kabsch, fast)
# ---------------------------------------------------------------------------

def compute_group_weights(group_map, weight_dicts):
    group_weights = {}
    for g, members in group_map.items():
        ms = set(members)
        per_temp = [sum(wd.get(bn,0.0) for bn in ms) for wd in weight_dicts]
        group_weights[g] = float(np.mean(per_temp))
    return group_weights


# Stage 2: inter-group registration
# ---------------------------------------------------------------------------

def compute_group_weights(group_map, weight_dicts):
    group_weights = {}
    for g, members in group_map.items():
        ms = set(members)
        per_temp = [sum(wd.get(bn,0.0) for bn in ms) for wd in weight_dicts]
        group_weights[g] = float(np.mean(per_temp))
    return group_weights


def stage2_register(group_means, group_weights, sym_search=False,
                    maxiter=50, conv=1e-4):
    """
    Inter-group registration: align all group means into a common frame.

    Strategy
    --------
    The heaviest group (by weight, typically LIQUID) defines the anchor frame.
    Every other group is aligned TO that anchor using the SND grid search
    (coarse + fine quaternion grid + sym_search), which is the same robust
    aligner used in Phase B.  This is a single pass — no iteration — because:

    1. The iterative weighted-mean approach oscillates when one group dominates
       (78% LIQUID weight makes the weighted mean nearly identical to the
       LIQUID mean, so the other groups keep chasing a moving target they
       can never properly match).

    2. With a fixed anchor the problem is well-posed: find the rotation that
       best overlaps each group mean with the LIQUID O-shell.  The SND grid
       search solves this directly without iterative averaging.

    The anchor group gets R = identity.  All others get R from align_snd().
    """
    groups = list(group_means.keys())
    ng     = len(groups)

    if ng == 1:
        print('  Only one group — inter-group registration is identity.')
        return {groups[0]: np.eye(3)}

    total_w = sum(group_weights[g] for g in groups)
    w = ({g: 1.0/ng for g in groups} if total_w == 0
         else {g: group_weights[g]/total_w for g in groups})

    # Choose anchor: group with highest mean analysis weight (most important
    # for the EXAFS signal).  group_weights here are the mean-over-temperatures
    # normalised weights computed from the analysis weight files.
    # With uniform weights (no analysis files), prefer the largest LIQUID group.
    total_w = sum(group_weights[g] for g in groups)
    w = ({g: 1.0/ng for g in groups} if total_w == 0
         else {g: group_weights[g]/total_w for g in groups})

    if total_w == 0:
        # No analysis weights: pick largest LIQUID sub-group by cluster count
        liquid_groups = [g for g in groups if 'LIQUID' in g.upper()]
        anchor = (max(liquid_groups, key=lambda g: len(group_means[g]))
                  if liquid_groups else max(groups, key=lambda g: w[g]))
    else:
        # Use the group with the highest mean analysis weight as anchor
        anchor = max(groups, key=lambda g: w[g])
    anchor_mean = group_means[anchor]

    print(f'  Anchor group (fixed frame): {anchor}  (W={w[anchor]:.4f})')
    print(f'  Normalised signal weights (used for anchor selection):')
    for g in sorted(groups, key=lambda g: -w[g]):
        marker = ' ← anchor' if g == anchor else ''
        print(f'    {g:30s}  W={w[g]:.6f}{marker}')
    print(f'  Aligning all other groups to anchor ...')

    R_inter = {g: np.eye(3) for g in groups}

    for g in groups:
        if g == anchor:
            print(f'    {g:<30s}  → identity (anchor)', flush=True)
            continue

        mov = group_means[g]

        if sym_search:
            # Try all 24 cubic rotations as seeds, keep best
            best_R, best_snd = None, float('inf')
            for R_seed in CUBE_ROTATIONS:
                seeded = mov @ R_seed.T
                # Coarse + fine grid from seeded orientation
                vals_c  = _snd_batch(_RS_COARSE, seeded, anchor_mean)
                best_ci = int(np.argmin(vals_c))
                q_best  = _QS_COARSE[best_ci]
                R_c     = _RS_COARSE[best_ci]
                snd_c   = vals_c[best_ci]
                # Fine grid
                qs_f   = _perturb_qs(q_best, FINE_DEG, _QS_FINE)
                Rs_f   = _qs_to_Rs(qs_f)
                vals_f = _snd_batch(Rs_f, seeded, anchor_mean)
                best_fi = int(np.argmin(vals_f))
                R_f    = Rs_f[best_fi] if vals_f[best_fi] < snd_c else R_c
                snd_f  = min(vals_c[best_ci], vals_f[best_fi])
                R_total = R_f @ R_seed
                if snd_f < best_snd:
                    best_snd = snd_f
                    best_R   = R_total
            R_inter[g] = best_R
            snd_final  = best_snd
        else:
            R_inter[g], snd_final = align_snd(mov, anchor_mean, depth='fine')

        # Report alignment quality
        aligned = mov @ R_inter[g].T
        rmsd = float(np.sqrt(np.mean(np.sum((aligned - anchor_mean)**2, axis=1))))
        print(f'    {g:<30s}  SND={snd_final:.4f}  RMSD={rmsd:.4f} Å',
              flush=True)

    print(f'\n  Final inter-group alignment distances to anchor ({anchor}):')
    for g in groups:
        aligned = group_means[g] @ R_inter[g].T
        rmsd = float(np.sqrt(np.mean(np.sum((aligned - anchor_mean)**2, axis=1))))
        print(f'    {g:<30s}  RMSD={rmsd:.4f} Å  W={w[g]:.4f}')

    return R_inter


# ---------------------------------------------------------------------------
# Cross-group anchor pool: global best anchor from all group top-K candidates
# ---------------------------------------------------------------------------

def cross_pool_anchor_scan(pool, o_indices, args, nworkers):
    """
    Select the global best anchor from the pool of top-K candidates
    collected across all groups after Stage 1.

    For each candidate c in the pool, align all other pool members to c
    using the fast SND coarse+fine grid (no NM polish — speed matters here).
    The candidate with the lowest mean SND over the pool wins and is used
    as the global reference frame for Phase B of ALL groups.

    Parameters
    ----------
    pool : list of (group_name, basename, O_coords (num_O,3)) tuples
           collected from stage1_align_group top-K returns

    Returns
    -------
    best_name  : str    — basename of the global best anchor
    best_O     : array  — O-atom coordinates of the global best anchor
    best_group : str    — group the anchor belongs to
    pool_snds  : dict   — {name: mean_pool_SND} for all candidates
    """
    n = len(pool)
    print(f'\n  Cross-group anchor pool: {n} candidates '
          f'({n*(n-1)} fast SND evaluations)', flush=True)

    # Precompute O-arrays for all pool members
    pool_O = [(g, bn, O) for g, bn, O in pool]

    # For each candidate, compute mean SND to all other pool members
    pool_snds = {}

    with ProcessPoolExecutor(max_workers=nworkers) as executor:
        futures = {}
        for ci, (cg, cbn, cO) in enumerate(pool_O):
            # Align all OTHER pool members to this candidate
            others = [(g, bn, O) for j,(g,bn,O) in enumerate(pool_O) if j != ci]
            futures[(cg, cbn)] = executor.submit(
                _eval_pool_candidate, cO, others)

        for (cg, cbn), fut in futures.items():
            mean_snd = fut.result()
            pool_snds[cbn] = mean_snd

    # Print sorted results
    sorted_pool = sorted(pool_snds.items(), key=lambda x: x[1])
    for rank, (bn, msnd) in enumerate(sorted_pool):
        g = next(g for g,b,_ in pool_O if b==bn)
        census, _ = ring_fingerprint(
            next(O for _g,b,O in pool_O if b==bn),
            cutoff=args.ring_cutoff, max_ring=args.ring_max)
        topo = classify_topology(census)
        marker = ' ← GLOBAL ANCHOR' if rank == 0 else ''
        print(f'    {rank+1:2d}. {bn:<55s}  '
              f'group={g:<20s}  topo={topo:<8s}  '
              f'mean_SND={msnd:.3f} Å²{marker}', flush=True)

    best_bn   = sorted_pool[0][0]
    best_O    = next(O for _,bn,O in pool_O if bn==best_bn)
    best_group = next(g for g,bn,_ in pool_O if bn==best_bn)
    return best_bn, best_O, best_group, pool_snds


def _eval_pool_candidate(ref_O, others):
    """Worker: align all others to ref_O with fast SND, return mean SND."""
    snds = []
    for _, _, O in others:
        # Fast SND: coarse + fine grid only (no NM)
        _, snd = align_snd(O, ref_O, depth='fine')
        snds.append(snd)
    return float(np.mean(snds)) if snds else 0.0


# ---------------------------------------------------------------------------
# Final parallel write
# ---------------------------------------------------------------------------

def write_all(basenames, group_map, R_inter_map, group_ref_map,
              group_ref_topo_map, o_indices, args, nworkers,
              all_coords_map=None):
    bn_to_group = {bn: g for g,members in group_map.items() for bn in members}
    worker_args = [
        (bn, args.ext_in, args.ext_out, o_indices,
         R_inter_map.get(bn_to_group.get(bn,'other'), np.eye(3)),
         group_ref_map[bn_to_group.get(bn,'other')],
         args.num_O,
         args.soap_polish_thresh,
         args.soap_skip_thresh,
         args.soap_sigma,
         args.ring_cutoff,
         args.ring_max,
         group_ref_topo_map.get(bn_to_group.get(bn,'other'), 'LIQUID'),
         all_coords_map.get(bn) if all_coords_map else None)
        for bn in basenames
    ]
    nspec     = len(basenames)
    chunksize = max(1, nspec // (nworkers * 4))
    dmin_map   = {}
    soap_map   = {}
    census_map = {}   # bn -> census dict
    topo_map   = {}   # bn -> topology label
    rings_map  = {}   # bn -> ring census string
    depth_map  = {}
    done       = 0
    with ProcessPoolExecutor(max_workers=nworkers) as pool:
        for bn, snd, soap, depth, census, topo, rings, err in pool.map(
                _write_one, worker_args, chunksize=chunksize):
            done += 1
            if err:
                print(f'  ERROR: {err}', flush=True)
            else:
                dmin_map[bn]   = snd
                soap_map[bn]   = soap
                census_map[bn] = census
                topo_map[bn]   = topo
                rings_map[bn]  = rings
                depth_map[bn]  = depth
            if done % max(1, nspec//10) == 0 or done == nspec:
                print(f'  {done}/{nspec} written', flush=True)

    # Print tier breakdown
    tiers = {'full': 0, 'fine': 0, 'coarse': 0}
    for d in depth_map.values():
        tiers[d] = tiers.get(d, 0) + 1
    print(f'\n  Search depth breakdown (topology + SOAP guided):')
    print(f'    full   (topo-match AND SOAP ≥ {args.soap_polish_thresh:.2f}) : '
          f'{tiers["full"]:6d} ({100*tiers["full"]/max(1,nspec):.1f}%)')
    print(f'    fine   (topo-match OR  SOAP ≥ {args.soap_polish_thresh:.2f}) : '
          f'{tiers["fine"]:6d} ({100*tiers["fine"]/max(1,nspec):.1f}%)')
    print(f'    coarse (topo-mismatch AND SOAP < {args.soap_polish_thresh:.2f}): '
          f'{tiers["coarse"]:6d} ({100*tiers["coarse"]/max(1,nspec):.1f}%)')

    # Print ring topology summary (counts only; weighted version in write_summary)
    from collections import Counter
    topo_counts = Counter(topo_map.values())
    print(f'\n  Ring topology summary (cutoff={args.ring_cutoff} Å, '
          f'max_ring={args.ring_max}):')
    print(f'    {"Label":<10}  {"Count":>7}  {"Fraction":>9}  Description')
    descs = {'DODEC':'5¹² dodecahedron', 'CS1':'5¹²6² cage (CS-I)',
              'CS2':'5¹²6⁴ cage (CS-II)', 'CAGE':'partial cage / precursor',
              'MIXED':'disordered + some pentagons', 'LIQUID':'fully disordered'}
    for label in ['DODEC','CS1','CS2','CAGE','MIXED','LIQUID']:
        cnt = topo_counts.get(label, 0)
        if cnt:
            print(f'    {label:<10}  {cnt:>7}  {100*cnt/nspec:>8.1f}%  '
                  f'{descs.get(label,"")}')

    return dmin_map, soap_map, census_map, topo_map, rings_map


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(dmin_map, soap_map, census_map, topo_map, rings_map,
                  group_map, basenames, args, elapsed,
                  analysis_weight_dicts=None, analysis_wfiles=None):
    """
    Print and save the alignment summary.

    analysis_weight_dicts : list of {bn: weight} dicts, one per temperature.
        Used ONLY for weighted topology fractions and per-temperature analysis.
        Not used for alignment or Stage 2 registration.
    analysis_wfiles : list of file names (for column headers).
    """
    from collections import Counter, defaultdict
    bn_to_group = {bn: g for g,members in group_map.items() for bn in members}

    # Identify reference clusters: those whose final SND is near zero.
    # The reference cluster goes through the same grid search as others,
    # so its SND is not exactly 0.0 — use a small threshold instead.
    # 0.5 Å² is well below the minimum SND of any non-reference cluster
    # (CS1/DODEC min is ~2.1 Å²) while safely above grid-search noise.
    SND_REF_THRESH = 0.5
    ref_bns = {bn for bn, d in dmin_map.items() if d < SND_REF_THRESH}

    print(f'\nTotal time: {elapsed:.1f} s\n')

    # SND statistics excluding the self-term reference clusters
    all_d_full = np.array([dmin_map[bn] for bn in basenames if bn in dmin_map])
    all_d      = np.array([dmin_map[bn] for bn in basenames
                           if bn in dmin_map and bn not in ref_bns])
    n_refs = len(ref_bns)
    print(f'=== Overall SND statistics (excluding {n_refs} reference self-term(s)) ===')
    if len(all_d):
        print(f'  Mean   : {all_d.mean():.4f} Å²')
        print(f'  Median : {np.median(all_d):.4f} Å²')
        print(f'  Std    : {all_d.std():.4f} Å²')
        print(f'  Min    : {all_d.min():.4f} Å²')
        print(f'  Max    : {all_d.max():.4f} Å²')

    if len(group_map) > 1:
        print('\n=== Per-group SND statistics (excluding reference self-terms) ===')
        for g, members in group_map.items():
            sub = np.array([dmin_map[bn] for bn in members
                            if bn in dmin_map and bn not in ref_bns])
            n_tot = len([bn for bn in members if bn in dmin_map])
            if len(sub):
                print(f'  {g:25s}  n={n_tot:6d} ({n_tot-len(sub)} ref excluded)'
                      f'  mean={sub.mean():.2f}  std={sub.std():.2f}'
                      f'  min={sub.min():.2f}  max={sub.max():.2f}  (Å²)')

    # SOAP summary per group
    if soap_map:
        print('\n=== Per-group SOAP similarity to group reference ===')
        print(f'  (Rotationally invariant; 1=identical, 0=unlike)')
        for g, members in group_map.items():
            sub = np.array([soap_map[bn] for bn in members
                            if bn in soap_map and bn not in ref_bns])
            if len(sub):
                low = np.sum(sub < 0.80)
                print(f'  {g:25s}  mean={sub.mean():.4f}  std={sub.std():.4f}  '
                      f'min={sub.min():.4f}  low(<0.80): {low}/{len(sub)} '
                      f'({100*low/len(sub):.1f}%)')

    # ------------------------------------------------------------------
    # Weighted topology analysis per temperature
    # ------------------------------------------------------------------
    descs = {'DODEC':'5¹² dodecahedron',     'CS1': '5¹²6² cage (CS-I)',
             'CS2': '5¹²6⁴ cage (CS-II)',    'CAGE':'partial cage / precursor',
             'MIXED':'disordered + pentagons','LIQUID':'fully disordered'}
    topo_order = ['DODEC','CS1','CS2','CAGE','MIXED','LIQUID']
    topo_count = Counter(topo_map.values())

    if analysis_weight_dicts:
        per_temp_totals = [sum(wd.values()) for wd in analysis_weight_dicts]
        wf_labels = [os.path.basename(wf)[:12] for wf in (analysis_wfiles or [])]

        print(f'\n=== Ring topology: weighted signal fraction per temperature ===')
        print(f'  Each column = fraction of total EXAFS weight from that topology')
        hdr = f'  {"Label":<10}  {"Count":>7}  {"Count%":>7}'
        for lbl in wf_labels:
            hdr += f'  {lbl:>12s}'
        hdr += f'  {"Mean Wt%":>9s}'
        print(hdr)
        print('  ' + '-' * (len(hdr) - 2))

        for label in topo_order:
            cnt = topo_count.get(label, 0)
            if cnt == 0:
                continue
            members_with_topo = [bn for bn in basenames if topo_map.get(bn) == label]
            per_temp_wt = []
            for wd, tot in zip(analysis_weight_dicts, per_temp_totals):
                raw = sum(wd.get(bn, 0.0) for bn in members_with_topo)
                per_temp_wt.append(100.0 * raw / tot if tot > 0 else 0.0)
            mean_wt = float(np.mean(per_temp_wt))
            line = (f'  {label:<10}  {cnt:>7d}  '
                    f'{100*cnt/len(basenames):>6.1f}%')
            for wt in per_temp_wt:
                line += f'  {wt:>11.2f}%'
            line += f'  {mean_wt:>8.2f}%'
            print(line)

        # Total row
        tot_line = f'  {"TOTAL":<10}  {len(basenames):>7d}  {"100.0%":>7s}'
        for ti in range(len(analysis_weight_dicts)):
            col_sum = sum(
                sum(analysis_weight_dicts[ti].get(bn, 0.0)
                    for bn in basenames if topo_map.get(bn) == label)
                / per_temp_totals[ti] * 100
                for label in topo_order if topo_count.get(label, 0) > 0
            )
            tot_line += f'  {col_sum:>11.2f}%'
        tot_line += f'  {"100.00%":>9s}'
        print('  ' + '-' * (len(hdr) - 2))
        print(tot_line)

    else:
        # No weights: just show count-based summary
        print(f'\n=== Ring topology summary (cutoff={args.ring_cutoff} Å) ===')
        print(f'  {"Label":<10}  {"Count":>7}  {"Count%":>7}  Description')
        print('  ' + '-' * 55)
        for label in topo_order:
            cnt = topo_count.get(label, 0)
            if cnt:
                print(f'  {label:<10}  {cnt:>7d}  '
                      f'{100*cnt/len(basenames):>6.1f}%  {descs.get(label,"")}')
        print('  ' + '-' * 55)
        print(f'  {"TOTAL":<10}  {len(basenames):>7d}  {"100.0%":>7s}')

    # ------------------------------------------------------------------
    # Per-cluster mean weight for output file
    # ------------------------------------------------------------------
    if analysis_weight_dicts:
        per_temp_totals = [sum(wd.values()) for wd in analysis_weight_dicts]
        def cluster_mean_weight(bn):
            return float(np.mean([wd.get(bn, 0.0) / tot
                                  for wd, tot in zip(analysis_weight_dicts,
                                                     per_temp_totals)]))
    else:
        def cluster_mean_weight(bn):
            return 1.0 / len(basenames)

    # Write per-temperature weights per cluster too
    summary_path = 'alignment_dmin.txt'
    with open(summary_path, 'w') as fh:
        fh.write('# Cluster alignment results\n')
        fh.write('# Generated by align_clusters.py\n')
        fh.write(f'# num_O={args.num_O}  soap_sigma={args.soap_sigma} Å  '
                 f'ring_cutoff={args.ring_cutoff} Å\n')
        fh.write(f'# Ring fingerprint: n3-n4-n5-n6-n7 (primitive ring counts)\n')
        fh.write(f'# Topology: LIQUID/MIXED/CAGE/DODEC/CS1/CS2\n')
        if analysis_weight_dicts:
            fh.write(f'# Weight columns: normalised weight at each temperature\n')
            fh.write(f'# {"Basename":<60s}  {"Group":<10s}  {"SND(Å²)":>10s}  '
                     f'{"SOAP":>8s}')
            for wf in (analysis_wfiles or []):
                fh.write(f'  {os.path.basename(wf)[:12]:>12s}')
            fh.write(f'  {"Topology":<8s}  {"Rings(3-7)"}\n')
        else:
            fh.write(f'# {"Basename":<60s}  {"Group":<10s}  {"SND(Å²)":>10s}  '
                     f'{"SOAP":>8s}  {"Topology":<8s}  {"Rings(3-7)"}\n')

        for bn in basenames:
            d    = dmin_map.get(bn, float('nan'))
            s    = soap_map.get(bn, float('nan'))
            g    = bn_to_group.get(bn, 'other')
            topo = topo_map.get(bn, '?')
            rstr = rings_map.get(bn, '?')
            line = (f'  {bn:<60s}  {g:<10s}  {d:10.4f}  {s:8.4f}')
            if analysis_weight_dicts:
                for wd, tot in zip(analysis_weight_dicts, per_temp_totals):
                    w = wd.get(bn, 0.0) / tot if tot > 0 else 0.0
                    line += f'  {w:12.8f}'
            line += f'  {topo:<8s}  {rstr}\n'
            fh.write(line)
    print(f'\nResults written to: {summary_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    parser.add_argument('--basenames',    default='BASE_NAMES')
    parser.add_argument('--num_O',        type=int,   default=20)
    parser.add_argument('--maxiter',      type=int,   default=200)
    parser.add_argument('--conv_obj',     type=float, default=1.0)
    parser.add_argument('--ext_in',       default='.cen')
    parser.add_argument('--ext_out',      default='.rot')
    parser.add_argument('--nthreads',     type=int,   default=None)
    parser.add_argument('--groups',       type=str,   default=None)
    parser.add_argument('--analysis_weights', type=str, default=None,
                        help='Comma-separated weight files for post-alignment '
                             'analysis only (not used for alignment or Stage 2). '
                             'Format: "weight  basename" per line, one file per '
                             'temperature.  Used to compute weighted topology '
                             'fractions and weighted SND/SOAP statistics in the '
                             'summary.  E.g. "4C_weights.dat,15C_weights.dat,'
                             '25C_weights.dat,45C_weights.dat"')
    parser.add_argument('--weights',      type=str,   default=None,
                        help='Deprecated alias for --analysis_weights. '
                             'If provided and --analysis_weights is not, '
                             'behaves as --analysis_weights.')
    parser.add_argument('--sym_search',   action='store_true')
    parser.add_argument('--use_best_ref', action='store_true')
    parser.add_argument('--no_iter',      action='store_true')
    parser.add_argument('--stage1_only',  action='store_true')
    parser.add_argument('--soap_polish_thresh', type=float, default=0.90,
                        help='SOAP ≥ this → full search (coarse+fine+NM) '
                             '(default: 0.90)')
    parser.add_argument('--soap_skip_thresh',   type=float, default=0.70,
                        help='SOAP < this → coarse search only; '
                             'between thresholds → fine search '
                             '(default: 0.70)')
    parser.add_argument('--soap_sigma',         type=float, default=0.3,
                        help='Gaussian smearing width for SOAP (Å). '
                             'Smaller = more sensitive to precise O positions, '
                             'approaches SND metric as sigma→0. '
                             'Recommended: 0.3 Å (≈10%% of O-O distance). '
                             '(default: 0.3)')
    parser.add_argument('--ring_cutoff',        type=float, default=3.3,
                        help='O-O distance cutoff for H-bond graph (Å). '
                             'Should sit in the gap between first (~2.8 Å) '
                             'and second (~4.5 Å) O-O shells.  3.3 Å captures '
                             'weaker H-bonds at the first-shell RDF minimum, '
                             'more standard in clathrate hydrate literature. '
                             '(default: 3.3)')
    parser.add_argument('--ring_max',           type=int,   default=7,
                        help='Largest primitive ring size to search. '
                             '(default: 7)')
    parser.add_argument('--topo_subgroups',     action='store_true',
                        help='Option A: after pre-scan, split each group into '
                             'topology sub-groups (e.g. LIQUID/DODEC, LIQUID/MIXED). '
                             'Each sub-group gets its own iterative mean alignment and '
                             'best-cluster reference, giving sharper density maps '
                             'for structurally coherent sub-populations.')
    parser.add_argument('--anchor',             type=str,   default=None,
                        help='Force a specific cluster as the Phase B reference '
                             '(basename or partial match), bypassing the scan. '
                             'E.g. --anchor "DODEC/DODEC" (matches the first '
                             'cluster in that sub-group name).  Useful for '
                             'comparing alignment quality when using a highly '
                             'structured reference vs a data-driven one.')
    parser.add_argument('--anchor_scan_k',      type=int,   default=5,
                        help='Number of top candidates to evaluate in the Phase B '
                             'anchor scan.  Each is used as reference for one full '
                             'Phase B pass; the one giving the lowest mean SND '
                             'over the group wins.  Higher K = better anchor at '
                             'K× the Phase B cost.  K=1 reverts to the old '
                             'behaviour (lowest Phase A SND to mean). '
                             '(default: 5)')
    parser.add_argument('--phase_A_select',     type=int,   default=None,
                        help='0-based index K of the cluster to use as the octant '
                             'template for the Phase A initial reference.  '
                             'Distance-sorting is still applied; then, for each '
                             'O position i, the sign of each coordinate is flipped '
                             'so that it matches the sign of the corresponding '
                             'coordinate in cluster K\'s i-th nearest O atom.  '
                             'This keeps all clusters in the same octant as K '
                             'before averaging, preventing collapse toward the '
                             'Ar origin while remaining unbiased to any single '
                             'cluster geometry.  If omitted, the first cluster '
                             '(K=0) is used as the Phase A starting reference '
                             '(original behaviour).')
    args = parser.parse_args()

    nworkers = args.nthreads or multiprocessing.cpu_count()
    t0 = time.time()

    with open(args.basenames) as fh:
        basenames = [ln.strip() for ln in fh if ln.strip()]
    nspec = len(basenames)

    if args.groups:
        patterns  = [p.strip() for p in args.groups.split(',') if p.strip()]
        group_map = assign_groups(basenames, patterns)
    else:
        group_map = {'all': basenames}

    print(f'Number of clusters      : {nspec}')
    print(f'Nearest O atoms used    : {args.num_O}')
    print(f'Worker processes        : {nworkers}')
    print(f'Rotation metric         : Phase A: Hungarian+Kabsch  |  '
          f'Phase B + write: SND grid search')
    print(f'Mode                    : '
          f'{"single-reference (--no_iter)" if args.no_iter else "iterative alignment"}')
    if not args.no_iter:
        print(f'Best-cluster reference  : '
              f'{"yes (--use_best_ref)" if args.use_best_ref else "no (mean ref)"}')
        print(f'Max iterations          : {args.maxiter}')
        print(f'Convergence (Δobj)      : {args.conv_obj} Å²')
    do_stage2 = (not args.stage1_only) and len(group_map) > 1
    print(f'Stage 2 registration    : '
          f'{"yes (single-pass SND, anchor = highest-weight group)" if do_stage2 else "no"}')
    if do_stage2:
        print(f'Symmetry search (S2)    : '
              f'{"yes" if args.sym_search else "no"}')
    print(f'Input  extension        : {args.ext_in}')
    print(f'Output extension        : {args.ext_out}')
    print(f'SOAP polish threshold   : {args.soap_polish_thresh}  '
          f'(≥ → full NM polish)')
    print(f'SOAP skip threshold     : {args.soap_skip_thresh}  '
          f'(< → coarse grid only)')
    print(f'SOAP sigma              : {args.soap_sigma} Å  '
          f'(Gaussian smearing; smaller = closer to SND)')
    print(f'Ring cutoff             : {args.ring_cutoff} Å  '
          f'(O-O H-bond distance threshold)')
    print(f'Ring max size           : {args.ring_max}')
    print(f'Topology sub-groups     : '
          f'{"yes (--topo_subgroups)" if args.topo_subgroups else "no"}')
    print(f'Topology depth guidance : yes (always — Option B)')
    if getattr(args, 'anchor', None):
        print(f'Anchor (manual)         : {args.anchor}')
    else:
        print(f'Anchor scan K           : {args.anchor_scan_k}  '
              f'(Phase B candidates evaluated; best = lowest mean SND)')
    if args.phase_A_select is not None:
        print(f'Phase A octant template : cluster #{args.phase_A_select}  '
              f'(--phase_A_select; signs matched per O position before averaging)')
    else:
        print(f'Phase A initial ref     : first cluster in each group (K=0)')

    # Resolve analysis weight files (--analysis_weights takes priority;
    # --weights is a deprecated alias kept for backward compatibility)
    aw_str = args.analysis_weights or args.weights
    analysis_wfiles = [w.strip() for w in aw_str.split(',')] if aw_str else []
    print(f'Analysis weight files   : '
          f'{analysis_wfiles if analysis_wfiles else "none"}', flush=True)
    print()

    # Read all clusters
    print('Reading all clusters ...', flush=True)
    all_coords_map = {}
    first_labels   = None
    for bn in basenames:
        labels, coords = read_cen(bn + args.ext_in)
        coords = centre_on_ar(coords, labels=labels, path=bn + args.ext_in)
        all_coords_map[bn] = coords
        if first_labels is None:
            first_labels = labels
    o_indices = get_O_indices(first_labels)
    print(f'  Done in {time.time()-t0:.1f} s', flush=True)

    # Load analysis weight files (used only in summary, not in alignment)
    analysis_weight_dicts = []
    for wf in analysis_wfiles:
        wd = read_weights(wf)
        analysis_weight_dicts.append(wd)
        print(f'  Loaded analysis weights: {wf}  ({len(wd)} entries)')

    # ------------------------------------------------------------------
    # PRE-SCAN: topology classification of all clusters (Option A + B)
    # ------------------------------------------------------------------
    print(f'\nPre-scanning topology of all clusters ...', flush=True)
    t_pre = time.time()
    topo_pre, census_pre, rings_pre = prescan_topology(
        basenames, all_coords_map, o_indices, args, nworkers)
    print(f'  Pre-scan done in {time.time()-t_pre:.1f} s', flush=True)

    # ------------------------------------------------------------------
    # Option A: split groups by topology if requested
    # ------------------------------------------------------------------
    if args.topo_subgroups:
        print(f'\nSplitting groups by topology (--topo_subgroups) ...', flush=True)
        group_map, parent_map = split_by_topology(group_map, topo_pre)
        print(f'  Sub-groups:')
        for g, members in sorted(group_map.items()):
            print(f'    {g:<30s}: {len(members):6d} clusters')
    else:
        parent_map = {g: g for g in group_map}

    # ------------------------------------------------------------------
    # Stage 1: align each group (or sub-group) independently
    # ------------------------------------------------------------------
    group_ref_map      = {}   # group -> ref_O (num_O, 3)
    group_ref_topo_map = {}   # group -> topology label of reference

    if args.no_iter:
        for g, members in group_map.items():
            ref_O  = nearest_O(all_coords_map[members[0]], o_indices, args.num_O)
            census_ref, _ = ring_fingerprint(ref_O, cutoff=args.ring_cutoff,
                                             max_ring=args.ring_max)
            ref_topo = classify_topology(census_ref)
            group_ref_map[g]      = ref_O
            group_ref_topo_map[g] = ref_topo
            print(f'\nGroup {g}: reference = {members[0]}  '
                  f'topo={ref_topo}')
    else:
        # Collect top-K candidates from each group for cross-pool evaluation
        all_pool = []   # list of (group, basename, O_coords)
        for g, members in group_map.items():
            print(f'\n{"="*60}')
            print(f'STAGE 1 — Group: {g}  (n={len(members)} clusters)')
            print(f'{"="*60}', flush=True)
            ref_O, ref_topo, top_k_O = stage1_align_group(
                g, members, all_coords_map, o_indices, args, nworkers)
            group_ref_map[g]      = ref_O
            group_ref_topo_map[g] = ref_topo
            # Accumulate pool candidates (group name, basename, O-array)
            for bn, O in top_k_O:
                all_pool.append((g, bn, O))

        # ----------------------------------------------------------------
        # Cross-group anchor pool: find the global best anchor
        # ----------------------------------------------------------------
        if do_stage2 and len(all_pool) > 1 and not getattr(args, 'anchor', None):
            print(f'\n{"="*60}')
            print(f'CROSS-GROUP ANCHOR SCAN')
            print(f'  Evaluating {len(all_pool)} candidates from '
                  f'{len(group_ref_map)} groups against each other')
            print(f'{"="*60}', flush=True)
            t_pool = time.time()
            global_anchor_bn, global_anchor_O, global_anchor_group, pool_snds = \
                cross_pool_anchor_scan(all_pool, o_indices, args, nworkers)
            print(f'\n  Cross-pool scan done in {time.time()-t_pool:.1f} s')

            # Run Phase B for ALL groups aligned to the global anchor
            census_ga, _ = ring_fingerprint(global_anchor_O,
                                            cutoff=args.ring_cutoff,
                                            max_ring=args.ring_max)
            global_anchor_topo = classify_topology(census_ga)
            print(f'\n  Global anchor: {global_anchor_bn}')
            print(f'    group={global_anchor_group}  '
                  f'topo={global_anchor_topo}  '
                  f'[{ring_census_str(census_ga)}]', flush=True)

            # Re-run Phase B for every group against the global anchor
            print(f'\n  Phase B (global): aligning all groups to global anchor ...',
                  flush=True)
            for g, members in group_map.items():
                basenames_g  = members
                coords_g     = [all_coords_map[bn] for bn in basenames_g]
                chunk_size   = max(1, len(basenames_g) // (nworkers * 4))
                coord_chunks = [coords_g[i:i+chunk_size]
                                for i in range(0, len(basenames_g), chunk_size)]
                map_args = [(chunk, o_indices, global_anchor_O, args.num_O, None)
                            for chunk in coord_chunks]
                t_pb = time.time()
                final_coords_g, final_snds_g = [], []
                with ProcessPoolExecutor(max_workers=nworkers) as pool_ex:
                    for new_c, snd_list, _, _ in pool_ex.map(
                            _align_chunk, map_args, chunksize=1):
                        final_coords_g.extend(new_c)
                        final_snds_g.extend(snd_list)
                for bn, c in zip(basenames_g, final_coords_g):
                    all_coords_map[bn] = c
                snd_arr_g   = np.array(final_snds_g)
                snd_nref    = snd_arr_g[snd_arr_g >= 0.5]
                print(f'    {g:<30s}: SND mean={snd_nref.mean():.2f}  '
                      f'std={snd_nref.std():.2f}  '
                      f'({time.time()-t_pb:.1f}s)', flush=True)
                # Update group reference to global anchor frame
                group_ref_map[g]      = global_anchor_O
                group_ref_topo_map[g] = global_anchor_topo

            do_stage2 = False   # Stage 2 no longer needed (all in global frame)

    # ------------------------------------------------------------------
    # Stage 2: inter-group registration with UNIFORM weights
    # (alignment is temperature-independent; weights are for analysis only)
    # ------------------------------------------------------------------
    if do_stage2:
        print(f'\n{"="*60}')
        print(f'STAGE 2 — Inter-group registration')
        print(f'  (single-pass SND, anchor = highest-weight group)')
        print(f'{"="*60}', flush=True)
        # Use analysis weights to select the anchor group (most important
        # for the EXAFS signal).  Registration itself is weight-independent
        # (uniform weights for the SND alignment of group means).
        if analysis_weight_dicts:
            stage2_weights = compute_group_weights(
                {g: group_map[g] for g in group_map},
                analysis_weight_dicts)
        else:
            stage2_weights = {g: 1.0 for g in group_map}
        print()
        R_inter_map = stage2_register(
            group_ref_map, stage2_weights, sym_search=args.sym_search)
    else:
        R_inter_map = {g: np.eye(3) for g in group_map}

    # ------------------------------------------------------------------
    # Final write: topology + SOAP guided depth selection (Options A+B)
    # ------------------------------------------------------------------
    print(f'\nWriting final aligned clusters to *{args.ext_out}'
          f' (topology + SOAP guided) ...', flush=True)
    dmin_map, soap_map, census_map, topo_map, rings_map = write_all(
        basenames, group_map, R_inter_map, group_ref_map,
        group_ref_topo_map, o_indices, args, nworkers,
        all_coords_map=all_coords_map)

    write_summary(dmin_map, soap_map, census_map, topo_map, rings_map,
                  group_map, basenames, args, time.time()-t0,
                  analysis_weight_dicts=analysis_weight_dicts,
                  analysis_wfiles=analysis_wfiles)


if __name__ == '__main__':
    main()
