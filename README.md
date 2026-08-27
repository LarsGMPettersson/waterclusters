Description of contents
--- recenter_argon.py ------
*** Purpose:
Center the argon in the cavity formed by its nearest neighbors out to R_max.
Example of a structure file given in TIP42005_LIQUID_253K_5ATM11.xyz
BASE_NAMES (provided) lists the files to process (without extension).

*** Usage
    python recenter_argon.py [options]

    --basenames FILE   File listing cluster base names      (default: BASE_NAMES)
    --ext_in    STR    Input file extension                 (default: .xyz)
    --ext_out   STR    Output file extension                (default: .cen)
    --R_max     FLOAT  O-Ar distance cutoff for Stage 1    (default: 5.5 Å)
    --nthreads  INT    Worker processes                     (default: all CPUs)
    --out       STR    Diagnostic summary file              (default: recentering.txt)
    --short_thresh FLOAT  Flag clusters with any O < this  (default: 2.9 Å)
=======================================================================================
    --- align_clusters.py
    *** Purpose:
    Rotate centered clusters (from recenter_argon.py) to maximize overlap in oxygen atom positions
    Output: Rotated xyz coordinate files to be used as input for density_average.py
    *** Usage:

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
==============================================================================
--- density_average.py ---
*** Purpose:
Assign Gaussians to each center in the centered and rotated cluster files. Add up the densities
on a cubic grid and output in format suitable for visualizing with VESTA (extension .ped)

*** Usage:
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
=============================================================================
--- angle_distribution.py ---
*** Purpose:

