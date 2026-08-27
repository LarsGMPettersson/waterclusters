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
    Output: Cubic grid of computed density in format adapted for visualization using VESTA (extension
    .ped).
