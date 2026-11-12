# EMSE Submission Artifacts — DyMAM

This repository release contains the manuscript source, selected analysis code, and aggregate figures and tables reported in the study. Host-level prediction data, per-run result JSON, original reconnaissance bundles, and unreviewed derived-data candidates are not included.

## Contents

- `paper/`: LaTeX manuscript source, bibliography, Springer Nature class/style files, and manuscript figures.
- `code/`: ARCS analysis and evaluation scripts; the full DyMAM runtime system is not included.
- `experiments/`: frozen design notes, scope note, and figure/table generation scripts.
- `results/`: aggregate figures and LaTeX table sources.

The release includes 73 top-level ARCS scripts, 45 supplemental revision scripts, 20 result-table TeX files, and aggregate figures. It does not support an end-to-end rerun because the host-level inputs and per-run result records are withheld.

## Build the manuscript

From `paper/`, compile `sn-article.tex` with a LaTeX toolchain that supports the included figures and bibliography.

## Data and code availability

> **Data and code availability.** The analysis code, evaluation scripts, and aggregate figures and tables reported in this study will be available from the project repository at <https://github.com/chenxuan0248-a11y/DYMAM-analysis-code> upon publication. Host-level prediction records and original reconnaissance bundles are not included. Some underlying raw reconnaissance data may be requested from the corresponding author, subject to applicable data-use terms, privacy safeguards, and review; access cannot be guaranteed.

The aggregate artifacts support inspection of reported results, not independent reconstruction of measurements from original observations. The repository does not contain a code or data licence; reuse permissions must be confirmed with the authors.
