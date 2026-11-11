# ARCS offline analysis code (code-only candidate)

This GitHub candidate contains analysis code and protocols only. It deliberately excludes research bundles, source observations, organization-level tables, result JSON/CSV files, logs, caches, plans, hostnames, IP addresses, and screenshots.

## Scope

The included runner implements the retrospective broader-candidate sensitivity analysis and common clean-run alert-budget comparison. It reads frozen bundles from the existing ARCS project tree and does not make network requests or contact research targets. The candidate does not contain those bundles and therefore cannot independently reproduce the paper's numerical results.

## Integration

The original project modules are dependencies and are not copied into this small add-on. To use the code within the complete project source tree, preserve the relative location `outputs/emse_three_open_items_20261105/`. The source modules under `experiments/exp_arcs/` and compatible local input files must already be available to the analyst. No research data are included or linked here.

The analysis runner imports NumPy, scikit-learn, PyYAML, and the original project's Python modules. The project environment is specified in its source repository; this package intentionally does not copy the full environment lockfile or unrelated application dependencies. A short analysis dependency list is provided in `requirements-analysis.txt`.

## Data and access

This code-only repository provides no data-access route. Do not describe the underlying bundles as available from this repository. The archived source data and per-host planner outputs remain outside this package and are subject to separate rights and privacy review.

## Post hoc status

Both protocols and analyses are retrospective and exploratory, not preregistered or independent validation. The budget analysis uses leave-one-organization-out calibration and a deterministic host-hash tie break. See the protocol amendment recorded in the budget protocol.

## License

No license is included in this candidate. Confirm ownership and redistribution rights for every included code file before adding a public license.

## Detailed provenance and reproducibility limits

See [DATA_ACCESS.md](DATA_ACCESS.md). The local aggregate review artifact is excluded and has no approved external access route.
