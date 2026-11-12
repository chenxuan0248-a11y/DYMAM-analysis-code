"""Reviewer item H (FP mechanism) — does the peer-anomaly false-alarm rate scale
with organization size?

Hypothesis a reviewer will raise: peer-anomaly uses THIS run's cross-host family
distribution, so a SMALL org (few hosts) has a noisy base-rate estimate -> a host
looks anomalously sparse more easily -> higher false-alarm rate. Test it on the 20
fault-free (clean) E3 runs: per org, universe size vs combined/peer false-alarm
rate (every flag on a clean run is a false positive). Spearman rho over 20 orgs.

Cheap: jsonl reads only, reuses detector_eval signal fns, no DAG builds. All FP is
peer-anomaly's (self=0, manifest=0 on clean; e3_detector_fp_v1.json) -- verified
here too.

    poetry run python -m experiments.exp_arcs.detector_fp_by_orgsize
"""
from __future__ import annotations
import json
import statistics as st
from pathlib import Path

from experiments.exp_arcs.detector_eval import (
    detector_localize, peer_anomaly_hosts, manifest_dropped_hosts,
)

D = Path("experiments/exp_arcs")
OUT = D / "detector_fp_by_orgsize_v1.json"


def _n_hosts(run_dir: Path) -> int:
    hosts = set()
    for ln in (run_dir / "evidence_event.jsonl").read_text().splitlines():
        if ln.strip():
            hosts.add(json.loads(ln)["target"])
    return len(hosts)


def _spearman(xs, ys):
    """Spearman rho via Pearson on ranks (ties -> average rank)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 3) if den else 0.0


def main() -> int:
    grid = json.loads((D / "e3_grid_v1.json").read_text())["cells"]
    clean_by_org = {}
    for c in grid:
        clean_by_org.setdefault(c["org"], c["clean_dir"])

    rows = []
    for org, cd in sorted(clean_by_org.items()):
        p = Path(cd)
        n = _n_hosts(p)
        self_f = detector_localize(p)["flagged_hosts"]
        peer_f = peer_anomaly_hosts(p)          # default t=0.5
        man_f = manifest_dropped_hosts(p)
        comb = self_f | peer_f | man_f
        rows.append({
            "org": org, "n_hosts": n,
            "self_fp": len(self_f), "peer_fp": len(peer_f), "manifest_fp": len(man_f),
            "combined_fp": len(comb),
            "combined_fp_rate": round(len(comb) / n, 4) if n else None,
            "peer_fp_rate": round(len(peer_f) / n, 4) if n else None,
        })

    sizes = [r["n_hosts"] for r in rows]
    fp_rate = [r["combined_fp_rate"] for r in rows]
    peer_rate = [r["peer_fp_rate"] for r in rows]
    rho_size_fp = _spearman(sizes, fp_rate)
    rho_size_peer = _spearman(sizes, peer_rate)

    all_self = sum(r["self_fp"] for r in rows)
    all_man = sum(r["manifest_fp"] for r in rows)

    result = {
        "phase": "reviewer-H false-alarm vs org size (20 clean runs)",
        "n_orgs": len(rows),
        "all_fp_is_peer": {"total_self_fp": all_self, "total_manifest_fp": all_man,
                           "note": "confirms self=0, manifest=0 on clean runs -> all FP is peer-anomaly"},
        "spearman_rho_size_vs_combined_fp_rate": rho_size_fp,
        "spearman_rho_size_vs_peer_fp_rate": rho_size_peer,
        "mean_combined_fp_rate": round(st.mean(fp_rate), 4),
        "org_size_range": [min(sizes), max(sizes)],
        "rows": sorted(rows, key=lambda r: r["n_hosts"]),
    }
    OUT.write_text(json.dumps(result, indent=2))

    print(f"{'org':12} {'n_hosts':>7} {'peer_fp':>7} {'comb_fp':>7} {'fp_rate':>7}")
    for r in result["rows"]:
        print(f"{r['org']:12} {r['n_hosts']:>7} {r['peer_fp']:>7} {r['combined_fp']:>7} "
              f"{r['combined_fp_rate']:>7}")
    print(f"\nself FP total={all_self}, manifest FP total={all_man} (all FP is peer-anomaly if both 0)")
    print(f"Spearman rho(size, combined_fp_rate) = {rho_size_fp}")
    print(f"Spearman rho(size, peer_fp_rate)     = {rho_size_peer}")
    print(f"mean combined FP rate = {result['mean_combined_fp_rate']} "
          f"(size range {result['org_size_range']})")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
