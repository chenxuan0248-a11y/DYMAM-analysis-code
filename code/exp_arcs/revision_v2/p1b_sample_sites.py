"""P1-B new-site study: build the frozen injection-site list (read-only on frozen bundles).

Implements the sampling rules of STEP4_NEW_SITE_PROTOCOL.md section 3. Uses only the
20 frozen clean bundles and e3_grid_v1.json; never reads any faulted result of a new
site (none exists when this runs) and never reads detector scores, alerts or Success.

Strata (on the clean run's evidence families F(h) of host h):
  dns_only   F(h) == {dns}
  dns_multi  dns in F(h) and F(h) != {dns}
  ao         active_observation in F(h)
Eligibility: valid host identity; for the DNS strata, h was not a target of the original
F4@dns or F4-2host-dns cells of that org; for ao, h was not the original
F4@active_observation target. Per org and stratum, draw min(n, |pool|) hosts without
replacement from the name-sorted pool, n = 3 (dns_only, dns_multi) or 5 (ao).
RNG: numpy PCG64 seeded with [MASTER, child], child = first 8 bytes of
SHA-256("<org>|<stratum>|<role>") as an unsigned integer; role is "formal" or "pilot".
Pilot orgs (mozilla, reddit) get one host per stratum (role "pilot") and are excluded
from the formal sample.

Output: revision_v2/p1b_site_list_v1.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
D = REPO / "experiments/exp_arcs"
GRID = D / "e3_grid_v1.json"
OUT = D / "revision_v2/p1b_site_list_v1.json"
MASTER = 20261008
PILOT_ORGS = ("mozilla", "reddit")
N_PER = {"dns_only": 3, "dns_multi": 3, "ao": 5}
FAMILY = {"dns_only": "dns", "dns_multi": "dns", "ao": "active_observation"}
EXCLUDE_VARIANTS = {"dns_only": ("F4@dns", "F4-2host-dns"), "dns_multi": ("F4@dns", "F4-2host-dns"),
                    "ao": ("F4@active_observation",)}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def child_seed(org: str, stratum: str, role: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{org}|{stratum}|{role}".encode()).digest()[:8], "big")


def inject_hosts(inode: list) -> list[str]:
    kind, who, *_ = inode
    return [who] if kind == "target" else list(who)


def base_domains(org: str) -> set[str]:
    doc = json.loads((REPO / f"data/gt/{org}_gt.json").read_text())
    return set(doc["base_domains"])          # only this field is read


def main() -> int:
    grid = json.loads(GRID.read_text())["cells"]
    by_org = defaultdict(list)
    for c in grid:
        by_org[c["org"]].append(c)
    sites, pools = [], {}
    for org in sorted(by_org):
        cells = by_org[org]
        clean = REPO / cells[0]["clean_dir"]
        fams: dict[str, set] = defaultdict(set)
        for line in (clean / "evidence_event.jsonl").read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                if isinstance(e.get("target"), str) and e["target"]:
                    fams[e["target"]].add(e.get("source_family"))
        manifest = set(json.loads((clean / "discovery_manifest.json").read_text()).get("resolved_hosts", []))
        roots = base_domains(org)
        orig = {c["cell"]: inject_hosts(c["inject_node"]) for c in cells}
        used_any = {h: sorted(v for v, hs in orig.items() if h in hs) for v, hs in orig.items() for h in hs}
        role = "pilot" if org in PILOT_ORGS else "formal"
        for stratum, n in N_PER.items():
            fam = FAMILY[stratum]
            if stratum == "dns_only":
                ok = lambda F: F == {"dns"}
            elif stratum == "dns_multi":
                ok = lambda F: "dns" in F and F != {"dns"}
            else:
                ok = lambda F: "active_observation" in F
            excl = {h for v in EXCLUDE_VARIANTS[stratum] for h in orig.get(v, [])}
            pool = sorted(h for h, F in fams.items() if ok(F) and h not in excl)
            k = 1 if role == "pilot" else min(n, len(pool))
            rng = np.random.Generator(np.random.PCG64([MASTER, child_seed(org, stratum, role)]))
            pick = sorted(rng.choice(pool, size=min(k, len(pool)), replace=False).tolist()) if pool else []
            pools[f"{org}|{stratum}"] = {"N_pool": len(pool), "n_drawn": len(pick), "role": role,
                                         "n_excluded_original": len(excl & set(h for h, F in fams.items() if ok(F)))}
            for h in pick:
                sites.append({"site_id": f"{org}|{stratum}|{h}", "org": org, "stratum": stratum, "role": role,
                              "host": h, "family": fam, "inject_node": ["target", h, fam],
                              "clean_families": sorted(f for f in fams[h] if f),
                              "in_clean_manifest": h in manifest, "is_root": h in roots,
                              "used_by_original_variants": used_any.get(h, [])})
    out = {"protocol": "experiments/exp_arcs/revision_v2/STEP4_NEW_SITE_PROTOCOL.md",
           "script_sha256": sha(Path(__file__)), "grid_sha256": sha(GRID),
           "master_seed": MASTER, "pilot_orgs": list(PILOT_ORGS), "n_per_stratum": N_PER,
           "pools": pools, "sites": sites,
           "counts": {r: {s: sum(1 for x in sites if x["role"] == r and x["stratum"] == s) for s in N_PER}
                      for r in ("formal", "pilot")},
           "orgs_with_sites": {s: len({x["org"] for x in sites if x["role"] == "formal" and x["stratum"] == s})
                               for s in N_PER}}
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"counts": out["counts"], "orgs_with_sites": out["orgs_with_sites"]}))
    print(f"wrote {OUT}  sha256={sha(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
