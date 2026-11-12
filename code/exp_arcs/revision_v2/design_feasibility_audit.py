"""Post-hoc design audit of frozen artifacts. Standard library only; no pipeline imports.

Run from repository root:
  python3 experiments/exp_arcs/revision_v2/design_feasibility_audit.py
Writes a new diagnostic JSON under the gitignored _audit directory. Does not
run Step 3, choose new injection sites, fit detectors, or replay the pipeline.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/exp_arcs"
OUT = BASE / "_audit/design_feasibility_20261008/result_v2.json"
INPUT_HASHES = {}


def read(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    raw = path.read_bytes()
    INPUT_HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return raw.decode()


def js(path):
    return json.loads(read(path))


def rows(path):
    return [json.loads(line) for line in read(path).splitlines() if line.strip()]


def ev_index(directory):
    evs = rows(Path(directory) / "evidence_event.jsonl")
    fams, per_fam = defaultdict(set), defaultdict(Counter)
    for e in evs:
        fams[e["target"]].add(e.get("source_family"))
        per_fam[e.get("source_family")][e["target"]] += 1
    return fams, per_fam, Counter(e["target"] for e in evs)


def flags(directory, thresholds):
    """Read-only transcription of detector_eval's three union components.

    No repair-success reconstruction here. Output nesting is checked directly;
    Success monotonicity additionally relies on the existing coverage predicate.
    """
    fams, per_fam, _ = ev_index(directory)
    claims = {c["claim_id"]: c.get("subject")
              for c in rows(Path(directory) / "claim.jsonl")}
    fixed = set()
    for d in rows(Path(directory) / "decision.jsonl"):
        if d.get("decision") == "revalidate" or "no_supporting_observation" in (d.get("reason_codes") or []):
            h = claims.get(d.get("claim_id"))
            if h:
                fixed.add(h)
    manifest = js(Path(directory) / "discovery_manifest.json")
    manifest_hosts = set(manifest.get("resolved_hosts", []))
    fixed |= manifest_hosts - set(fams)
    out = {}
    for t in thresholds:
        expected = {f for f, hosts in per_fam.items()
                    if fams and len(hosts) / len(fams) >= t}
        out[t] = fixed | {h for h, fs in fams.items() if expected - fs}
    return out, len(set(fams) | manifest_hosts)


def main():
    grid = js(BASE / "e3_grid_v1.json")["cells"]
    step2 = js(BASE / "revision_v2/step2_baselines_v1.json")
    sweep = js(BASE / "detector_threshold_sweep_v2.json")
    ts = sorted(sweep["thresholds"])
    org_cells = defaultdict(list)
    for c in grid:
        org_cells[c["org"]].append(c)
    inventories, selector_errors, clean_rates = {}, [], {}
    for org, cells in sorted(org_cells.items()):
        directory = cells[0]["clean_dir"]
        fams, per_fam, totals = ev_index(directory)
        rich = {h for h, fs in fams.items() if len(fs) >= 2}
        predicted = {}
        for family in ["dns", "active_observation", "certificate", "registration"]:
            predicted[f"F4@{family}"] = [max(sorted(per_fam[family]), key=per_fam[family].get)]
        predicted["F10-local"] = [max(sorted(per_fam["active_observation"]), key=per_fam["active_observation"].get)]
        predicted["F4-host"] = [max(sorted(rich), key=lambda h: (len(fams[h]), totals[h]))]
        predicted["F4-2host-dns"] = sorted(per_fam["dns"], key=lambda h: (-per_fam["dns"][h], h))[:2]
        ranked = sorted(totals, key=lambda h: (-totals[h], h))
        predicted["F10-2host"], predicted["F10-3host"] = ranked[:2], ranked[:3]
        old_hosts = set()
        for c in cells:
            inode = c["inject_node"]
            actual = [inode[1]] if inode[0] == "target" else inode[1]
            old_hosts.update(actual)
            if set(actual) != set(predicted[c["cell"]]):
                selector_errors.append([org, c["cell"], actual, predicted[c["cell"]]])
        item = {}
        manifest_hosts = set(js(Path(directory) / "discovery_manifest.json").get("resolved_hosts", []))
        for family in ["dns", "active_observation"]:
            candidates = set(per_fam[family])
            only = {h for h in candidates if fams[h] == {family}}
            excluded = set(predicted[f"F4@{family}"])
            eligible = candidates - excluded
            item[family] = {
                "n_hosts": len(candidates), "n_only_family": len(only),
                "eligible_after_same_variant_original": len(eligible),
                "eligible_only_family": len(eligible & only),
                "eligible_multifamily": len(eligible - only),
                "eligible_after_any_old_injection_host": len(candidates - old_hosts),
                "strict_eligible_only_family": len((candidates - old_hosts) & only),
                "strict_eligible_multifamily": len((candidates - old_hosts) - only),
                "manifest_contains_all_candidates": candidates <= manifest_hosts,
                "only_family_manifest_listed": len(only & manifest_hosts),
                "eligible_only_manifest_listed": len(eligible & only & manifest_hosts),
                "eligible_only_manifest_unlisted": len((eligible & only) - manifest_hosts),
                "eligible_multi_manifest_listed": len((eligible - only) & manifest_hosts),
                "eligible_multi_manifest_unlisted": len((eligible - only) - manifest_hosts),
            }
        item["planned_dns_3plus3"] = min(3, item["dns"]["eligible_only_family"]) + min(3, item["dns"]["eligible_multifamily"])
        item["planned_ao_5"] = min(5, item["active_observation"]["eligible_after_same_variant_original"])
        inventories[org] = item
        fs, denominator = flags(directory, ts)
        clean_rates[org] = {str(t): len(fs[t]) / denominator for t in ts}
    nesting_errors = []
    directories = sorted({c[k] for c in grid for k in ["clean_dir", "faulted_dir"]})
    for directory in directories:
        fs, _ = flags(directory, ts)
        for lower, higher in zip(ts, ts[1:]):
            if not fs[higher] <= fs[lower]:
                nesting_errors.append([directory, lower, higher])
    folds = {}
    for alpha in [0.106, 0.020]:
        entries = []
        for held in sorted(clean_rates):
            train = {t: st.mean(clean_rates[o][str(t)] for o in clean_rates if o != held) for t in ts}
            feasible = [t for t in ts if train[t] <= alpha]
            t = min(feasible) if feasible else None
            entries.append({"held_out": held, "threshold": t,
                            "train_macro_alert_rate": train[t] if t is not None else None,
                            "held_out_alert_rate": clean_rates[held][str(t)] if t is not None else None,
                            "held_out_exceeds_alpha": clean_rates[held][str(t)] > alpha if t is not None else None})
        folds[str(alpha)] = entries
    summary = {}
    for family in ["dns", "active_observation"]:
        nums = [v[family]["n_hosts"] for v in inventories.values()]
        summary[family] = {"min_hosts": min(nums), "max_hosts": max(nums),
                           "only_family_total": sum(v[family]["n_only_family"] for v in inventories.values()),
                           "only_family_manifest_listed": sum(v[family]["only_family_manifest_listed"] for v in inventories.values()),
                           "orgs_with_at_least_5_only": sum(v[family]["n_only_family"] >= 5 for v in inventories.values())}
    formal = {o: v for o, v in inventories.items() if o not in {"mozilla", "reddit"}}
    summary["formal_design"] = {
        "n_orgs": len(formal),
        "dns_only_orgs": sum(v["dns"]["eligible_only_family"] > 0 for v in formal.values()),
        "dns_multifamily_orgs": sum(v["dns"]["eligible_multifamily"] > 0 for v in formal.values()),
        "dns_only_cells": sum(min(3, v["dns"]["eligible_only_family"]) for v in formal.values()),
        "dns_multifamily_cells": sum(min(3, v["dns"]["eligible_multifamily"]) for v in formal.values()),
        "ao_cells": sum(v["planned_ao_5"] for v in formal.values()),
        "dns_only_manifest_listed_orgs": sum(v["dns"]["eligible_only_manifest_listed"] > 0 for v in formal.values()),
    }
    cvar = step2["combined"]["primary"]["summary"]["by_variant"]
    ivar = step2["baselines"]["iforest|manifest_aware"]["by_alpha"]["primary"]["summary"]["by_variant"]
    subgroups = {v: {"combined_t05": cvar[v], "iforest_alpha0106": ivar[v],
                    "detect1_difference_rounded": round(cvar[v]["detect1"] - ivar[v]["detect1"], 4)}
                for v in ["F4@dns", "F4@active_observation"]}
    for filename in ["local_oracle.py", "run_e3.py", "detector_eval.py", "select.py", "counterfactual.py", "revision_v2/step2_baselines.py"]:
        read(BASE / filename)
    read(Path(__file__))
    result = {"analysis_type": "post-hoc design feasibility; not Step 3 or new-site results",
              "input_sha256": INPUT_HASHES, "inventories": inventories, "summary": summary,
              "selector_check": {"n_cells": len(grid), "mismatches": selector_errors},
              "nested_flag_sets_check": {"n_bundles": len(directories), "thresholds": ts, "violations": nesting_errors},
              "historical_aggregate_success_monotone": all(sweep["sweep"][str(b)]["combined_success"] <= sweep["sweep"][str(a)]["combined_success"] for a, b in zip(ts, ts[1:])),
              "historical_sweep": sweep["sweep"], "old_subgroup_metrics": subgroups,
              "exploratory_clean_only_folds": folds,
              "limitations": ["No new-site outcome evaluated or sampled.", "No per-cell repair success recomputation; nested-set check plus source-level coverage proof only.", "Exploratory folds do not test equality to supervised tie-breaking and are not a frozen new validation experiment.", "Aggregate subgroup numbers are rounded source values; no inferential test performed."]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if OUT.exists() and OUT.read_text() != payload:
        raise RuntimeError("Existing audit differs; choose a new version instead of overwriting.")
    OUT.write_text(payload)
    print(json.dumps({"output": str(OUT), "summary": summary, "selector_mismatches": len(selector_errors),
                      "nesting_violations": len(nesting_errors), "old_subgroups": subgroups,
                      "folds": {a: {"threshold_counts": dict(Counter(str(x['threshold']) for x in rr)),
                                     "held_out_exceedances": sum(x['held_out_exceeds_alpha'] is True for x in rr),
                                     "held_out_mean_alert_rate": st.mean(x['held_out_alert_rate'] for x in rr if x['threshold'] is not None)} for a, rr in folds.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
