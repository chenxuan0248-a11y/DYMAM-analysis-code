import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent
cells=json.loads((root/'experiments/exp_arcs/e3_grid_v1.json').read_text())['cells']
files=('source_artifact.jsonl','evidence_event.jsonl','claim.jsonl','decision.jsonl','report_item.jsonl','discovery_manifest.json')
seen={}
for c in cells:
  for role,key in [('clean','clean_dir'),('faulted','faulted_dir')]:
    rel=c[key]
    b=root/rel
    entry=seen.setdefault(rel,{'bundle':rel,'roles':[],'files':{}})
    if role not in entry['roles']: entry['roles'].append(role)
    for name in files:
      p=b/name
      entry['files'][name]=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
manifest={'scope':'offline inputs required for candidate features, planning, DAG evaluation, and contamination diff',
          'grid_sha256':hashlib.sha256((root/'experiments/exp_arcs/e3_grid_v1.json').read_bytes()).hexdigest(),
          'bundles':list(seen.values())}
(out/'bundle_input_hashes.json').write_text(json.dumps(manifest,sort_keys=True,indent=1)+'\n')
