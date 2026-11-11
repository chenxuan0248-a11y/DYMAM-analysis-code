from __future__ import annotations
import hashlib, json, math, random, statistics as st, sys
import bisect
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
D=ROOT/'experiments/exp_arcs'; REV=D/'revision_v2'; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from experiments.exp_arcs.detector_eval import detector_localize, manifest_dropped_hosts
from experiments.exp_arcs.detector_pr_roc import _host_scores
from experiments.exp_arcs.localizer_baselines import _truth
from experiments.exp_arcs.rq2_combined_topk import tie_aware_hit
from experiments.exp_arcs.contamination import contamination_closure
from experiments.exp_arcs.dag import build_dag
from experiments.exp_arcs.select import evaluate_scope
from experiments.exp_arcs.revision_v2.repair_plan_no_clean_run import contract as get_contract
from experiments.exp_arcs.revision_v2 import step2_baselines as s2

GRID=D/'e3_grid_v1.json'; SPEC=D/'cost_harm_spec.yaml'; PROTOCOLS=[OUT/'BROADER_CANDIDATE_SENSITIVITY_PROTOCOL.md',OUT/'UNIFIED_ALERT_BUDGET_PROTOCOL.md']
BUDGET=.1055; B=10000; SEED=42
FAMS=('dns','certificate','active_observation','registration','other')
ACTION_FAMS=('dns','certificate','registration','active_observation')
CTYPES=s2.CTYPES; DECS=s2.DECS; FEATURES=s2.FEATURES

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def getsets(bundle):
    b=Path(bundle); ev=lines(b/'evidence_event.jsonl'); claims=lines(b/'claim.jsonl'); decisions=lines(b/'decision.jsonl')
    e={x['target'] for x in ev}; m=json.loads((b/'discovery_manifest.json').read_text()) if (b/'discovery_manifest.json').exists() else {}
    r=set(m.get('resolved_hosts',[])); c={x.get('subject') for x in claims if x.get('subject')}
    cm={x['claim_id']:x.get('subject') for x in claims}
    c |= {cm[x.get('claim_id')] for x in decisions if x.get('claim_id') in cm and cm[x.get('claim_id')]}
    return ev,claims,decisions,e,r,c

def candidate(bundle, kind):
    ev,cl,de,e,r,c=getsets(bundle)
    return (e|r) if kind=='main' else (e|r|c)

def feature_matrix(bundle, kind):
    ev,claims,decisions,e,r,c=getsets(bundle)
    hs=tuple(sorted((e|r) if kind=='main' else (e|r|c))); ix={h:i for i,h in enumerate(hs)}
    X=np.zeros((len(hs),len(FEATURES))); col={x:i for i,x in enumerate(FEATURES)}
    for v in ev:
        fam=v.get('source_family') if v.get('source_family') in FAMS else 'other'
        X[ix[v['target']],col[f'E_{fam}']]+=1
    claimmap={q['claim_id']:q for q in claims}
    for q in claims:
        h=q.get('subject'); t=q.get('claim_type')
        if h in ix and t in CTYPES: X[ix[h],col[f'C_{t}']]+=1
    for d in decisions:
        q=claimmap.get(d.get('claim_id'),{}); h=q.get('subject')
        if h not in ix: continue
        dv=d.get('decision') if d.get('decision') in DECS else 'reject'
        X[ix[h],col[f'D_{dv}']]+=1
        if 'no_supporting_observation' in (d.get('reason_codes') or []): X[ix[h],col['N_no_supporting_observation']]+=1
    for f in FAMS: X[:,col[f'P_{f}']]=(X[:,col[f'E_{f}']]>0).astype(float)
    logcols=[j for j,f in enumerate(FEATURES) if not f.startswith('P_')]; X[:,logcols]=np.log1p(X[:,logcols])
    return hs,X

@lru_cache(maxsize=None)
def _score_cached(bundle,kind,method):
    hs,X=feature_matrix(bundle,kind)
    if method=='combined':
        v=_host_scores(Path(bundle)); return {h:float(v.get(h,0.)) for h in hs}
    if method=='contract_b':
        ev,_,_,_,_,_=getsets(bundle); fams=defaultdict(set)
        for item in ev: fams[item['target']].add(item.get('source_family','other'))
        return {h:float(not ('dns' in fams.get(h,()) and 'active_observation' in fams.get(h,()))) for h in hs}
    if len(hs)<3: return {h:0. for h in hs}
    vals=s2.DET_FN[method](X)
    return dict(zip(hs,map(float,vals)))

def score(bundle,kind,method):
    return _score_cached(str(Path(bundle)),kind,method)

def calibrate(runs,kind,method):
    # A stable host-hash breaks exact score ties, allowing the realized clean
    # macro rate to approach the budget without consulting fault labels.
    def key(h,v):
        hv=int(hashlib.sha256(('alert-budget-v1|'+h).encode()).hexdigest()[:16],16)/2**64
        return (float(v),hv)
    run_keys=[sorted(key(h,v) for h,v in score(run,kind,method).items()) for run in runs]
    candidates=sorted({k for ks in run_keys for k in ks})
    candidates.append((float('inf'),1.0))
    def rate(cut): return st.mean((len(ks)-bisect.bisect_left(ks,cut))/len(ks) for ks in run_keys)
    # Pick the attainable rate closest to the target; ties prefer the lower rate.
    cut=min(candidates,key=lambda t:(abs(rate(t)-BUDGET),rate(t)))
    return [cut[0],cut[1]]

def alerted_at(scores,cut):
    tau,hashcut=cut
    def tie(h): return int(hashlib.sha256(('alert-budget-v1|'+h).encode()).hexdigest()[:16],16)/2**64
    return {h for h,v in scores.items() if (float(v),tie(h)) >= (tau,hashcut)}

def paired_boot(values):
    rng=random.Random(SEED); n=len(values); draws=sorted(st.mean(rng.choice(values) for _ in range(n)) for _ in range(B))
    return [round(draws[int(.025*B)-1],4),round(draws[int(.975*B)-1],4)]

def planner(bundle,alerted,contract):
    ev,claims,decisions,e,r,c=getsets(bundle); fs=defaultdict(set); fh=defaultdict(set)
    for x in ev:
        f=x.get('source_family','other'); fs[x['target']].add(f); fh[f].add(x['target'])
    evidence_hosts=set(fs); rates={f:len(v)/len(evidence_hosts) for f,v in fh.items()} if evidence_hosts else {}
    clmap={x['claim_id']:x.get('subject') for x in claims}; self_flag=set()
    for d in decisions:
        if d.get('decision')=='revalidate' or 'no_supporting_observation' in (d.get('reason_codes') or []):
            if clmap.get(d.get('claim_id')): self_flag.add(clmap[d['claim_id']])
    plans={}
    for pol in ('selective','fixed_four_family'):
        actions=[]
        for h in sorted(alerted):
            if pol=='fixed_four_family' or h not in evidence_hosts or h in self_flag: families=ACTION_FAMS
            else:
                miss=[f for f in ACTION_FAMS if f not in fs.get(h,()) and rates.get(f,0)>=.5]
                families=miss or ACTION_FAMS
            for f in families: actions.append((h,f))
        plans[pol]={'actions':actions,'cost':sum(contract['cost_by_family'][f] for h,f in actions)}
    return plans

def input_hashes():
    return {'grid':sha(GRID),'step2_code':sha(REV/'step2_baselines.py'),'cost_spec':sha(SPEC),
            'broader_protocol':sha(PROTOCOLS[0]),'budget_protocol':sha(PROTOCOLS[1]),'runner':sha(Path(__file__))}

def run():
    grid=json.loads(GRID.read_text())['cells']; assert len(grid)==180 and len({c['org'] for c in grid})==20
    org_clean={c['org']:str(ROOT/c['clean_dir']) for c in grid}
    orgs=sorted(org_clean); contract=get_contract(); methods=('combined','iforest','robust_z','contract_b')
    # LOOCV thresholds, independently calibrated for each contract and detector.
    thresholds={}
    for kind in ('main','broad'):
        thresholds[kind]={}
        for method in methods[:3]:
            thresholds[kind][method]={}
            for org in orgs:
                train=[p for o,p in org_clean.items() if o!=org]
                thresholds[kind][method][org]=calibrate(train,kind,method)
    # Plan phase: scores/alerts and plans serialized without clean bundle access.
    planned=[]
    clean_alerts={}
    for kind in ('main','broad'):
        clean_alerts[kind]={}
        for method in methods:
            clean_alerts[kind][method]={}
            for org,p in org_clean.items():
                sc=score(p,kind,method)
                if method=='contract_b':
                    ev,cl,de,e,r,c=getsets(p); fs=defaultdict(set)
                    for x in ev: fs[x['target']].add(x.get('source_family','other'))
                    fl={h for h in candidate(p,kind) if not ('dns' in fs.get(h,()) and 'active_observation' in fs.get(h,()))}
                else: fl=alerted_at(sc,thresholds[kind][method][org])
                clean_alerts[kind][method][org]={'alerted':len(fl),'candidates':len(sc)}
        for cell in grid:
            b=ROOT/cell['faulted_dir']; cand=candidate(b,kind); truth,_=_truth(cell['inject_node'])
            for method in methods:
                sc=score(b,kind,method)
                if method=='contract_b':
                    ev,cl,de,e,r,c=getsets(b); fs=defaultdict(set)
                    for x in ev: fs[x['target']].add(x.get('source_family','other'))
                    fl={h for h in cand if not ('dns' in fs.get(h,()) and 'active_observation' in fs.get(h,()))}
                else: fl=alerted_at(sc,thresholds[kind][method][cell['org']])
                plans=planner(b,fl,contract)
                planned.append({'kind':kind,'method':method,'org':cell['org'],'cell':cell['cell'],'faulted_dir':cell['faulted_dir'],
                    'candidate_n':len(cand),'truth':sorted(truth),'scores':sc,'alerted':sorted(fl),
                    'detect1':tie_aware_hit(sc,truth,1),'truth_recall':len(truth&fl)/len(truth) if truth else 0.,
                    'plans':plans,'plan_hash':hashlib.sha256(json.dumps(plans,sort_keys=True,separators=(',',':')).encode()).hexdigest()})
    plan_path=OUT/'sensitivity_plans_local.json'
    plan_doc={'phase':'plans_only_before_clean_dag_evaluation','posthoc':True,'budget':BUDGET,'thresholds':thresholds,
              'clean_alerts':clean_alerts,'hashes':input_hashes(),'rows':planned}
    plan_path.write_text(json.dumps(plan_doc,sort_keys=True,indent=1)+'\n')
    # Evaluation phase begins only after plan file is written and reloaded.
    plan_doc=json.loads(plan_path.read_text()); assert plan_doc['hashes']==input_hashes()
    rows=[]
    cache={}
    for p in plan_doc['rows']:
        key=(p['org'],p['cell'])
        cell=next(x for x in grid if (x['org'],x['cell'])==key)
        clean=ROOT/cell['clean_dir']; fault=ROOT/cell['faulted_dir']
        if key not in cache:
            dag=build_dag(clean); dfx=contamination_closure(clean,fault)
            cache[key]=(dag,dfx)
        dag,dfx=cache[key]
        res={**{k:p[k] for k in ('kind','method','org','cell','candidate_n','detect1','truth_recall')},
             'candidate_target':bool(set(p['truth']) & set(p['scores'])),'policies':{}}
        for pol,plan in p['plans'].items():
            pairs=set(map(tuple,plan['actions']))
            scope={n for n,v in dag.nodes.items() if v.get('repairable') and (v['target'],v['source_family']) in pairs}
            ev=evaluate_scope(clean,dfx,scope,dag=dag)
            res['policies'][pol]={'success':bool(ev['success']),'cost':plan['cost'],'actions':len(pairs)}
        rows.append(res)
    # summaries per contract/method, organization macro; candidate coverage counts target pairs
    summary={}
    for kind in ('main','broad'):
      summary[kind]={}
      for method in methods:
        selected=[]
        for cell in grid:
            pr=next(p for p in planned if p['kind']==kind and p['method']==method and p['org']==cell['org'] and p['cell']==cell['cell'])
            evrow=next(x for x in rows if x['kind']==kind and x['method']==method and x['org']==cell['org'] and x['cell']==cell['cell'])
            selected.append((pr,evrow))
        byorg=defaultdict(list)
        for pr,evrow in selected: byorg[pr['org']].append((pr,evrow))
        cr=clean_alerts[kind][method]
        cleanmacro=st.mean(v['alerted']/v['candidates'] for v in cr.values())
        entry={'target_candidate_pairs':sum(len(set(pr['truth']) & set(pr['scores'])) for pr,_ in selected),
               'detect1_org_macro':round(st.mean(st.mean(evrow['detect1'] for _,evrow in vv) for vv in byorg.values()),4),
               'truth_alert_recall_org_macro':round(st.mean(st.mean(pr['truth_recall'] for pr,_ in vv) for vv in byorg.values()),4),
               'clean_alert_macro':round(cleanmacro,4),
               'mean_fault_alerts_per_run':round(st.mean(len(pr['alerted']) for pr,_ in selected),4),
               'policies':{}}
        for pol in ('selective','fixed_four_family'):
            succorg=[st.mean(float(evrow['policies'][pol]['success']) for _,evrow in vv) for vv in byorg.values()]
            costorg=[st.mean(evrow['policies'][pol]['cost'] for _,evrow in vv) for vv in byorg.values()]
            actorg=[st.mean(evrow['policies'][pol]['actions'] for _,evrow in vv) for vv in byorg.values()]
            entry['policies'][pol]={'OPR_cells':sum(evrow['policies'][pol]['success'] for _,evrow in selected),
                'OPR_org_macro':round(st.mean(succorg),4),'mean_cost_org_macro':round(st.mean(costorg),4),
                'mean_actions_org_macro':round(st.mean(actorg),4)}
        summary[kind][method]=entry
    # paired contract effects by detector; resampling organizations
    paired={}
    for method in methods:
      paired[method]={}
      for field in ('detect1','truth_recall'):
        diffs=[]
        for org in orgs:
          a=[];b=[]
          for cell in grid:
            if cell['org']!=org: continue
            pmain=next(p for p in planned if p['kind']=='main' and p['method']==method and p['org']==org and p['cell']==cell['cell'])
            pbroad=next(p for p in planned if p['kind']=='broad' and p['method']==method and p['org']==org and p['cell']==cell['cell'])
            a.append(pmain[field if field!='detect1' else 'detect1']); b.append(pbroad[field if field!='detect1' else 'detect1'])
          diffs.append(st.mean(b)-st.mean(a))
        paired[method][field+'_broad_minus_main']={'org_mean':round(st.mean(diffs),4),'ci95':paired_boot(diffs)}
      for pol in ('selective','fixed_four_family'):
        paired[method][pol]={}
        for field in ('success','cost'):
          diffs=[]
          for org in orgs:
            a=[];b=[]
            for cell in grid:
              if cell['org']!=org: continue
              ra=next(x for x in rows if x['kind']=='main' and x['method']==method and x['org']==org and x['cell']==cell['cell'])
              rb=next(x for x in rows if x['kind']=='broad' and x['method']==method and x['org']==org and x['cell']==cell['cell'])
              a.append(float(ra['policies'][pol][field]) if field=='success' else ra['policies'][pol][field])
              b.append(float(rb['policies'][pol][field]) if field=='success' else rb['policies'][pol][field])
            diffs.append(st.mean(b)-st.mean(a))
          paired[method][pol][field+'_broad_minus_main']={'org_mean':round(st.mean(diffs),4),'ci95':paired_boot(diffs)}
    # Unified-budget detector paired comparisons on the main contract.
    budget_pairs={}
    for method in ('combined','iforest','robust_z'):
      budget_pairs[method]={}
      for other in ('combined','iforest','robust_z'):
       if method==other: continue
       budget_pairs[method]['minus_'+other]={}
       for field in ('detect1','truth_recall'):
        diffs=[]
        for org in orgs:
         aa=[r[field] for r in rows if r['kind']=='main' and r['method']==method and r['org']==org]
         bb=[r[field] for r in rows if r['kind']=='main' and r['method']==other and r['org']==org]
         diffs.append(st.mean(aa)-st.mean(bb))
        budget_pairs[method]['minus_'+other][field]={'org_mean_difference':round(st.mean(diffs),4),'ci95':paired_boot(diffs)}
       for pol in ('selective','fixed_four_family'):
        budget_pairs[method]['minus_'+other][pol]={}
        for field in ('success','cost'):
         diffs=[]
         for org in orgs:
          aa=[(float(r['policies'][pol][field]) if field=='success' else r['policies'][pol][field]) for r in rows if r['kind']=='main' and r['method']==method and r['org']==org]
          bb=[(float(r['policies'][pol][field]) if field=='success' else r['policies'][pol][field]) for r in rows if r['kind']=='main' and r['method']==other and r['org']==org]
          diffs.append(st.mean(aa)-st.mean(bb))
         budget_pairs[method]['minus_'+other][pol][field]={'org_mean_difference':round(st.mean(diffs),4),'ci95':paired_boot(diffs)}
       
    output={'status':'completed_local_posthoc_exploratory','budget_macro_clean_candidate_alert_rate':BUDGET,
        'thresholds':thresholds,'summary':summary,'candidate_contract_paired_differences':paired,
        'unified_budget_detector_paired_differences':budget_pairs,
        'per_cell':rows,'hashes':input_hashes(),'plans_file_sha256':sha(plan_path),
        'n_cells':180,'n_orgs':20,'resampling_unit':'organization','bootstrap':{'B':B,'seed':SEED,'CI':'percentile 95%'},
              'method_axis_note':'Per-cell rows include explicit candidate contract and detector fields.'}
    (OUT/'sensitivity_results.json').write_text(json.dumps(output,sort_keys=True,indent=1)+'\n')
    print(json.dumps({'output':str(OUT/'sensitivity_results.json'),'summary':summary,'hashes':input_hashes()},indent=2))

if __name__=='__main__': run()
