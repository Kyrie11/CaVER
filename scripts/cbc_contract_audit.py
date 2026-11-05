#!/usr/bin/env python3
"""Fail-fast audit of frozen CBC record/checkpoint contracts."""
import argparse, gzip, json, hashlib, re
from pathlib import Path

def rows(path):
    op=gzip.open if str(path).endswith('.gz') else open
    with op(path,'rt',encoding='utf8') as f:
        return [json.loads(x) for x in f if x.strip()]

def seeds(r):
    out=set()
    for a in r.get('actions',[]):
        for key in ('source_by_action','reference_by_action'):
            for z in r.get(key,{}).get(a.get('action_id'),[]):
                if 'world_seed' in z: out.add((z['world_seed'],z.get('family')))
    # raw records intentionally omit rollout arrays; use declared deterministic ids
    if not out:
        seed=r.get('scene',{}).get('provenance',{}).get('seed')
        if seed is not None: out.add((seed,'scene'))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--test',required=True)
    p.add_argument('--calibration');p.add_argument('--checkpoint',action='append',default=[])
    p.add_argument('--expected-train',type=int);p.add_argument('--expected-test',type=int);p.add_argument('--expected-calibration',type=int)
    p.add_argument('--expected-source-worlds',type=int);p.add_argument('--expected-audit-worlds',type=int)
    p.add_argument('--output',required=True);a=p.parse_args()
    paths={'train':a.train,'test':a.test};
    if a.calibration: paths['calibration']=a.calibration
    result={'schema':'cbc_contract_audit_v1','files':{},'errors':[],'warnings':[]}
    sets={}
    for label,path in paths.items():
        if not Path(path).is_file(): result['errors'].append(f'MISSING_{label.upper()}:{path}'); continue
        rr=rows(path);sets[label]=rr
        splits={r.get('experimental_split') for r in rr}
        worlds={(r.get('source_worlds'),r.get('audit_worlds_per_family'),tuple(r.get('audit_families',[]))) for r in rr}
        result['files'][label]={'path':str(path),'records':len(rr),'splits':sorted(map(str,splits)),
          'worlds_families':[list(x[:2])+[list(x[2])] for x in sorted(worlds,key=str)],
          'scene_id_sha256':hashlib.sha256('\n'.join(sorted(r.get('scene_id','') for r in rr)).encode()).hexdigest()}
        expected={'train':a.expected_train,'test':a.expected_test,'calibration':a.expected_calibration}[label]
        if expected is not None and len(rr)!=expected: result['errors'].append(f'{label}_COUNT:{len(rr)}!={expected}')
        if any(r.get('experimental_split')!=label for r in rr): result['errors'].append(f'{label}_SPLIT_MISMATCH')
        if len({r.get('scene_id') for r in rr})!=len(rr): result['errors'].append(f'{label}_DUPLICATE_SCENE_ID')
        if len(worlds)>1: result['errors'].append(f'{label}_INCONSISTENT_WORLDS_OR_FAMILIES')
        if worlds and a.expected_source_worlds is not None and next(iter(worlds))[0]!=a.expected_source_worlds: result['errors'].append(f'{label}_SOURCE_WORLDS')
        if worlds and a.expected_audit_worlds is not None and next(iter(worlds))[1]!=a.expected_audit_worlds: result['errors'].append(f'{label}_AUDIT_WORLDS')
    # Synthetic scene IDs encode the deterministic seed; report any seed overlap
    # even when a caller changes the textual scene prefix. Native records fall
    # back to scene-ID/group checks because raw rollout seeds are intentionally
    # not serialized.
    seed_sets={label:{m.group(1) for r in rr for m in [re.search(r'(?:synth|synthetic)-?(?:stress-)?(\d+)$', str(r.get('scene_id','')))] if m} for label,rr in sets.items()}
    result['seed_overlap']={}
    for x in seed_sets:
        for y in seed_sets:
            if x<y and seed_sets[x] & seed_sets[y]:
                ov=sorted(seed_sets[x] & seed_sets[y]); result['seed_overlap'][f'{x}-{y}']=ov
                result['errors'].append(f'SEED_OVERLAP:{x}:{y}:{len(ov)}')
    for x in sets:
        for y in sets:
            if x<y:
                ids={r.get('scene_id') for r in sets[x]} & {r.get('scene_id') for r in sets[y]}
                if ids: result['errors'].append(f'SCENE_OVERLAP:{x}:{y}:{len(ids)}')
    for cp in a.checkpoint:
        try:
            import torch
            ck=torch.load(cp,map_location='cpu',weights_only=True);meta=ck.get('meta',{})
            n=int(meta.get('training_scenes',-1)); expected=len(sets.get('train',[]))
            result.setdefault('checkpoints',[]).append({'path':cp,'training_scenes':n,'expected_train':expected,'variant':ck.get('variant')})
            if n!=expected: result['errors'].append(f'CHECKPOINT_TRAINING_COUNT:{cp}:{n}!={expected}')
        except Exception as e: result['errors'].append(f'CHECKPOINT_READ:{cp}:{e}')
    result['status']='PASS' if not result['errors'] else 'FAIL'
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2)); return 0 if not result['errors'] else 2
if __name__=='__main__': raise SystemExit(main())
