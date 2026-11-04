"""ReactSim-Bench custom_behaviors PKL inventory; unpickling is intentionally avoided.

Official preprocessor + checkpoints needed for real reactive evaluation. A
strict, externally generated JSONL rollout format is the supported exchange.
"""
from pathlib import Path
import pickletools
import json

def inventory(path, repository=None, nuplan_maps=None):
    root=Path(path)
    files=sorted(root.glob('*.pkl')) if root.exists() else []
    repo=Path(repository) if repository else root.parent.parent/'repo'
    base=root.parent.parent
    weights=base/'ckpts'
    anchor_candidates=[repo/'data/motion_anchors',base/'data/motion_anchors']
    preprocessing_candidates=[repo/'data/nuplan_processed', repo/'data/test/nuplan_processed',base/'data/nuplan_processed']
    return {'custom_behaviors':str(root),'exists':root.exists(),'files':len(files),
            'examples':[x.name for x in files[:5]],'usable_as_reactive_rollouts':False,
            'repository':str(repo),'repository_exists':(repo/'reactsim_data_pipeline').is_dir(),
            'checkpoint_directory':str(weights),'checkpoint_count':len(list(weights.glob('*.ckpt'))),
            'motion_anchors_present':any(p.exists() for p in anchor_candidates),
            'processed_scenarios_present':any(p.exists() for p in preprocessing_candidates),
            'nuplan_maps_present':bool(nuplan_maps and Path(nuplan_maps).exists()),
            'native_rollout_executed':False,
            'requires':'Official model-specific environment, preprocessing, checkpoint, controlled ego actions and *executed* reactive rollouts'}

def validate_export(path):
    required={'scene_id','action_id','reactor','scene_digest','action_digest','rollout_digest',
              'ego_collision','downstream_collision','progress','runtime_ms','physical_contact_source'}
    rows=[]; seen=set()
    for ln,line in enumerate(Path(path).read_text().splitlines(),1):
        if not line.strip():continue
        r=json.loads(line);missing=required-set(r)
        if missing:raise ValueError(f'Line {ln} missing {missing}')
        key=(r['scene_id'],r['action_id'],r['reactor'])
        if key in seen:raise ValueError('duplicate exported rollout '+str(key))
        seen.add(key)
        if r['physical_contact_source'] not in ('sim_collision','trajectory_swept_obb'):
            raise ValueError('Invalid physical_contact_source')
        if any(type(r[k])!=bool for k in ('ego_collision','downstream_collision')):
            raise ValueError('Collision fields must be boolean')
        rows.append(r)
    return rows
