"""Streaming, observation-only WOMD preprocessing for research pilots.

A source sample is accepted or rejected using observations at the planning
instant, never logged future. Source sampling is reproducible at the *shard*
level and is not claimed to be i.i.d. full-WOMD sampling.
"""
from collections import Counter
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import gzip
import json
import math
import os
import zlib

from .womd import read, find_shards
from .corridor import corridor_subset
from .splits import training_partition
from .schema import Scene


def _policy_view(scene, max_agents=24, radius=100.):
    ego = next(a for a in scene.agents if a.agent_id == scene.ego_id)
    nearby = sorted((a for a in scene.agents if a.agent_id != scene.ego_id),
                    key=lambda a: ((a.x-ego.x)**2+(a.y-ego.y)**2, a.agent_id))
    retained = [ego]+[a for a in nearby if math.hypot(a.x-ego.x,a.y-ego.y) <= radius][:max_agents-1]
    out = replace(scene, agents=retained, map_data=None,
                  provenance={**scene.provenance, 'export_profile':'OBSERVATION_ONLY_POLICY_PRETRAIN',
                              'agent_filter':'CURRENT_POSITION_WITHIN_100M_NEAREST_24',
                              'filtered_at_planning_instant':True,
                              'original_agent_count':len(scene.agents)})
    return out.validate()


def prepare_womd(root, split, partition, purpose, output, limit=100, max_scan=30000,
                 shard_seed=17, max_shards=100, per_shard=400, max_output_mb=45.,
                 fmt='scenario'):
    """Write a bounded-size compressed JSONL and manifest; never load all scenes.

    `corridor` is geometric and may miss intersections/adjacent lane topology;
    it is a *pilot subset*, not verified lawful candidate generation.
    """
    if fmt!='scenario':
        raise ValueError('Canonical preparation uses Scenario proto; tf.Example is a separate adapter')
    if purpose not in ('policy','corridor') or partition not in ('train','calibration','test'):
        raise ValueError('Invalid purpose/partition')
    if (split=='training') != (partition in ('train','calibration')):
        raise ValueError('training must be train/calibration; validation must be test')
    if purpose=='policy' and partition!='train':
        raise ValueError('Observed policy fitting must use train partition only')
    if limit < 1 or max_scan < 1 or max_output_mb <= 0:
        raise ValueError('Nonpositive limits')
    _, shards=find_shards(root,fmt,split)
    if not shards:
        raise FileNotFoundError(f'Missing WOMD Scenario {split} shards')
    dest=Path(output)
    if dest.suffix!='.gz' or not str(dest).endswith('.jsonl.gz'):
        raise ValueError('To guarantee shareable outputs, use <name>.jsonl.gz')
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_name(dest.name+'.part.gz')
    reasons=Counter(); kept=0; scanned=0; estimate_bytes=0; ids=set(); groups=set()
    max_budget=int(max_output_mb*1024*1024)
    finished='scan_limit'
    try:
        with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=6) as fh:
            for sc in read(root,format=fmt,split=split,limit=max_scan,with_future=False,
                           shard_seed=shard_seed,max_shards=max_shards,per_shard=per_shard):
                scanned+=1
                if split=='training':
                    try:assigned=training_partition(sc)
                    except ValueError:
                        reasons['UNKNOWN_SOURCE_GROUP']+=1
                        continue
                    if assigned!=partition:
                        reasons['NOT_REQUESTED_PARTITION']+=1
                        continue
                if purpose=='corridor':
                    result,reason=corridor_subset(sc)
                    if result is None:
                        reasons[reason]+=1
                        continue
                else:
                    result=_policy_view(sc)
                if sc.scene_id in ids:
                    raise ValueError('Duplicate scenario in prepared export')
                group=sc.provenance.get('overlap_group','')
                if not group or group.startswith('UNKNOWN:'):
                    reasons['UNKNOWN_OVERLAP_GROUP']+=1
                    continue
                # Check cannot accidentally serialize future/ground-truth into planner input.
                assert result.provenance.get('history_only') is True
                assert result.provenance.get('future_trajectory_exposed') is False
                result.provenance['preparation_partition']=partition
                result.provenance['preparation_purpose']=purpose
                line=json.dumps(result.to_json(),separators=(',',':'),allow_nan=False)+'\n'
                # Each record compressed separately is a conservative budget
                # proxy (one continuous gzip stream is typically smaller).
                upper=len(zlib.compress(line.encode(),6))+96
                if estimate_bytes+upper > max_budget:
                    reasons['OUTPUT_BUDGET_STOP']+=1
                    finished='size_budget'
                    break
                fh.write(line)
                estimate_bytes+=upper
                kept+=1;ids.add(sc.scene_id);groups.add(group)
                if kept>=limit:
                    finished='target_count'
                    break
            else:
                finished='exhausted_available_shards_or_scan_limit'
        if tmp.stat().st_size > max_budget:
            raise ValueError('Compressed output exceeds requested byte budget')
        os.replace(tmp,dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    digest=sha256(dest.read_bytes()).hexdigest()
    manifest={'format_version':'CAVER_WOMD_PREP_V1','source':'WOMD_SCENARIO_PROTO',
              'official_split':split,'partition':partition,'purpose':purpose,
              'future_labels_accessed':False,'future_trajectory_exposed':False,
              'selection':'DETERMINISTIC_SHARD_ORDER_FIRST_N_PER_SHARD_NOT_IID',
              'shard_seed':int(shard_seed),'max_shards':int(max_shards),
              'per_shard':int(per_shard),'max_scan':int(max_scan),'scanned':scanned,
              'selected':kept,'unique_map_groups':len(groups),'skip_reasons':dict(reasons),
              'stop_reason':finished,'requested_selected':limit,
              'file':str(dest),'file_bytes':dest.stat().st_size,'file_sha256':digest,
              'max_output_mb':max_output_mb,
              'limitations':['Map group is a HEURISTIC; not a verified original log identifier',
                             'Temporal/time-token and spatial map topology are not proven by corridor filter',
                             'Source shards and truncation are a sampled pilot not full-dataset prevalence']}
    mp=Path(str(dest)+'.manifest.json')
    mp.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    if kept==0:
        raise ValueError('No eligible scenes exported. See '+str(mp)+'; relax scan budget or inspect geometry.')
    return manifest
