"""Reproducible split audit with conservative overlapping-WOMD-window grouping."""
import hashlib
from collections import Counter


def source_group(scene):
    p = scene.provenance
    if scene.source.startswith('NUPLAN'):
        return 'NUPLAN_LOG:' + str(p.get('log_name', 'UNKNOWN'))
    if p.get('overlap_group'):
        return str(p['overlap_group'])
    if scene.source.startswith('SYNTHETIC'):
        return 'SYNTHETIC:' + scene.scene_id
    return None


def partition(scene, num_buckets=10, calibration_buckets=5):
    """Only for mutually exclusive subgroups of the *same official validation*.

    A map-based group is a conservative heuristic; an *exact* parent segment
    ID (unavailable in standard WOMD Scenario proto) would be stronger.
    """
    g = source_group(scene)
    if not g or 'UNKNOWN:' in g:
        raise ValueError(f'Unknown overlap group for scene {scene.scene_id}; cannot assert split independence')
    value = int(hashlib.sha256(g.encode()).hexdigest()[:16],16) % num_buckets
    return 'calibration' if value < calibration_buckets else 'test'



def training_partition(scene, num_buckets=10, calibration_buckets=2):
    """Reserve calibration from official WOMD training; final holdout=validation.

    Train vs calibration is conservative-map-group disjoint. This protects the
    final official validation split more strongly than dividing it in two;
    exact 20s parent-segment independence still requires external metadata.
    """
    g=source_group(scene)
    if not g or 'UNKNOWN:' in g:
        raise ValueError('Cannot assign training group without auditable source group')
    h=int(hashlib.sha256(g.encode()).hexdigest()[:16],16)%num_buckets
    return 'calibration' if h<calibration_buckets else 'train'

def audit_record_splits(parts):
    ids, groups, dup = {}, {}, {}
    for label, rows in parts.items():
        sid = [r['scene_id'] for r in rows]
        dup[label] = sorted(k for k,v in Counter(sid).items() if v>1)
        ids[label] = set(sid)
        groups[label] = {r.get('scene',{}).get('provenance',{}).get('overlap_group',
                           r.get('scene',{}).get('scene_id',r['scene_id'])) for r in rows}
    comparisons = [('train','calibration'),('train','test'),('calibration','test')]
    overlaps = {f'{a}-{b}':sorted(ids[a]&ids[b]) for a,b in comparisons}
    group_overlaps = {f'{a}-{b}':sorted(groups[a]&groups[b]) for a,b in comparisons}
    passed = not any(dup.values()) and not any(overlaps.values()) and not any(group_overlaps.values())
    uncertain = sum(1 for v in parts.values() for r in v
                    if r.get('scene',{}).get('provenance',{}).get('overlap_group_quality','').startswith('HEURISTIC'))
    return {'passed':passed,'split_sizes':{k:len(v) for k,v in parts.items()},
            'duplicate_scene_ids':dup,'scene_id_overlap':overlaps,
            'overlap_group_overlap':group_overlaps,
            'womd_heuristic_groups':uncertain,
            'guarantee':'Exact ID and heuristic map-group checks only; identical logs with changed map IDs remain unverified'}
