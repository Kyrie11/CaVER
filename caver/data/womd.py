"""WOMD v1.3.1 adapters with an enforceable planning/label boundary.

Input Scene = static map + observed tracks <= planning instant ONLY.
Future observations are obtained only through read(..., with_future=True) as a
separate offline label dictionary. Never supply label dictionaries to planners.
"""
from pathlib import Path
from collections import defaultdict
import hashlib
import numpy as np
from .schema import Agent, Scene

_SPLITS = {'training', 'validation', 'testing', 'training_20s',
           'validation_interactive', 'testing_interactive'}


def find_shards(root, fmt='scenario', split='training'):
    if fmt not in ('scenario', 'tf_example') or split not in _SPLITS:
        raise ValueError('Unknown WOMD format / split')
    folder = Path(root)/'uncompressed'/fmt/split
    files = sorted(p for p in folder.rglob('*') if p.is_file() and '.tfrecord' in p.name) if folder.exists() else []
    return folder, files


def inspect(root):
    return {fmt: {split: {'path': str(folder), 'shards': len(files),
                          'bytes': sum(f.stat().st_size for f in files),
                          'examples': [str(p) for p in files[:2]]}
                  for split in sorted(_SPLITS)
                  for folder, files in [find_shards(root, fmt, split)]}
            for fmt in ('scenario', 'tf_example')}


def _tfrecord(shards, limit=None):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError('WOMD decode requires TensorFlow and the matching waymo-open-dataset wheel') from exc
    total = 0
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    for sh in shards:
        comp = 'GZIP' if sh.name.endswith('.gz') else ''
        for raw in tf.data.TFRecordDataset([str(sh)], compression_type=comp):
            yield raw.numpy(), str(sh)
            total += 1
            if limit is not None and total >= limit:
                return


def _dims(length, width):
    L, W = float(length), float(width)
    return (L, W) if np.isfinite(L) and np.isfinite(W) and min(L, W) >= .1 else (4.5, 1.8)


def _group_signature(lines, scene_id):
    """Conservative *heuristic* for overlapping windows, not a verified log ID.

    Lanes from same region are allowed to merge into one holdout group. False
    merging reduces sample count but is safer than splitting identical windows.
    """
    lane_ids = sorted(set(x['feature_id'] for x in lines if x['kind']=='lane'))
    if not lane_ids:
        return 'UNKNOWN:' + str(scene_id)
    return 'MAP_LANE_SET:' + hashlib.sha256(','.join(map(str, lane_ids)).encode()).hexdigest()[:24]


def parse_scenario(raw, include_future=False):
    try:
        from waymo_open_dataset.protos import scenario_pb2
    except ImportError as exc:
        raise RuntimeError('Matching waymo-open-dataset Scenario protobuf required') from exc
    msg = scenario_pb2.Scenario.FromString(raw)
    idx = int(msg.current_time_index)
    total = len(msg.timestamps_seconds)
    if not 0 <= idx < total:
        raise ValueError('Invalid WOMD current time index')
    ego_idx = int(msg.sdc_track_index)
    if not 0 <= ego_idx < len(msg.tracks):
        raise ValueError('Invalid WOMD SDC track index')
    ego_state = msg.tracks[ego_idx].states[idx]
    if not ego_state.valid:
        raise ValueError('SDC absent at planning time')
    ego_id = str(msg.tracks[ego_idx].id)
    lo = max(0, idx - 10)
    agents, future = [], {}
    for tr in msg.tracks:
        if len(tr.states) <= idx:
            raise ValueError('Track shorter than current planning index')
        st = tr.states[idx]
        if int(tr.object_type) != 1 or not st.valid:
            continue
        # Vehicle positions with invalid history mask are ignored downstream;
        # do not interpret padded zeroes as actual locations.
        history = [[float(z.center_x) if z.valid else 0.,
                    float(z.center_y) if z.valid else 0.,
                    float(z.velocity_x) if z.valid else 0.,
                    float(z.velocity_y) if z.valid else 0.,
                    float(z.heading) if z.valid else 0., int(z.valid)]
                   for z in tr.states[lo:idx+1]]
        L, W = _dims(st.length, st.width)
        aid = str(tr.id)
        agents.append(Agent(aid, 'vehicle', float(st.center_x), float(st.center_y),
                            float(st.velocity_x), float(st.velocity_y), float(st.heading),
                            L, W, history=history))
        if include_future:
            future[aid] = [[float(z.center_x) if z.valid else 0.,
                            float(z.center_y) if z.valid else 0., int(z.valid)]
                           for z in tr.states[idx+1:]]
    x0, y0 = float(ego_state.center_x), float(ego_state.center_y)
    lines = []
    for mf in msg.map_features:
        kind, points = None, None
        for name in ('lane', 'road_line', 'road_edge'):
            if mf.HasField(name):
                kind, points = name, getattr(mf, name).polyline
                break
        if kind is not None:
            pts = [[float(p.x), float(p.y)] for p in points
                   if (p.x-x0)**2+(p.y-y0)**2 <= 100**2]
            if pts:
                lines.append({'kind': kind, 'feature_id': int(mf.id), 'xy': pts})
    # Dynamic traffic lights: observation slices ONLY, at or before idx.
    lights = []
    for it, state in enumerate(msg.dynamic_map_states[:idx+1]):
        lights.append({'step': it,
                       'lane_signals': [{'lane_id': int(t.lane), 'state': int(t.state)}
                                        for t in state.lane_states]})
    # Use the complete *static* map ID collection for a stable overlap signature;
    # clipping to ego location would split overlapping windows from one log.
    all_lanes = [{'kind':'lane','feature_id':int(mf.id)} for mf in msg.map_features if mf.HasField('lane')]
    sig = _group_signature(all_lanes, msg.scenario_id)
    sc = Scene(msg.scenario_id, 'WOMD_SCENARIO_PROTO', .1, ego_id, agents,
               {'planning_index': idx, 'history_first_index': lo,
                'observed_steps': idx-lo+1, 'source_format': 'scenario',
                'history_only': True, 'future_trajectory_exposed': False,
                'overlap_group': sig, 'overlap_group_quality': 'HEURISTIC_MAP_NOT_VERIFIED_PARENT'},
               map_data={'polylines': lines, 'observed_signals': lights}).validate()
    return sc, future


def _feature(ex, key, kind, required=True):
    raw = ex.features.feature
    if key not in raw:
        if required:
            raise ValueError('Missing WOMD tf.Example feature: ' + key)
        return []
    value = raw[key]
    # No fallback to a list of the wrong protobuf kind (silently all-zero).
    if not value.HasField(kind):
        raise ValueError(f'WOMD feature {key} is not {kind}')
    return list(getattr(value, kind).value)


def parse_tf_example(raw, include_future=False):
    try:
        from tensorflow.train import Example
    except ImportError as exc:
        raise RuntimeError('TensorFlow required for WOMD tf.Example') from exc
    ex = Example.FromString(raw)
    field = lambda k: _feature(ex, 'state/current/'+k, 'int64_list' if k == 'valid' else 'float_list')
    arr = {k: field(k) for k in ('x','y','velocity_x','velocity_y','bbox_yaw','length','width','valid')}
    n = len(arr['x'])
    if not n or any(len(v) != n for v in arr.values()):
        raise ValueError('Current state tensor shape inconsistent')
    # Official WOMD spec: state/type and state/id are FLOAT arrays, not int64.
    types = _feature(ex, 'state/type', 'float_list')
    ids = _feature(ex, 'state/id', 'float_list')
    is_sdc = _feature(ex, 'state/is_sdc', 'int64_list')
    if any(len(v) != n for v in (types, ids, is_sdc)):
        raise ValueError('WOMD object metadata shape inconsistent')
    ego_indices = [i for i in range(n) if is_sdc[i] == 1]
    if len(ego_indices) != 1 or not arr['valid'][ego_indices[0]] or int(types[ego_indices[0]]) != 1:
        raise ValueError('WOMD missing valid vehicle SDC')
    keyid = lambda i: str(int(ids[i]))
    seen = set()
    valid_ids = [keyid(i) for i in range(n) if arr['valid'][i] and int(types[i]) == 1]
    if len(valid_ids) != len(set(valid_ids)):
        raise ValueError('Object IDs collided after WOMD float conversion')
    sidraw = _feature(ex, 'scenario/id', 'bytes_list')
    if len(sidraw) != 1:
        raise ValueError('WOMD scenario/id missing or ambiguous')
    sid = sidraw[0].decode('utf-8')
    histories = {}
    past = {}
    for key in ('x', 'y', 'velocity_x', 'velocity_y', 'bbox_yaw', 'valid'):
        kind = 'int64_list' if key == 'valid' else 'float_list'
        v = _feature(ex, 'state/past/' + key, kind)
        if len(v) % n:
            raise ValueError('WOMD past tensor shape incorrect: '+key)
        past[key] = np.asarray(v).reshape(n, -1)
    hlen = past['x'].shape[1]
    if hlen < 1 or any(x.shape != (n, hlen) for x in past.values()):
        raise ValueError('WOMD historical tensors disagree on time axis')
    agents = []
    for i in range(n):
        if not arr['valid'][i] or int(types[i]) != 1:
            continue
        hist = [[float(past['x'][i, t]) if past['valid'][i, t] else 0.,
                 float(past['y'][i, t]) if past['valid'][i, t] else 0.,
                 float(past['velocity_x'][i, t]) if past['valid'][i, t] else 0.,
                 float(past['velocity_y'][i, t]) if past['valid'][i, t] else 0.,
                 float(past['bbox_yaw'][i, t]) if past['valid'][i, t] else 0.,
                 int(past['valid'][i,t])]
                for t in range(hlen)]
        hist += [[float(arr['x'][i]),float(arr['y'][i]),float(arr['velocity_x'][i]),
                  float(arr['velocity_y'][i]),float(arr['bbox_yaw'][i]),1]]
        L, W = _dims(arr['length'][i], arr['width'][i])
        agents.append(Agent(keyid(i), 'vehicle', float(arr['x'][i]),float(arr['y'][i]),
                            float(arr['velocity_x'][i]),float(arr['velocity_y'][i]),
                            float(arr['bbox_yaw'][i]), L, W, history=hist))
    # tf.Example has sampled map points, not vector topology.
    map_data = {'samples': []}
    xyz = _feature(ex, 'roadgraph_samples/xyz', 'float_list', required=False)
    if xyz:
        rg_valid = _feature(ex, 'roadgraph_samples/valid', 'int64_list')
        rg_ids = _feature(ex, 'roadgraph_samples/id', 'int64_list')
        rg_type = _feature(ex, 'roadgraph_samples/type', 'int64_list')
        if len(xyz) != 3*len(rg_valid) or len(rg_ids)!=len(rg_valid) or len(rg_type)!=len(rg_valid):
            raise ValueError('WOMD roadgraph shapes inconsistent')
        x0, y0 = arr['x'][ego_indices[0]], arr['y'][ego_indices[0]]
        for i, ok in enumerate(rg_valid):
            x,y = xyz[3*i:3*i+2]
            if ok and (x-x0)**2+(y-y0)**2 <= 100**2:
                map_data['samples'].append({'xy':[float(x),float(y)],
                                            'feature_id':int(rg_ids[i]),'type':int(rg_type[i])})
    # Current and past signals only; never access future in planning input.
    map_data['observed_signals'] = []
    for phase in ('past', 'current'):
        v = _feature(ex, f'traffic_light_state/{phase}/state','int64_list',required=False)
        if v:
            map_data['observed_signals'].append({'phase': phase, 'states':list(map(int,v))})
    rgids = sorted(set(map(int,rg_ids))) if xyz else []
    group = ('MAP_SAMPLE_IDS:' + hashlib.sha256(','.join(map(str,rgids)).encode()).hexdigest()[:24]) if rgids else 'UNKNOWN:'+sid
    sc = Scene(sid, 'WOMD_TF_EXAMPLE', .1, keyid(ego_indices[0]), agents,
               {'planning_index': hlen, 'observed_steps': hlen+1,
                'history_only': True,'future_trajectory_exposed':False,
                'source_format':'tf_example', 'overlap_group':group,
                'overlap_group_quality':'HEURISTIC_MAP_NOT_VERIFIED_PARENT'}, map_data).validate()
    future = {}
    if include_future:
        fut = {}
        for k in ('x','y','valid'):
            kind = 'int64_list' if k=='valid' else 'float_list'
            v = _feature(ex, 'state/future/'+k, kind, required=False)
            if v and len(v)%n:
                raise ValueError('Future label tensor incompatible with object dimension')
            fut[k] = np.asarray(v).reshape(n,-1) if v else np.empty((n,0))
        if len({x.shape for x in fut.values()})!=1:
            raise ValueError('Future label shapes inconsistent')
        for i in range(n):
            if arr['valid'][i] and int(types[i])==1:
                future[keyid(i)] = [[float(fut['x'][i,t]) if fut['valid'][i,t] else 0.,
                                      float(fut['y'][i,t]) if fut['valid'][i,t] else 0.,
                                      int(fut['valid'][i,t])]
                                     for t in range(fut['x'].shape[1])]
    return sc, future


def read(root, format='scenario', split='training', limit=None, with_future=False):
    folder, shards = find_shards(root, format, split)
    if not shards:
        raise FileNotFoundError(f'No WOMD uncompressed {format}/{split} TFRecord shards in {folder}')
    parse = parse_scenario if format=='scenario' else parse_tf_example
    ids = set()
    for raw, shard in _tfrecord(shards, limit):
        sc, labels = parse(raw, include_future=with_future)
        if sc.scene_id in ids:
            raise ValueError('Duplicate WOMD scenario ID; data mixed or duplicated: '+sc.scene_id)
        ids.add(sc.scene_id)
        sc.provenance.update({'split': split, 'shard': shard,
                              'counterfactual_labels': 'NO_UNOBSERVED_ACTION_OUTCOMES'})
        yield (sc, labels) if with_future else sc


def export(root, format, split, out, limit):
    from .schema import write_jsonl
    scenes = list(read(root, format, split, limit, with_future=False))
    write_jsonl(out, scenes)
    return {'source': 'WOMD', 'format': format,'split': split,
            'planning_only': True, 'decoded_scenes':len(scenes),
            'unique_ids':len(set(s.scene_id for s in scenes)), 'out':str(out),
            'future_labels_exported':False}
