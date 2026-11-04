"""Lossless enough normalized kinematics; map and provenance remain separate."""
from dataclasses import dataclass, asdict
from typing import Any
import json
import numpy as np

@dataclass
class Agent:
    agent_id: str
    kind: str
    x: float
    y: float
    vx: float
    vy: float
    yaw: float
    length: float
    width: float
    history: list | None = None

@dataclass
class Scene:
    scene_id: str
    source: str
    dt: float
    ego_id: str
    agents: list[Agent]
    provenance: dict
    map_data: dict | None = None

    def validate(self):
        ids=[a.agent_id for a in self.agents]
        if self.ego_id not in ids or len(ids)!=len(set(ids)): raise ValueError('ego missing or duplicate agent ID')
        if self.dt<=0 or not np.isfinite(self.dt):raise ValueError('invalid dt')
        for a in self.agents:
            if not all(np.isfinite(z) for z in (a.x,a.y,a.vx,a.vy,a.yaw,a.length,a.width)) or min(a.length,a.width)<=0:
                raise ValueError('invalid dimensions or nonfinite state')
            if a.history is not None:
                for h in a.history:
                    if len(h)!=6 or h[5] not in (0,1) or not all(np.isfinite(z) for z in h):
                        raise ValueError('invalid or nonfinite six-field observation history')
                if a.history and a.history[-1][5] != 1:
                    raise ValueError('current observation must be present in last history frame')
                if a.history and (abs(a.history[-1][0]-a.x)>1e-2 or abs(a.history[-1][1]-a.y)>1e-2):
                    raise ValueError('last observed frame does not match planning state')
        return self

    def to_json(self):return asdict(self)

    @classmethod
    def from_json(cls, d):
        return cls(**{**d,'agents':[Agent(**a) for a in d['agents']]}).validate()

def write_jsonl(path, scenes):
    from pathlib import Path
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w') as f:
        for sc in scenes:f.write(json.dumps(sc.to_json(),separators=(',',':'))+'\n')

def read_jsonl(path,limit=None):
    with open(path) as f:
        for idx,line in enumerate(f):
            if limit is not None and idx>=limit:return
            if line.strip():yield Scene.from_json(json.loads(line))
