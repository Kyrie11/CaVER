"""Centralize server paths without packaging any private dataset."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass
class Paths:
    womd: str = '/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1'
    nuplan: str = '/data0/senzeyu2/dataset/nuplan/data/cache'
    reactsim: str = '/data0/senzeyu2/dataset/ReactSim-Bench/data/custom_behaviors'
    nuplan_maps: str = '/data0/senzeyu2/dataset/nuplan/maps'
    reactsim_repository: str = '/data0/senzeyu2/dataset/ReactSim-Bench/repo'
    carla_host: str = '127.0.0.1'
    carla_port: int = 2000

    @classmethod
    def load(cls, path=None):
        if path is None: return cls()
        d=json.loads(Path(path).read_text())
        return cls(**{key:d[key] for key in asdict(cls()) if key in d})
