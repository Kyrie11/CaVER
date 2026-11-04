"""Read-only ReactSim token cross-check adapted from the audited RIDE token protocol.

A raw lidar_pc token existing in DB is NOT evidence of benchmark scenario
eligibility, actor coverage, or an executed reactive response.
"""
import re,sqlite3
from pathlib import Path

_HEX=re.compile(r'^[0-9a-fA-F]{2,64}$')

def check_token(db,token):
    if len(token)%2 or not _HEX.fullmatch(token):raise ValueError('Invalid nuPlan lidar_pc hex token')
    d=Path(db).resolve()
    with sqlite3.connect(d.as_uri()+'?mode=ro',uri=True) as conn:
        columns={x[1] for x in conn.execute('PRAGMA table_info(lidar_pc)')}
        if not {'token','timestamp','scene_token'}.issubset(columns):
            raise ValueError('DB does not have expected nuPlan lidar_pc schema')
        row=conn.execute('SELECT timestamp,scene_token FROM lidar_pc WHERE token=?',(bytes.fromhex(token),)).fetchone()
        if row is None:return {'token':token,'status':'RAW_TOKEN_ABSENT'}
        if row[1] is None:return {'token':token,'status':'MISSING_SCENE_LINK'}
        scene=conn.execute('SELECT token FROM scene WHERE token=?',(row[1],)).fetchone()
        return {'token':token,'status':'RAW_JOIN_PRESENT' if scene else 'MISSING_SCENE_ROW',
                'timestamp_us':row[0],'benchmark_eligibility':'NOT_ESTABLISHED'}
