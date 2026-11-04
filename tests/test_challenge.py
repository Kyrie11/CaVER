from pathlib import Path
from caver.experiment import frozen_challenge

def test_frozen_posthoc_mechanism():
    root=Path(__file__).resolve().parents[1]
    out=frozen_challenge(root/'configs/fourcar_challenge.json')
    assert not out['actions']['early']['ego_collision']
    assert not out['actions']['late']['ego_collision']
    assert out['actions']['early']['BC_collision']
    assert not out['actions']['late']['BC_collision']
    assert not out['B_mediator_exchange']['a_with_b_mediator']['BC_contact']
    assert out['B_mediator_exchange']['b_with_a_mediator']['BC_contact']
    assert 'POSTHOC' in out['fixture']
    assert out['action_max_position_distance_m']<4.
