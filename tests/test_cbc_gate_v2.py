import json
from pathlib import Path
from caver.validation.cbc_gate import summarize


def test_gate_explicitly_abstains_without_resolved_pairs(tmp_path):
    p=tmp_path/'study.json'
    p.write_text(json.dumps({'methods':{'source':{'matched_kinematic_proxy_scenes':80,
                                                  'critical_pair_ref_sign_resolved':0},
                                           'anchor_copy':{},'anchor_graph':{}},
                             'test_scenes':100,'reference_worlds_per_family':100,
                             'reference_families':['a','b']}))
    d=summarize(p)
    assert d['verdict']=='INSUFFICIENT_REFERENCE_EVIDENCE'
    assert 'REFERENCE_RESOLVED_CRITICAL_PAIRS_BELOW_DECLARED_MINIMUM' in d['evidence_gaps']
