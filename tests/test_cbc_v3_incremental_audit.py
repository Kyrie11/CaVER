"""Guard against inflation of graph/event claims on unequal baselines."""
import json
import math
from caver.validation.cbc_gate import summarize


def test_novelty_requires_stronger_equal_budget_control(tmp_path):
    results = {
        'test_scenes': 120,
        'reference_worlds_per_family': 128,
        'reference_families': ['a', 'b'],
        'reference_worlds_per_candidate': 256,
        'source_worlds_per_candidate': 4,
        'methods': {
            'source': {'matched_kinematic_proxy_scenes': 90, 'critical_pair_ref_sign_resolved': 40,
                       'finite_mc_radius': {'alpha': 0.05, 'n_pairs': 3, 'n_worlds': 256,
                                            'radius': (2.0 * math.log(120.0) / 256.0) ** 0.5},
                       'downstream_union_reference_risk_mae': 0.3},
            'anchor_copy': {'reference_resolved_sign_error_rate': 1.0},
            'anchor_graph': {'reference_resolved_sign_error_rate': 0.30,
                             'downstream_union_reference_risk_mae': 0.1},
            'anchor_full': {'reference_resolved_sign_error_rate': 0.20},
            'full': {'reference_resolved_sign_error_rate': 0.05},
            'no_events': {'reference_resolved_sign_error_rate': 0.05},
        }
    }
    f = tmp_path / 'report.json'
    f.write_text(json.dumps(results))
    gated = summarize(f)
    # A positive comparison to a trivially weak anchor-copy predictor cannot
    # establish any incremental value of a learned temporal graph.
    assert gated['verdict'] == 'DESCRIPTIVE_SIMULATOR_CONDITIONAL_SIGNAL_ONLY'
    assert gated['novelty_verdict'] == 'INCREMENTAL_GRAPH_AND_EVENT_GAIN_NOT_DEMONSTRATED'
    assert not gated['incremental_ablation']['graph_strictly_better_than_anchor_full_point_estimate']
    assert not gated['incremental_ablation']['full_strictly_better_than_no_events_point_estimate']


def test_missing_critical_reference_forces_abstention(tmp_path):
    f=tmp_path/'empty.json'
    f.write_text(json.dumps({'test_scenes':120,'reference_families':['a','b'],
       'reference_worlds_per_family':96,'reference_worlds_per_candidate':192,
       'methods':{'source':{'matched_kinematic_proxy_scenes':15,
                            'critical_pair_ref_sign_resolved':0,
                            'finite_mc_radius': {'alpha':0.05,'n_pairs':3,'n_worlds':192,
                                                 'radius': (2.0 * math.log(120.0) / 192.0) ** 0.5}}}}))
    got=summarize(f)
    assert got['novelty_verdict']=='NOT_TESTABLE_WITH_CURRENT_EVIDENCE'
    assert 'REFERENCE_RESOLVED_CRITICAL_PAIRS_BELOW_DECLARED_MINIMUM' in got['evidence_gaps']



def test_underpowered_reference_worlds_block_descriptive_verdict_even_with_resolved_pairs(tmp_path):
    # 96 worlds/family = 192 total worlds; with P=3 and alpha=.05 the
    # finite-MC radius is about .223, above the predeclared .20 contrast.
    results = {
        'test_scenes': 120,
        'reference_worlds_per_family': 96,
        'reference_worlds_per_candidate': 192,
        'reference_families': ['a', 'b'],
        'critical_pair_abs_delta_threshold': .20,
        'methods': {
            'source': {
                'matched_kinematic_proxy_scenes': 90,
                'critical_pair_ref_sign_resolved': 40,
                'finite_mc_radius': {'alpha': .05, 'n_pairs': 3, 'n_worlds': 192,
                                     'radius': (2.0 * math.log(120.0) / 192.0) ** .5},
                'downstream_union_reference_risk_mae': .3,
            },
            'anchor_copy': {'reference_resolved_sign_error_rate': 1.0},
            'anchor_graph': {'reference_resolved_sign_error_rate': .30,
                             'downstream_union_reference_risk_mae': .1},
            'anchor_full': {'reference_resolved_sign_error_rate': .20},
            'full': {'reference_resolved_sign_error_rate': .05},
            'no_events': {'reference_resolved_sign_error_rate': .05},
        },
    }
    f=tmp_path/'underpowered.json'
    f.write_text(json.dumps(results))
    gated=summarize(f)
    assert gated['verdict']=='INSUFFICIENT_REFERENCE_EVIDENCE'
    assert 'REFERENCE_MONTE_CARLO_WORLD_COUNT_BELOW_RESOLUTION_REQUIREMENT' in gated['evidence_gaps']


def test_reference_world_metadata_mismatch_forces_abstention(tmp_path):
    results = {
        'test_scenes': 120,
        'reference_worlds_per_family': 128,
        'reference_worlds_per_candidate': 256,
        'reference_families': ['a', 'b'],
        'methods': {
            'source': {
                'matched_kinematic_proxy_scenes': 90,
                'critical_pair_ref_sign_resolved': 40,
                'finite_mc_radius': {'alpha': .05, 'n_pairs': 3, 'n_worlds': 192,
                                     'radius': (2.0 * math.log(120.0) / 192.0) ** .5},
            },
            'anchor_copy': {'reference_resolved_sign_error_rate': 1.0},
            'anchor_graph': {'reference_resolved_sign_error_rate': .30,
                             'downstream_union_reference_risk_mae': .1},
            'anchor_full': {'reference_resolved_sign_error_rate': .20},
            'full': {'reference_resolved_sign_error_rate': .05},
            'no_events': {'reference_resolved_sign_error_rate': .05},
        },
    }
    f=tmp_path/'mismatch.json'
    f.write_text(json.dumps(results))
    gated=summarize(f)
    assert gated['verdict']=='INSUFFICIENT_REFERENCE_EVIDENCE'
    assert 'FINITE_MC_WORLD_COUNT_METADATA_MISMATCH' in gated['evidence_gaps']



def _matched_test_row(ego_risk=.01, lawful_route_match='NOT_EVALUATED'):
    pairs=['A|B','A|C','B|C']
    audited={'early/ego':ego_risk,'late/ego':ego_risk,
             'early/down':.10,'late/down':.45}
    predicted={'early/ego':ego_risk,'late/ego':ego_risk,
               'early/down':.12,'late/down':.40}
    for pair in pairs:
        audited[f'early/{pair}']=.70
        audited[f'late/{pair}']=.25
        predicted[f'early/{pair}']=.65
        predicted[f'late/{pair}']=.30
    return {
        'scene_id':'scene-001',
        'actions':[{'action_id':'early'},{'action_id':'late'}],
        'pairs':pairs,
        'audit_families':['family-a','family-b'],
        'audit_worlds_per_family':128,
        'audited':audited,
        'predicted':predicted,
        'candidate_matching':{'pairs':{'early|late':{
            'same_maneuver_intent':True,'d_e':.01,'delta_e_max':.10,
            'task_cost_difference':.01,'delta_J_max':.10,
            'kinematic_feasible_both':True,'matched_kinematic_proxy':True,
            'lawful_route_match':lawful_route_match}}},
        'boundary_diagnostics':{},
    }


def test_primary_critical_ranking_excludes_reference_ego_unsafe_pairs():
    from caver.validation.cbc_study import _score_prediction
    row=_matched_test_row(ego_risk=.10)
    scored=_score_prediction(row,row['predicted'],threshold=.20,detection=.05)
    assert scored['matched']==1
    assert scored['matched_ego_safe'] is False
    assert scored['critical']==[]
    assert scored['resolved']==[]


def test_primary_critical_ranking_keeps_ego_safe_simulator_pairs_but_not_legal_claim():
    from caver.validation.cbc_study import _score_prediction
    row=_matched_test_row(ego_risk=.01,lawful_route_match='NOT_EVALUATED')
    scored=_score_prediction(row,row['predicted'],threshold=.20,detection=.05)
    assert scored['matched']==1
    assert scored['matched_ego_safe'] is True
    assert scored['matched_legal_ego_safe'] is False
    assert scored['matched_ego_safe_legal_unknown'] is True
    assert len(scored['critical'])==3
    assert len(scored['resolved'])==3


def test_explicitly_illegal_pair_is_excluded_from_primary_ranking():
    from caver.validation.cbc_study import _score_prediction
    row=_matched_test_row(ego_risk=.01,lawful_route_match=False)
    scored=_score_prediction(row,row['predicted'],threshold=.20,detection=.05)
    assert scored['matched']==1
    assert scored['analysis_eligible'] is False
    assert scored['critical']==[]
    assert 'ROUTE_ILLEGAL' in scored['match_failure_codes']


def test_inconsistent_matching_flag_cannot_override_failed_intent_check():
    from caver.validation.cbc_study import _score_prediction
    row=_matched_test_row(ego_risk=.01)
    row['candidate_matching']['pairs']['early|late']['same_maneuver_intent']=False
    scored=_score_prediction(row,row['predicted'],threshold=.20,detection=.05)
    assert scored['matched']==0
    assert scored['critical']==[]
    assert 'INTENT_MISMATCH' in scored['match_failure_codes']



def test_gate_does_not_accept_ci_from_too_few_valid_bootstrap_resamples():
    from caver.validation.cbc_gate import _ci_support
    comparison={'anchor_graph':{
        'reference_resolved_sign_error_rate':{
            'delta_method_minus_source':-.5,
            'bootstrap_95pct_ci':[-.8,-.2],
            'resamples_valid':3,
            'resamples_requested':2000,
        }
    }}
    assert _ci_support(comparison,'anchor_graph')['status']=='NOT_MEASURED'


def test_gate_accepts_component_ci_only_with_sufficient_valid_bootstrap_fraction():
    from caver.validation.cbc_gate import _ci_support
    comparison={'anchor_graph':{
        'reference_resolved_sign_error_rate':{
            'delta_method_minus_source':-.2,
            'bootstrap_95pct_ci':[-.3,-.1],
            'resamples_valid':1800,
            'resamples_requested':2000,
        }
    }}
    result=_ci_support(comparison,'anchor_graph')
    assert result['status']=='SUPPORTS_INCREMENTAL_GAIN'
    assert result['valid_bootstrap_fraction']==.9



def test_graph_and_anchor_full_active_parameter_budgets_are_matched():
    from caver.models.cbc import CascadeRiskHead, JointRiskHead, FEATURES
    from caver.models.temporal import GraphCascadeRiskHead

    anchor=CascadeRiskHead(variant='anchor_full')
    anchor.joint=JointRiskHead()
    graph=CascadeRiskHead(variant='anchor_graph')
    graph.joint=JointRiskHead()
    graph.graph_model=GraphCascadeRiskHead(len(FEATURES),channels=16)
    for module in (graph.feature_norm,graph.encoder,graph.output):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    count=lambda model:sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio=count(graph)/count(anchor)
    assert abs(ratio-1.0)<=.01


def test_equal_budget_audit_requires_same_updates_seed_data_and_active_capacity():
    from caver.validation.cbc_gate import _equal_budget_audit
    common={'optimizer_updates_per_epoch':300,'optimizer_updates_total':19500,
            'active_parameter_count':25425,'seed':11,'contrast_weight':0.0,
            'contrastive_scenes_per_epoch':0,'contrastive_pairs_per_epoch':0,
            'contrast_threshold':.20,'training_scene_set_sha256':'same-train-split'}
    graph={**common,'active_parameter_count':25371}
    report={'checkpoints':{
        'anchor_graph':{'training_meta':graph},
        'anchor_full':{'training_meta':common},
    }}
    assert _equal_budget_audit(report,'anchor_graph','anchor_full')['status']=='MATCHED'
    report['checkpoints']['anchor_graph']['training_meta']['optimizer_updates_total']=19000
    assert _equal_budget_audit(report,'anchor_graph','anchor_full')['status']=='MISMATCH'
    assert _equal_budget_audit({},'anchor_graph','anchor_full')['status']=='NOT_VERIFIED'



def test_component_gain_requires_joint_metrics_ci_and_equal_budget_metadata(tmp_path):
    from caver.validation.cbc_gate import summarize

    def ci(delta=-.05):
        return {'delta_method_minus_source':delta,
                'bootstrap_95pct_ci':[delta-.02,delta-.01],
                'resamples_valid':1800,'resamples_requested':2000}
    common_meta={'optimizer_updates_per_epoch':300,'optimizer_updates_total':19500,
                 'seed':11,'contrast_weight':0.0,
                 'contrastive_scenes_per_epoch':0,'contrastive_pairs_per_epoch':0,
                 'contrast_threshold':.20,'training_scene_set_sha256':'train-split'}
    report={
        'test_scenes':120,'reference_worlds_per_family':128,
        'reference_worlds_per_candidate':256,'reference_families':['a','b'],
        'critical_pair_abs_delta_threshold':.20,
        'methods':{
            'source':{'matched_kinematic_proxy_scenes':90,'critical_pair_ref_sign_resolved':40,
                      'finite_mc_radius':{'alpha':.05,'n_pairs':3,'n_worlds':256,
                                          'radius':(2*math.log(120)/256)**.5},
                      'downstream_union_reference_risk_mae':.30},
            'anchor_copy':{'reference_resolved_sign_error_rate':.8},
            'anchor_full':{'reference_resolved_sign_error_rate':.20,
                           'critical_pair_underresolution_rate':.30,
                           'critical_pair_miss_rate':.40,
                           'matched_ego_safe_safety_ranking_regret':.10},
            'anchor_graph':{'reference_resolved_sign_error_rate':.10,
                            'critical_pair_underresolution_rate':.20,
                            'critical_pair_miss_rate':.30,
                            'matched_ego_safe_safety_ranking_regret':.05,
                            'downstream_union_reference_risk_mae':.20},
            'full':{'reference_resolved_sign_error_rate':.10,
                    'critical_pair_underresolution_rate':.20,
                    'critical_pair_miss_rate':.30,
                    'matched_ego_safe_safety_ranking_regret':.05},
            'no_events':{'reference_resolved_sign_error_rate':.20,
                         'critical_pair_underresolution_rate':.30,
                         'critical_pair_miss_rate':.40,
                         'matched_ego_safe_safety_ranking_regret':.10},
        },
        'paired_scene_bootstrap_anchor_graph_minus_anchor_full':{
            key:ci() for key in ('reference_resolved_sign_error_rate',
                                 'critical_pair_underresolution_rate',
                                 'critical_pair_miss_rate',
                                 'matched_ego_safe_safety_ranking_regret')},
        'paired_scene_bootstrap_full_minus_no_events':{
            key:ci() for key in ('reference_resolved_sign_error_rate',
                                 'critical_pair_underresolution_rate',
                                 'critical_pair_miss_rate',
                                 'matched_ego_safe_safety_ranking_regret')},
        'checkpoints':{
            'anchor_graph':{'training_meta':{**common_meta,'active_parameter_count':25371}},
            'anchor_full':{'training_meta':{**common_meta,'active_parameter_count':25425}},
            'full':{'training_meta':{**common_meta,'active_parameter_count':25425}},
            'no_events':{'training_meta':{**common_meta,'active_parameter_count':25425}},
        },
    }
    f=tmp_path/'component_gain.json'
    f.write_text(json.dumps(report))
    gated=summarize(f)
    assert gated['novelty_verdict']=='PRELIMINARY_SIMULATOR_CONDITIONAL_COMPONENT_GAIN'
    assert gated['incremental_ablation']['graph_equal_budget_audit']['status']=='MATCHED'

    # If training metadata is absent, a statistically favorable CI alone is
    # not enough to claim a component gain at equal budget.
    report.pop('checkpoints')
    f.write_text(json.dumps(report))
    gated=summarize(f)
    assert gated['novelty_verdict']=='COMPONENT_GAIN_NOT_EQUAL_BUDGET_VERIFIED'



def test_contrastive_rank_mask_requires_matched_ego_safe_mc_resolved_critical_pairs():
    from caver.experiment import _contrastive_reference_mask
    row=_matched_test_row(ego_risk=.01,lawful_route_match='NOT_EVALUATED')
    row['audited']['early/A|B']=.46
    row['audited']['late/A|B']=.25  # delta=.21
    row['audited']['early/A|C']=.70
    row['audited']['late/A|C']=.25  # delta=.45
    row['audited']['early/B|C']=.65
    row['audited']['late/B|C']=.25  # delta=.40
    mask=_contrastive_reference_mask(row,contrast_threshold=.20,alpha=.05)
    assert mask.tolist()==[True,True,True]  # K=256, radius < .20

    row['audit_worlds_per_family']=96  # K=192, radius > .20
    mask=_contrastive_reference_mask(row,contrast_threshold=.20,alpha=.05)
    assert mask.tolist()==[False,True,True]

    row['audited']['early/ego']=.10
    assert _contrastive_reference_mask(row,contrast_threshold=.20,alpha=.05) is None


def test_contrastive_rank_mask_rejects_explicitly_illegal_or_unmatched_pairs():
    from caver.experiment import _contrastive_reference_mask
    row=_matched_test_row(ego_risk=.01,lawful_route_match=False)
    assert _contrastive_reference_mask(row) is None
    row=_matched_test_row(ego_risk=.01)
    row['candidate_matching']['pairs']['early|late']['same_maneuver_intent']=False
    assert _contrastive_reference_mask(row) is None



def test_contrastive_ablation_budget_allows_loss_weight_treatment_but_requires_same_updates():
    from caver.validation.cbc_gate import _equal_budget_audit
    control={'optimizer_updates_per_epoch':300,'optimizer_updates_total':19500,
             'active_parameter_count':25425,'seed':11,'contrast_weight':0.0,
             'contrastive_scenes_per_epoch':12,'contrastive_pairs_per_epoch':24,
             'contrast_threshold':.20,'training_scene_set_sha256':'same-train-split'}
    treatment={**control,'contrast_weight':.10}
    report={'checkpoints':{
        'full_contrast':{'training_meta':treatment},
        'full':{'training_meta':control},
    }}
    assert _equal_budget_audit(
        report,'full_contrast','full',contrast_weight_mode='higher')['status']=='MATCHED'
    report['checkpoints']['full_contrast']['training_meta']['optimizer_updates_total']=19600
    assert _equal_budget_audit(
        report,'full_contrast','full',contrast_weight_mode='higher')['status']=='MISMATCH'
