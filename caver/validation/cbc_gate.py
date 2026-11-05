"""CBC-first claim audit: do not mistake weak-baseline gains for mechanism gains.

This is NOT a physical certificate or a test of real human counterfactuals.
Every incremental claim has a method-capacity-matched control and a paired
scene-level confidence interval, if available.  Missing comparisons abstain.
"""
import json
import math
from pathlib import Path


def _ci_support(report, key, metric='reference_resolved_sign_error_rate'):
    """Improvement means negative difference with *upper* CI strictly below zero."""
    record=report.get(key,{}).get(metric,{})
    ci=record.get('bootstrap_95pct_ci')
    if ci is None or len(ci)!=2:
        return {'status':'NOT_MEASURED','delta':None,'paired_scene_ci95':None}
    result={'delta':record.get('delta_method_minus_source'), 'paired_scene_ci95':ci,
            'valid_bootstrap_resamples':record.get('resamples_valid')}
    result['status']='SUPPORTS_INCREMENTAL_GAIN' if ci[1]<0 else (
        'SUPPORTS_INCREMENTAL_HARM' if ci[0]>0 else 'INCONCLUSIVE_OR_NO_GAIN')
    return result


def summarize(study_path, boundary_path=None, min_resolved=30,
              min_matched=50, min_scenes=100):
    report=json.loads(Path(study_path).read_text())
    methods=report.get('methods',{})
    baseline=methods.get('source',{})
    reuse=methods.get('anchor_copy',{})
    graph=methods.get('anchor_graph',{})
    anchor_no_graph=methods.get('anchor_full',{})
    n_scene=int(report.get('test_scenes',0))
    n_match=int(baseline.get('matched_kinematic_proxy_scenes') or 0)
    n_sign=int(baseline.get('critical_pair_ref_sign_resolved') or 0)
    n_ref=int(report.get('reference_worlds_per_family') or 0)*len(report.get('reference_families',[]))
    mc_meta=baseline.get('finite_mc_radius',{}) or {}
    mc_alpha=float(mc_meta.get('alpha',.05))
    mc_pairs=max(1,int(mc_meta.get('n_pairs',3)))
    critical_delta=float(report.get('critical_pair_abs_delta_threshold',.20))
    required_worlds=int(math.ceil(2.*math.log(2.*mc_pairs/mc_alpha)/(critical_delta**2)))
    needed=[]
    if n_scene<min_scenes:needed.append('TEST_SCENES_BELOW_DECLARED_MINIMUM')
    if n_match<min_matched:needed.append('KINEMATIC_MANEUVERS_MATCHED_BELOW_DECLARED_MINIMUM')
    if n_sign<min_resolved:needed.append('REFERENCE_RESOLVED_CRITICAL_PAIRS_BELOW_DECLARED_MINIMUM')
    if n_ref<required_worlds:needed.append('REFERENCE_MONTE_CARLO_WORLD_COUNT_BELOW_RESOLUTION_REQUIREMENT')
    comparisons={}
    for name,m in methods.items():
        comparisons[name]={
            'resolved_critical_sign_error_rate':m.get('reference_resolved_sign_error_rate'),
            'all_critical_underresolution_rate':m.get('critical_pair_underresolution_rate'),
            'near_boundary_pair_risk_mae':m.get('near_boundary_pair_risk_mae'),
            'union_risk_mae':m.get('downstream_union_reference_risk_mae'),
            'matched_pair_ranking_regret':m.get('matched_ego_safe_safety_ranking_regret')}
    boundary=None
    if boundary_path:
        d=json.loads(Path(boundary_path).read_text())
        boundary={'status':d.get('status'),'heldout':d.get('heldout'),
                  'screen_is_verified_physical_certificate':False}
        held=d.get('heldout',{})
        if d.get('development_split') != ['calibration']:
            needed.append('BOUNDARY_DEVELOPMENT_NOT_INDEPENDENT_CALIBRATION')
        if not held.get('reference_contact_flip_pairs'):
            needed.append('NO_HELDOUT_CONTACT_FLIP_PAIRS_FOR_SCREEN_EVALUATION')
        if d.get('status')!='FITTED_EMPIRICAL_SCREENER_NOT_CERTIFIED':
            needed.append('NO_DEVELOPMENT_SUPPORT_FOR_EMPIRICAL_REUSE_SCREEN')
    graph_vs_copy=_ci_support(report.get('paired_scene_bootstrap_method_minus_anchor_copy',{}),
                              'anchor_graph')
    graph_vs_no_graph=_ci_support({'graph':report.get('paired_scene_bootstrap_anchor_graph_minus_anchor_full',{})},
                                  'graph')
    event_vs_no_event=_ci_support({'event':report.get('paired_scene_bootstrap_full_minus_no_events',{})},
                                  'event')
    # Older reports lack graph-vs-no-graph CI; their descriptive point estimates
    # are still informative, but do not turn them into statistically proved wins.
    graph_point_advantage=(graph.get('reference_resolved_sign_error_rate') is not None
        and anchor_no_graph.get('reference_resolved_sign_error_rate') is not None
        and graph['reference_resolved_sign_error_rate']<anchor_no_graph['reference_resolved_sign_error_rate'])
    event_point_advantage=(methods.get('full',{}).get('reference_resolved_sign_error_rate') is not None
        and methods.get('no_events',{}).get('reference_resolved_sign_error_rate') is not None
        and methods['full']['reference_resolved_sign_error_rate']<methods['no_events']['reference_resolved_sign_error_rate'])
    if needed:
        verdict='INSUFFICIENT_REFERENCE_EVIDENCE'
    elif not graph or graph.get('reference_resolved_sign_error_rate') is None or reuse.get('reference_resolved_sign_error_rate') is None:
        verdict='MISSING_PRIMARY_COMPARATORS'
    elif (graph.get('reference_resolved_sign_error_rate')<reuse.get('reference_resolved_sign_error_rate')
          and graph.get('downstream_union_reference_risk_mae',float('inf')) <= baseline.get('downstream_union_reference_risk_mae',float('inf'))):
        verdict='DESCRIPTIVE_SIMULATOR_CONDITIONAL_SIGNAL_ONLY'
    else:
        verdict='NO_DESCRIPTIVE_CBC_CASCADE_ADVANTAGE_DEMONSTRATED'
    # Mechanism-level research claim is stricter than the legacy weak-baseline gate.
    if needed:
        novelty='NOT_TESTABLE_WITH_CURRENT_EVIDENCE'
    elif not graph_point_advantage and not event_point_advantage:
        novelty='INCREMENTAL_GRAPH_AND_EVENT_GAIN_NOT_DEMONSTRATED'
    elif (graph_vs_no_graph['status']=='SUPPORTS_INCREMENTAL_GAIN'
          or event_vs_no_event['status']=='SUPPORTS_INCREMENTAL_GAIN'):
        novelty='PRELIMINARY_SIMULATOR_CONDITIONAL_COMPONENT_GAIN'
    else:
        novelty='DESCRIPTIVE_COMPONENT_GAIN_NOT_STATISTICALLY_RESOLVED'
    candidate_actions=report.get('candidate_actions',['early','late'])
    reference_actions=report.get('reference_baseline_actions',['reference'])
    c=len(candidate_actions)
    cost={'description':'Per-scene SOURCE REACTOR rollout calls, NOT measured wall-clock latency',
          'source_candidate_recomputes_per_scene':c,
          'candidate_actions':candidate_actions,
          'reference_baseline_recomputes_per_scene':len(reference_actions),
          'reference_baseline_actions':reference_actions,
          'source_rollouts_per_candidate':report.get('source_worlds_per_candidate'),
          'anchor_reuse_recomputes_per_scene':1,
          'single_anchor_rollouts_per_scene':report.get('source_worlds_per_candidate'),
          'source_rollout_reduction_fraction':(c-1)/c if c else None,
          'scope':'Counting only: real inference, graph-head and fallback costs not measured'}
    return {'verdict':verdict,'novelty_verdict':novelty,
            'protocol':'CBC_V3_ABLATION_AND_EVIDENCE_ADEQUACY',
            'evidence_gaps':needed,
            'declared_thresholds':{'heldout_scenes':min_scenes,
                'kinematic_matched_scenes':min_matched,
                'reference_resolved_critical_pairs':min_resolved,'reference_worlds':required_worlds,
                'reference_resolution_formula':'sqrt(2*log(2*n_pairs/alpha)/n_worlds)'},
            'observed_sample':{'heldout_scenes':n_scene,'kinematic_matched_scenes':n_match,
                'reference_resolved_critical_pairs':n_sign,'reference_worlds_per_action':n_ref,
                'reference_worlds_required_for_delta':required_worlds,
                'reference_matched_contact_flip_pairs':baseline.get('reference_matched_contact_flip_n'),
                'reference_all_contact_flip_pairs':baseline.get('reference_any_pair_contact_flip_all_candidates_n'),
                'reference_all_pair_comparisons':baseline.get('reference_all_pair_comparisons_n'),
                'reference_matched_pair_comparisons':baseline.get('reference_matched_pair_comparisons_n'),
                'matched_ego_safe_scenes':baseline.get('matched_ego_safe_scenes'),
                'match_failure_codes':baseline.get('match_failure_codes'),
                'reference_scene_familywise_resolved_critical_n':baseline.get('reference_familywise_within_scene_resolved_critical_n')},
            'method_comparison':comparisons,
            'incremental_ablation':{
                'anchor_graph_minus_anchor_copy':graph_vs_copy,
                'anchor_graph_minus_equal_anchor_budget_without_graph':graph_vs_no_graph,
                'full_minus_equal_full_source_budget_no_events':event_vs_no_event,
                'graph_strictly_better_than_anchor_full_point_estimate':graph_point_advantage,
                'full_strictly_better_than_no_events_point_estimate':event_point_advantage},
            'empirical_boundary_screen':boundary,'model_rollout_effort_proxy':cost,
            'scientific_interpretation':[
                'An anchor-copy gain does not establish the delayed graph is helpful: compare anchor_full',
                'An event-feature claim requires full vs no_events at matched capacity',
                'WOMD history is observational and does not identify ego-action counterfactuals',
                'Scene-wise bootstrap is not proof of external-model generalization; paired dependent edges require clustered uncertainty',
                'Native reactive closed-loop evidence and physical guard bounds are still required']}
