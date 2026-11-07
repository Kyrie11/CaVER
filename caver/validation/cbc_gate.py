"""CBC-first claim audit: separate descriptive signal from component evidence.

This is not a physical certificate or a test of real-human counterfactuals.
The reference-world adequacy check is tied to the finite-MC radius used by
cbc_study; weak-baseline gains cannot establish graph/event mechanism gains.
"""
import json
import math
from pathlib import Path


_COMPONENT_METRICS = (
    'reference_resolved_sign_error_rate',
    'critical_pair_underresolution_rate',
    'critical_pair_miss_rate',
    'downstream_union_resolved_sign_error_rate',
    'downstream_union_resolved_underresolution_rate',
    'matched_ego_safe_safety_ranking_regret',
)


def _number(value):
    try:
        x=float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _ci_support(report, key, metric='reference_resolved_sign_error_rate'):
    """Improvement means negative paired delta with upper CI strictly below 0."""
    record=report.get(key,{}).get(metric,{})
    ci=record.get('bootstrap_95pct_ci')
    if not isinstance(ci,(list,tuple)) or len(ci)!=2:
        return {'status':'NOT_MEASURED','delta':None,'paired_scene_ci95':None}
    lo,hi=_number(ci[0]),_number(ci[1])
    if lo is None or hi is None or lo>hi:
        return {'status':'NOT_MEASURED','delta':None,'paired_scene_ci95':None}
    delta=_number(record.get('delta_method_minus_source'))
    valid=record.get('resamples_valid')
    requested=record.get('resamples_requested')
    try:
        valid=int(valid) if valid is not None else 0
        requested=int(requested) if requested is not None else 0
    except (TypeError,ValueError):
        valid=requested=0
    valid_fraction=(valid/requested) if requested>0 else None
    result={'delta':delta,'paired_scene_ci95':[lo,hi],
            'valid_bootstrap_resamples':valid,
            'requested_bootstrap_resamples':requested or None,
            'valid_bootstrap_fraction':valid_fraction}
    # A narrow CI from only a handful of non-empty resamples is not evidence.
    # Require at least 100 valid resamples and, when known, >=80% of requests.
    if valid<100 or (requested>0 and valid_fraction<.80):
        result['status']='NOT_MEASURED'
    else:
        result['status']='SUPPORTS_INCREMENTAL_GAIN' if hi<0 else (
            'SUPPORTS_INCREMENTAL_HARM' if lo>0 else 'INCONCLUSIVE_OR_NO_GAIN')
    return result


def _joint_ci_support(report, metrics=_COMPONENT_METRICS):
    """Require all predeclared decision metrics to support the same component."""
    # The caller passes one comparison object; avoid mutating the input report.
    results={metric:_ci_support({'comparison':report},'comparison',metric)
             for metric in metrics}
    statuses=[v['status'] for v in results.values()]
    if all(s=='SUPPORTS_INCREMENTAL_GAIN' for s in statuses):
        status='SUPPORTS_INCREMENTAL_GAIN'
    elif any(s=='SUPPORTS_INCREMENTAL_HARM' for s in statuses):
        status='SUPPORTS_INCREMENTAL_HARM'
    elif any(s=='NOT_MEASURED' for s in statuses):
        status='NOT_MEASURED'
    else:
        status='INCONCLUSIVE_OR_NO_GAIN'
    return {'status':status,'metrics':results,
            'joint_criterion':'all sign-error, underresolution, critical-miss, and ego-safe regret CIs must favor the component'}


def _strict_point_gain(method, control):
    """Conservative descriptive check across all primary decision metrics."""
    values=[]
    for metric in _COMPONENT_METRICS:
        m=_number(method.get(metric))
        c=_number(control.get(metric))
        if m is None or c is None:
            return False
        values.append(m<c)
    return all(values)


def _noninferior_point_gain(method, control):
    """Descriptive check allowing ties on saturated zero-error metrics."""
    values=[]
    strict=False
    for metric in _COMPONENT_METRICS:
        m=_number(method.get(metric)); c=_number(control.get(metric))
        if m is None or c is None:
            return False
        if m>c:
            return False
        strict = strict or (m<c)
        values.append(m<=c)
    return bool(values) and strict and all(values)


def _ranking_sensitivity(report, min_resolved=1):
    """Summarize the lower-threshold ranking ablation separately.

    The primary CBC gate uses the declared critical contrast. A development
    ranking threshold can expose directional supervision, but it must not
    replace the critical denominator. This report is intentionally isolated
    from the primary verdict.
    """
    methods=report.get('ranking_methods',{})
    control=methods.get('anchor_graph_delay_kernel',{})
    treatment=methods.get('delay_kernel_rank',{})
    comparison=report.get('paired_scene_bootstrap_ranking_delay_kernel_rank_minus_delay_kernel',{})
    if not control or not treatment or not comparison:
        return {'status':'NOT_MEASURED',
                'reason':'ranking sensitivity report or paired comparison is missing'}
    resolved=max(int(treatment.get('critical_pair_ref_sign_resolved') or 0),
                 int(treatment.get('downstream_union_resolved_n') or 0))
    if resolved < int(min_resolved):
        return {'status':'NOT_MEASURED',
                'reason':'no finite-reference-resolved ranking pairs',
                'resolved_pairs':resolved}
    metrics=('matched_ego_safe_safety_ranking_regret',
             'matched_ego_safe_pair_ranking_regret',
             'downstream_union_reference_risk_mae')
    point={}
    strict=False
    for metric in metrics:
        m=_number(treatment.get(metric)); c=_number(control.get(metric))
        if m is None or c is None:
            continue
        point[metric]={'treatment':m,'control':c,'delta':m-c}
        strict = strict or m < c
        if m > c:
            return {'status':'NO_RANKING_GAIN', 'resolved_pairs':resolved,
                    'point_comparison':point,
                    'reason':'the treatment is worse on at least one available ranking metric'}
    if not point:
        return {'status':'NOT_MEASURED','resolved_pairs':resolved,
                'reason':'ranking metrics are unavailable'}
    ci={}
    for metric in metrics:
        record=comparison.get(metric,{})
        interval=record.get('bootstrap_95pct_ci')
        if isinstance(interval,(list,tuple)) and len(interval)==2:
            lo,hi=_number(interval[0]),_number(interval[1])
            if lo is not None and hi is not None:
                ci[metric]={'delta':_number(record.get('delta_method_minus_source')),
                            'bootstrap_95pct_ci':[lo,hi]}
    strict_ci=any(v['bootstrap_95pct_ci'][1] < 0. for v in ci.values())
    return {'status':'PRELIMINARY_RANKING_GAIN' if strict and strict_ci else
            'RANKING_GAIN_NOT_STATISTICALLY_RESOLVED',
            'resolved_pairs':resolved,'point_comparison':point,
            'paired_confidence_intervals':ci,
            'threshold':report.get('ranking_pair_abs_delta_threshold'),
            'detection_threshold':report.get('ranking_predicted_detection_threshold'),
            'critical_gate_unchanged':True}


def _equal_budget_audit(report, method_name, control_name, parameter_tolerance=.01,
                        contrast_weight_mode='equal', rank_weight_mode='equal'):
    """Verify same training data/seed/updates and near-equal active parameters."""
    checkpoints=report.get('checkpoints',{})
    method=checkpoints.get(method_name,{}).get('training_meta',{})
    control=checkpoints.get(control_name,{}).get('training_meta',{})
    required=('optimizer_updates_per_epoch','optimizer_updates_total',
              'active_parameter_count','seed','contrast_weight','rank_weight',
              'contrastive_scenes_per_epoch','contrastive_pairs_per_epoch','contrast_threshold',
              'training_scene_set_sha256')
    missing=[key for key in required
             if method.get(key) is None or control.get(key) is None]
    if missing:
        return {'status':'NOT_VERIFIED','missing_metadata':missing,
                'method':method_name,'control':control_name}
    updates_epoch_equal=(int(method['optimizer_updates_per_epoch'])==
                         int(control['optimizer_updates_per_epoch']))
    updates_total_equal=(int(method['optimizer_updates_total'])==
                         int(control['optimizer_updates_total']))
    seed_equal=(int(method['seed'])==int(control['seed']))
    rank_equal=math.isclose(float(method['rank_weight']),float(control['rank_weight']),rel_tol=0.,abs_tol=1e-12)
    rank_higher=float(method['rank_weight'])>float(control['rank_weight'])
    contrast_equal=math.isclose(float(method['contrast_weight']),
                                float(control['contrast_weight']),rel_tol=0.,abs_tol=1e-12)
    contrast_higher=float(method['contrast_weight'])>float(control['contrast_weight'])
    if contrast_weight_mode not in ('equal','higher'):
        raise ValueError('contrast_weight_mode must be equal or higher')
    contrast_weight_ok=contrast_equal if contrast_weight_mode=='equal' else contrast_higher
    if rank_weight_mode not in ('equal','higher'):
        raise ValueError('rank_weight_mode must be equal or higher')
    rank_weight_ok=rank_equal if rank_weight_mode=='equal' else rank_higher
    contrast_scenes_equal=(int(method['contrastive_scenes_per_epoch'])==
                           int(control['contrastive_scenes_per_epoch']))
    contrast_pairs_equal=(int(method['contrastive_pairs_per_epoch'])==
                          int(control['contrastive_pairs_per_epoch']))
    contrast_threshold_equal=math.isclose(float(method['contrast_threshold']),
                                           float(control['contrast_threshold']),
                                           rel_tol=0.,abs_tol=1e-12)
    # New checkpoints record the independent ranking threshold and the
    # finite-MC filtering confidence.  Keep old reports loadable while
    # requiring equality whenever both values are available.
    rank_threshold_equal=(
        math.isclose(float(method['rank_threshold']),float(control['rank_threshold']),
                     rel_tol=0.,abs_tol=1e-12)
        if method.get('rank_threshold') is not None and control.get('rank_threshold') is not None
        else True)
    reference_alpha_equal=(
        math.isclose(float(method['reference_alpha']),float(control['reference_alpha']),
                     rel_tol=0.,abs_tol=1e-12)
        if method.get('reference_alpha') is not None and control.get('reference_alpha') is not None
        else True)
    ranking_scene_count_equal=(
        int(method['ranking_scenes_per_epoch'])==int(control['ranking_scenes_per_epoch'])
        if method.get('ranking_scenes_per_epoch') is not None and control.get('ranking_scenes_per_epoch') is not None
        else True)
    ranking_pair_count_equal=(
        int(method['ranking_pairs_per_epoch'])==int(control['ranking_pairs_per_epoch'])
        if method.get('ranking_pairs_per_epoch') is not None and control.get('ranking_pairs_per_epoch') is not None
        else True)
    arrival_kernel_shift_equal=(
        bool(method['candidate_arrival_kernel_shift'])==bool(control['candidate_arrival_kernel_shift'])
        if method.get('candidate_arrival_kernel_shift') is not None and control.get('candidate_arrival_kernel_shift') is not None
        else True)
    split_equal=(method['training_scene_set_sha256']==control['training_scene_set_sha256'])
    p_method=float(method['active_parameter_count'])
    p_control=float(control['active_parameter_count'])
    parameter_ratio=(p_method/p_control) if p_control>0 else math.inf
    parameter_matched=(math.isfinite(parameter_ratio) and
                       abs(parameter_ratio-1.)<=parameter_tolerance)
    checks={'optimizer_updates_per_epoch_equal':updates_epoch_equal,
            'optimizer_updates_total_equal':updates_total_equal,
            'active_parameter_ratio':parameter_ratio,
            'active_parameters_within_tolerance':parameter_matched,
            'seed_equal':seed_equal,'rank_weight_equal':rank_equal,
            'contrast_weight_equal':contrast_equal,
            'contrast_weight_treatment_higher':contrast_higher,
            'contrastive_scene_count_equal':contrast_scenes_equal,
            'contrastive_pair_count_equal':contrast_pairs_equal,
            'contrast_threshold_equal':contrast_threshold_equal,
            'rank_threshold_equal':rank_threshold_equal,
            'reference_alpha_equal':reference_alpha_equal,
            'ranking_scene_count_equal':ranking_scene_count_equal,
            'ranking_pair_count_equal':ranking_pair_count_equal,
            'candidate_arrival_kernel_shift_equal':arrival_kernel_shift_equal,
            'training_scene_set_equal':split_equal,
            'rank_weight_mode_satisfied':rank_weight_ok}
    checks['contrast_weight_mode_satisfied']=contrast_weight_ok
    status='MATCHED' if all(v for k,v in checks.items()
                            if k not in ('active_parameter_ratio','contrast_weight_equal',
                                         'contrast_weight_treatment_higher')) and parameter_matched else 'MISMATCH'
    return {'status':status,'method':method_name,'control':control_name,
            'parameter_tolerance_fraction':parameter_tolerance,'checks':checks}


def summarize(study_path, boundary_path=None, min_resolved=30,
              min_matched=50, min_scenes=100):
    report=json.loads(Path(study_path).read_text())
    methods=report.get('methods',{})
    baseline=methods.get('source',{})
    reuse=methods.get('anchor_copy',{})
    graph=methods.get('anchor_graph',{})
    anchor_no_graph=methods.get('anchor_full',{})
    n_scene=int(report.get('test_scenes',0) or 0)
    n_match=int(baseline.get('matched_kinematic_proxy_scenes') or 0)
    n_sign=int(baseline.get('critical_pair_ref_sign_resolved') or 0)

    families=report.get('reference_families') or []
    worlds_per_family=int(report.get('reference_worlds_per_family') or 0)
    computed_ref=worlds_per_family*len(families)
    declared_ref=report.get('reference_worlds_per_candidate')
    n_ref=int(declared_ref) if declared_ref is not None else computed_ref

    mc_meta=baseline.get('finite_mc_radius') or {}
    mc_alpha=_number(mc_meta.get('alpha',.05))
    mc_pairs=int(mc_meta.get('n_pairs') or 3)
    critical_delta=_number(report.get('critical_pair_abs_delta_threshold',.20))
    if mc_alpha is None or not 0.<mc_alpha<1.:
        raise ValueError('CBC gate requires finite_mc_radius.alpha in (0,1)')
    if mc_pairs<1:
        raise ValueError('CBC gate requires finite_mc_radius.n_pairs >= 1')
    if critical_delta is None or not 0.<critical_delta<1.:
        raise ValueError('critical_pair_abs_delta_threshold must lie in (0,1)')

    required_worlds=int(math.ceil(2.*math.log(2.*mc_pairs/mc_alpha)/(critical_delta**2)))
    observed_radius=(math.sqrt(2.*math.log(2.*mc_pairs/mc_alpha)/n_ref)
                     if n_ref>0 else math.inf)
    needed=[]
    if n_scene<min_scenes:needed.append('TEST_SCENES_BELOW_DECLARED_MINIMUM')
    if n_match<min_matched:needed.append('KINEMATIC_MANEUVERS_MATCHED_BELOW_DECLARED_MINIMUM')
    if n_sign<min_resolved:needed.append('REFERENCE_RESOLVED_CRITICAL_PAIRS_BELOW_DECLARED_MINIMUM')
    if n_ref<required_worlds:needed.append('REFERENCE_MONTE_CARLO_WORLD_COUNT_BELOW_RESOLUTION_REQUIREMENT')

    if computed_ref and declared_ref is not None and int(declared_ref)!=computed_ref:
        needed.append('REFERENCE_WORLD_COUNT_METADATA_MISMATCH')
    if not mc_meta:
        needed.append('FINITE_MC_RESOLUTION_METADATA_MISSING')
    else:
        meta_worlds=mc_meta.get('n_worlds')
        meta_radius=_number(mc_meta.get('radius'))
        if meta_worlds is not None and int(meta_worlds)!=n_ref:
            needed.append('FINITE_MC_WORLD_COUNT_METADATA_MISMATCH')
        if meta_radius is not None and n_ref>0 and not math.isclose(
                meta_radius,observed_radius,rel_tol=1e-6,abs_tol=1e-9):
            needed.append('FINITE_MC_RADIUS_METADATA_MISMATCH')

    comparisons={}
    for name,m in methods.items():
        comparisons[name]={
            'resolved_critical_sign_error_rate':m.get('reference_resolved_sign_error_rate'),
            'all_critical_underresolution_rate':m.get('critical_pair_underresolution_rate'),
            'all_critical_miss_rate':m.get('critical_pair_miss_rate'),
            'near_boundary_pair_risk_mae':m.get('near_boundary_pair_risk_mae'),
            'union_risk_mae':m.get('downstream_union_reference_risk_mae'),
            'union_resolved_sign_error_rate':m.get('downstream_union_resolved_sign_error_rate'),
            'union_resolved_underresolution_rate':m.get('downstream_union_resolved_underresolution_rate'),
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

    graph_vs_copy=_ci_support(
        report.get('paired_scene_bootstrap_method_minus_anchor_copy',{}),'anchor_graph')
    graph_vs_no_graph=_joint_ci_support(
        report.get('paired_scene_bootstrap_anchor_graph_minus_anchor_full',{}))
    event_vs_no_event=_joint_ci_support(
        report.get('paired_scene_bootstrap_full_minus_no_events',{}))
    event_local_vs_graph=_joint_ci_support(
        report.get('paired_scene_bootstrap_anchor_graph_event_local_minus_anchor_graph',{}))
    delay_kernel_vs_graph=_joint_ci_support(
        report.get('paired_scene_bootstrap_anchor_graph_delay_kernel_minus_anchor_graph',{}))
    delay_kernel_rank_vs_kernel=_joint_ci_support(
        report.get('paired_scene_bootstrap_delay_kernel_rank_minus_delay_kernel',{}))

    graph_point_advantage=_strict_point_gain(graph,anchor_no_graph)
    graph_point_noninferior=_noninferior_point_gain(graph,anchor_no_graph)
    delay_kernel_point_noninferior=_noninferior_point_gain(
        methods.get('anchor_graph_delay_kernel',{}), graph)
    event_local_point_noninferior=_noninferior_point_gain(
        methods.get('anchor_graph_event_local',{}), graph)
    event_point_advantage=_strict_point_gain(methods.get('full',{}),
                                              methods.get('no_events',{}))
    graph_budget=_equal_budget_audit(report,'anchor_graph','anchor_full')
    event_budget=_equal_budget_audit(report,'full','no_events')
    full_contrast_ci=_joint_ci_support(
        report.get('paired_scene_bootstrap_full_contrast_minus_full',{}))
    graph_contrast_ci=_joint_ci_support(
        report.get('paired_scene_bootstrap_anchor_graph_contrast_minus_anchor_graph',{}))
    full_contrast_point=_strict_point_gain(methods.get('full_contrast',{}),methods.get('full',{}))
    graph_contrast_point=_strict_point_gain(methods.get('anchor_graph_contrast',{}),graph)
    full_contrast_budget=_equal_budget_audit(
        report,'full_contrast','full',contrast_weight_mode='higher')
    graph_contrast_budget=_equal_budget_audit(
        report,'anchor_graph_contrast','anchor_graph',contrast_weight_mode='higher')
    ranking_budget=_equal_budget_audit(
        report,'delay_kernel_rank','anchor_graph_delay_kernel',rank_weight_mode='higher')

    graph_component_supported=(graph_point_advantage
        and graph_vs_no_graph['status']=='SUPPORTS_INCREMENTAL_GAIN'
        and graph_budget['status']=='MATCHED')
    event_component_supported=(event_point_advantage
        and event_vs_no_event['status']=='SUPPORTS_INCREMENTAL_GAIN'
        and event_budget['status']=='MATCHED')
    full_contrast_supported=(full_contrast_point
        and full_contrast_ci['status']=='SUPPORTS_INCREMENTAL_GAIN'
        and full_contrast_budget['status']=='MATCHED')
    graph_contrast_supported=(graph_contrast_point
        and graph_contrast_ci['status']=='SUPPORTS_INCREMENTAL_GAIN'
        and graph_contrast_budget['status']=='MATCHED')
    ranking_point=_strict_point_gain(methods.get('delay_kernel_rank',{}),
                                     methods.get('anchor_graph_delay_kernel',{}))
    ranking_component_supported=(ranking_point
        and delay_kernel_rank_vs_kernel['status']=='SUPPORTS_INCREMENTAL_GAIN'
        and ranking_budget['status']=='MATCHED')
    ranking_sensitivity=_ranking_sensitivity(report)

    if full_contrast_supported or graph_contrast_supported or ranking_component_supported:
        contrastive_verdict='PRELIMINARY_SIMULATOR_CONDITIONAL_CONTRASTIVE_GAIN'
    elif ((full_contrast_point and full_contrast_ci['status']=='SUPPORTS_INCREMENTAL_GAIN'
           and full_contrast_budget['status']!='MATCHED')
          or (graph_contrast_point and graph_contrast_ci['status']=='SUPPORTS_INCREMENTAL_GAIN'
              and graph_contrast_budget['status']!='MATCHED')):
        contrastive_verdict='CONTRASTIVE_GAIN_NOT_EQUAL_BUDGET_VERIFIED'
    elif full_contrast_point or graph_contrast_point or ranking_point:
        contrastive_verdict='DESCRIPTIVE_CONTRASTIVE_GAIN_NOT_STATISTICALLY_RESOLVED'
    else:
        contrastive_verdict='CONTRASTIVE_GAIN_NOT_DEMONSTRATED'

    if needed:
        verdict='INSUFFICIENT_REFERENCE_EVIDENCE'
    else:
        graph_sign=_number(graph.get('reference_resolved_sign_error_rate'))
        reuse_sign=_number(reuse.get('reference_resolved_sign_error_rate'))
        graph_union=_number(graph.get('downstream_union_reference_risk_mae'))
        source_union=_number(baseline.get('downstream_union_reference_risk_mae'))
        if graph_sign is None or reuse_sign is None or graph_union is None or source_union is None:
            verdict='MISSING_PRIMARY_COMPARATORS'
        elif (graph_sign<=reuse_sign and graph_union<=source_union
              and (graph_sign<reuse_sign or graph_union<source_union)):
            # This is only a descriptive simulator-conditional contrast.
            verdict='DESCRIPTIVE_SIMULATOR_CONDITIONAL_SIGNAL_ONLY'
        else:
            verdict='NO_DESCRIPTIVE_CBC_CASCADE_ADVANTAGE_DEMONSTRATED'

    if needed:
        novelty='NOT_TESTABLE_WITH_CURRENT_EVIDENCE'
    elif not graph_point_advantage and not event_point_advantage and not ranking_point:
        novelty='INCREMENTAL_GRAPH_AND_EVENT_GAIN_NOT_DEMONSTRATED'
    elif graph_component_supported or event_component_supported or ranking_component_supported:
        novelty='PRELIMINARY_SIMULATOR_CONDITIONAL_COMPONENT_GAIN'
    elif ((graph_point_advantage and graph_vs_no_graph['status']=='SUPPORTS_INCREMENTAL_GAIN'
           and graph_budget['status']!='MATCHED')
          or (event_point_advantage and event_vs_no_event['status']=='SUPPORTS_INCREMENTAL_GAIN'
              and event_budget['status']!='MATCHED')):
        novelty='COMPONENT_GAIN_NOT_EQUAL_BUDGET_VERIFIED'
    elif graph_point_advantage or event_point_advantage or ranking_point:
        novelty='DESCRIPTIVE_COMPONENT_GAIN_NOT_STATISTICALLY_RESOLVED'
    else:
        novelty='INCREMENTAL_GRAPH_AND_EVENT_GAIN_NOT_DEMONSTRATED'

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

    # Keep evidence gaps unique and deterministic.
    needed=list(dict.fromkeys(needed))
    return {'verdict':verdict,'novelty_verdict':novelty,
            'protocol':'CBC_V3_ABLATION_AND_EVIDENCE_ADEQUACY',
            'evidence_gaps':needed,
            'declared_thresholds':{'heldout_scenes':min_scenes,
                'kinematic_matched_scenes':min_matched,
                'reference_resolved_critical_pairs':min_resolved,
                'reference_worlds':required_worlds,
                'reference_resolution_formula':'sqrt(2*log(2*n_pairs/alpha)/n_worlds)',
                'critical_pair_abs_delta_threshold':critical_delta,
                'finite_mc_alpha':mc_alpha,'pair_comparisons_per_scene':mc_pairs},
            'observed_sample':{'heldout_scenes':n_scene,
                'kinematic_matched_scenes':n_match,
                'reference_resolved_critical_pairs':n_sign,
                'reference_worlds_per_family':worlds_per_family,
                'reference_families':families,
                'reference_worlds_per_action':n_ref,
                'reference_worlds_required_for_delta':required_worlds,
                'observed_finite_mc_radius':observed_radius,
                'reference_resolved_critical_scenes':baseline.get('critical_pair_ref_sign_resolved_scenes'),
                'matched_ego_safe_scenes':baseline.get('matched_ego_safe_scenes'),
                'matched_legal_ego_safe_scenes':baseline.get('matched_legal_ego_safe_scenes'),
                'matched_ego_safe_legal_unknown_scenes':baseline.get('matched_ego_safe_legal_unknown_scenes'),
                'reference_matched_contact_flip_pairs':baseline.get('reference_matched_contact_flip_n'),
                'reference_all_contact_flip_pairs':baseline.get('reference_any_pair_contact_flip_all_candidates_n'),
                'reference_all_pair_comparisons':baseline.get('reference_all_pair_comparisons_n'),
                'reference_matched_pair_comparisons':baseline.get('reference_matched_pair_comparisons_n'),
                'match_failure_codes':baseline.get('match_failure_codes'),
                'reference_scene_familywise_resolved_critical_n':baseline.get('reference_familywise_within_scene_resolved_critical_n')},
            'method_comparison':comparisons,
            'incremental_ablation':{
                'anchor_graph_minus_anchor_copy':graph_vs_copy,
                'anchor_graph_minus_equal_anchor_budget_without_graph':graph_vs_no_graph,
                'full_minus_equal_full_source_budget_no_events':event_vs_no_event,
                'anchor_graph_event_local_minus_anchor_graph':event_local_vs_graph,
                'anchor_graph_delay_kernel_minus_anchor_graph':delay_kernel_vs_graph,
                'delay_kernel_rank_minus_delay_kernel':delay_kernel_rank_vs_kernel,
                'graph_equal_budget_audit':graph_budget,
                'event_equal_budget_audit':event_budget,
                'ranking_equal_budget_audit':ranking_budget,
                'graph_component_gain_supported':graph_component_supported,
                'event_component_gain_supported':event_component_supported,
                'graph_strictly_better_than_anchor_full_point_estimate':graph_point_advantage,
                'graph_noninferior_with_strict_metric_gain_point_estimate':graph_point_noninferior,
                'delay_kernel_noninferior_with_strict_metric_gain_point_estimate':delay_kernel_point_noninferior,
                'event_local_noninferior_with_strict_metric_gain_point_estimate':event_local_point_noninferior,
                'full_strictly_better_than_no_events_point_estimate':event_point_advantage,
                'ranking_component_gain_supported':ranking_component_supported,
                'ranking_point_gain':ranking_point,
                'ranking_sensitivity':ranking_sensitivity},
            'contrastive_ablation':{
                'full_contrast_minus_full':full_contrast_ci,
                'anchor_graph_contrast_minus_anchor_graph':graph_contrast_ci,
                'full_contrast_equal_budget_audit':full_contrast_budget,
                'anchor_graph_contrast_equal_budget_audit':graph_contrast_budget,
                'full_contrast_point_gain':full_contrast_point,
                'anchor_graph_contrast_point_gain':graph_contrast_point,
                'delay_kernel_rank_point_gain':ranking_point,
                'verdict':contrastive_verdict},
            'empirical_boundary_screen':boundary,'model_rollout_effort_proxy':cost,
            'scientific_interpretation':[
                'An anchor-copy gain does not establish the delayed graph is helpful: compare anchor_full',
                'Component gains require concordant improvement in sign error, underresolution, critical miss, ego-safe decision regret, and verified equal training budgets',
                'Critical ranking uses kinematically matched, reference-ego-safe candidate pairs; unknown route legality remains unverified',
                'An event-feature claim requires full vs no_events at matched capacity',
                'A ranking-loss claim requires a boundary-resolved downstream-union contrast, an independent delay-kernel control, and equal update/capacity metadata',
                'A contrastive-loss claim requires full_contrast vs full or anchor_graph_contrast vs anchor_graph under equal optimizer updates and active capacity',
                'WOMD history is observational and does not identify ego-action counterfactuals',
                'Scene-wise bootstrap is not proof of external-model generalization; paired dependent edges require clustered uncertainty',
                'Native reactive closed-loop evidence and physical guard bounds are still required']}
