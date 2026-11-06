"""Paired, held-out, small-output CBC falsification study.

Compares methods *on the same scene and independently simulated reference law*.
All source temporal features are predicted by a source reactor; reference
intervention labels are never model inputs. Both are synthetic unless an
independently executed native simulator supplies the test records.
"""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import math
import numpy as np

from caver.experiment import read_records,record_with_new_predictions
from caver.models.cbc import load,coherent_project
from caver.validation.extended_metrics import finite_mc_radius, finite_mc_radius_metadata, FINITE_MC_RADIUS_SCHEMA, FINITE_MC_RADIUS_FORMULA


def _avg(v):
    return float(np.mean(v)) if v else None


def _score_prediction(row, predicted, threshold, detection):
    """Score critical ranking only on matched, ego-safe candidate pairs.

    Unknown route legality is retained for simulator-conditional diagnostics,
    but an explicitly illegal pair is excluded.  Unknown legality is never
    counted as verified legal evidence.
    """
    aud=row['audited']
    names=[a['action_id'] for a in row['actions']
           if a['action_id'] in ('early','late')]
    empty = {'matched':0, 'analysis_eligible':False, 'lawful_route_match':None,
             'matched_ego_safe':False, 'matched_legal_ego_safe':False,
             'matched_ego_safe_legal_unknown':False,
             'critical':[], 'resolved':[], 'pair_errors':[], 'down_errors':[],
             'regret':None, 'pair_regrets':[], 'decision_abstain':None,
             'event':[], 'downstream_events':[], 'pair_errors_near':[], 'pair_errors_far':[],
             'risk_reversal':False, 'ego_safe':False}
    if not ('early' in names and 'late' in names):
        return empty

    match=row.get('candidate_matching',{}).get('pairs',{}).get('early|late',{})
    failures=[]

    def _within(value, limit):
        try:
            value=float(value); limit=float(limit)
            return math.isfinite(value) and math.isfinite(limit) and value <= limit
        except (TypeError, ValueError):
            return False

    if match.get('same_maneuver_intent') is not True:
        failures.append('INTENT_MISMATCH')
    if not _within(match.get('d_e'), match.get('delta_e_max')):
        failures.append('EGO_TRAJECTORY_DISTANCE')
    if not _within(match.get('task_cost_difference'), match.get('delta_J_max')):
        failures.append('TASK_COMFORT_MISMATCH')
    if match.get('kinematic_feasible_both') is not True:
        failures.append('EGO_DYNAMICS')

    reported_proxy=bool(match.get('matched_kinematic_proxy',False))
    kinematic_matched=reported_proxy and not failures
    if not reported_proxy and not failures:
        failures.append('UNKNOWN_MATCH_FAILURE')

    law=match.get('lawful_route_match')
    legal_unknown=(law is None or law in ('NOT_EVALUATED','UNVERIFIED'))
    legal_ok=(law is True or legal_unknown)
    if law is False:
        failures.append('ROUTE_ILLEGAL')
    elif not legal_ok:
        failures.append('ROUTE_LEGALITY_UNSUPPORTED')

    # The synthetic protocol may retain unknown legal status for a
    # simulator-conditional analysis, but never for a legal-route claim.
    eligible=kinematic_matched and legal_ok

    n=len(row.get('audit_families',[]))*int(row.get('audit_worlds_per_family',0))
    n_pairs=max(1,len(row['pairs']))
    radius=finite_mc_radius(n,n_pairs,.05)
    radius_scene=radius
    a,b='early','late'
    ego_early=float(aud[f'{a}/ego'])
    ego_late=float(aud[f'{b}/ego'])
    ego_safe=(math.isfinite(ego_early) and math.isfinite(ego_late)
              and ego_early<=.05 and ego_late<=.05)
    risk_reversal=(eligible and ego_safe
                   and abs(float(aud[f'{a}/down'])-float(aud[f'{b}/down']))>=threshold)

    pairs=[];errors=[];near=[];far=[];events=[]
    for pair in row['pairs']:
        true=float(aud[f'{a}/{pair}']-aud[f'{b}/{pair}'])
        est=float(predicted[f'{a}/{pair}']-predicted[f'{b}/{pair}'])
        # Primary critical ranking denominator: matched, not explicitly
        # illegal, and both candidate actions reference-ego-safe.
        if eligible and ego_safe and abs(true)>=threshold:
            events.append({'scene_id':row.get('scene_id'),'pair':pair,'true_delta':true,'estimate':est,
                           'underresolution':float(abs(est)<detection),
                           'sign_error':float(abs(est)>=detection and est*true<0),
                           'critical_miss':float(abs(est)<detection or (est*true<0)),
                           'reference_sign_resolved':bool(abs(true)>radius),
                           'reference_sign_resolved_scene_familywise':bool(abs(true)>radius_scene),
                           'reference_mc_radius':radius,
                           'reference_mc_radius_scene_familywise':radius_scene,
                           'finite_mc_radius':finite_mc_radius_metadata(n,n_pairs,.05)})
        # Regression MAE remains a broad all-scene candidate diagnostic;
        # it is not the matched/ego-safe primary ranking denominator.
        for name in names:
            key=f'{name}/{pair}'
            e=abs(float(predicted[key])-float(aud[key]))
            errors.append(e)
            d=row.get('boundary_diagnostics',{}).get(pair,{})
            dist=d.get('anchor_min_abs_clearance_m')
            if dist is not None:
                (near if dist<=2. else far).append(e)
        bd=row.get('boundary_diagnostics',{}).get(pair,{})
        if bd:
            pairs.append({'pair':pair,'reference_contact_flip':bd['reference']['contact_flip_rate'],
                          'source_contact_flip':bd['source']['contact_flip_rate'],
                          'ref_guard_flip':bd['reference']['any_brake_guard_flip_rate'],
                          'anchor_abs_clearance':bd['anchor_min_abs_clearance_m'],
                          'anchor_abs_guard':bd['anchor_min_abs_brake_guard_mps2']})

    down_errors=[abs(float(predicted[f'{name}/down'])-float(aud[f'{name}/down']))
                 for name in names]
    downstream_events=[]
    if eligible and ego_safe:
        down_true=float(aud['early/down'])-float(aud['late/down'])
        down_est=float(predicted['early/down'])-float(predicted['late/down'])
        if abs(down_true)>=threshold:
            downstream_events.append({'scene_id':row.get('scene_id'),
                'true_delta':down_true,'estimate':down_est,
                'underresolution':float(abs(down_est)<detection),
                'sign_error':float(abs(down_est)>=detection and down_est*down_true<0),
                'critical_miss':float(abs(down_est)<detection or down_est*down_true<0),
                'reference_sign_resolved':bool(abs(down_true)>radius),
                'reference_mc_radius':radius})
    regret=None; pair_regrets=[]; decision_abstain=None
    if eligible and ego_safe:
        pred_safe=[name for name in (a,b) if float(predicted[f'{name}/ego'])<=.05]
        ref_safe=[name for name in (a,b) if float(aud[f'{name}/ego'])<=.05]
        decision_abstain=not bool(pred_safe)
        if pred_safe and ref_safe:
            choice=min(pred_safe,key=lambda name:(predicted[f'{name}/down'],name))
            regret=float(aud[f'{choice}/down']-min(aud[f'{name}/down'] for name in ref_safe))
            for pair in row['pairs']:
                pchoice=min(pred_safe,key=lambda name:(predicted[f'{name}/{pair}'],name))
                pair_regrets.append(float(aud[f'{pchoice}/{pair}']-
                                         min(aud[f'{name}/{pair}'] for name in ref_safe)))

    matched_ego_safe=bool(eligible and ego_safe)
    return {'matched':int(kinematic_matched),
            'analysis_eligible':bool(eligible),
            'lawful_route_match':law if isinstance(law,bool) else None,
            'matched_ego_safe':matched_ego_safe,
            'matched_legal_ego_safe':bool(kinematic_matched and ego_safe and law is True),
            'matched_ego_safe_legal_unknown':bool(kinematic_matched and ego_safe and legal_unknown),
            'match_failure_codes':failures,
            'reference_ego_risk_early':ego_early,
            'reference_ego_risk_late':ego_late,
            'critical':events,'resolved':[x for x in events if x['reference_sign_resolved']],
            'downstream_events':downstream_events,
            'pair_errors':errors,'down_errors':down_errors,'regret':regret,
            'pair_regrets':pair_regrets,'decision_abstain':decision_abstain,
            'event':pairs,'pair_errors_near':near,'pair_errors_far':far,
            'risk_reversal':risk_reversal,'ego_safe':ego_safe,
            # Preserve the declared reference budget in the scored row.  The
            # aggregate is built from scored rows rather than raw records, so
            # dropping these fields silently produced n_worlds=0 metadata in
            # v5 reports.
            'audit_families':list(row.get('audit_families', [])),
            'audit_worlds_per_family':int(row.get('audit_worlds_per_family', 0)),
            'n_pairs':int(len(row.get('pairs', [])))}


def _aggregate(rows):
    critical=[c for r in rows for c in r['critical']]
    resolved=[c for r in rows for c in r['resolved']]
    downstream=[c for r in rows for c in r.get('downstream_events',[])]
    downstream_resolved=[c for c in downstream if c['reference_sign_resolved']]
    errors=[x for r in rows for x in r['pair_errors']]
    reg=[r['regret'] for r in rows if r['regret'] is not None]
    pair_reg=[x for r in rows for x in r.get('pair_regrets',[])]
    pairs=[p for r in rows for p in r['event']]
    hits=[p for p in pairs if p['reference_contact_flip']>0]
    matched_pairs=[p for r in rows if r['matched'] for p in r['event']]
    familywise=[c for c in critical if c['reference_sign_resolved_scene_familywise']]
    resolved_scene_ids={str(c['scene_id']) for c in resolved if c.get('scene_id') is not None}
    reasons=Counter(reason for r in rows for reason in r.get('match_failure_codes',[]))
    return {'scenes':len(rows),'matched_kinematic_proxy_scenes':sum(r['matched'] for r in rows),
        'match_failure_codes':dict(reasons),
        'reference_any_pair_contact_flip_all_candidates_n':len(hits),
        'reference_all_pair_comparisons_n':len(pairs),
        'reference_matched_pair_comparisons_n':len(matched_pairs),
        'reference_matched_contact_flip_n':sum(p['reference_contact_flip']>0 for p in matched_pairs),
        'reference_familywise_within_scene_resolved_critical_n':len(familywise),
        'finite_mc_radius':finite_mc_radius_metadata(
            int(rows[0].get('audit_worlds_per_family',0))*len(rows[0].get('audit_families',[])) if rows else 0,
            max(1, int(rows[0].get('n_pairs', 1))) if rows else 1, .05),
        'reference_mc_ci_familywise_scope':'within-scene pairwise only; not global-FWER across scenes',
        'matched_ego_safe_scenes':sum(int(r.get('matched_ego_safe',False)) for r in rows),
        'matched_legal_ego_safe_scenes':sum(int(r.get('matched_legal_ego_safe',False)) for r in rows),
        'matched_ego_safe_legal_unknown_scenes':sum(int(r.get('matched_ego_safe_legal_unknown',False)) for r in rows),
        'matched_downstream_contrast_scenes':sum(int(r['risk_reversal']) for r in rows),
        'critical_pair_tests':len(critical),'critical_pair_ref_sign_resolved':len(resolved),
        'critical_pair_ref_sign_resolved_scenes':len(resolved_scene_ids),
        'critical_pair_sign_error_rate':_avg([x['sign_error'] for x in critical]),
        'reference_resolved_sign_error_rate':_avg([x['sign_error'] for x in resolved]),
        'critical_pair_underresolution_rate':_avg([x['underresolution'] for x in critical]),
        'critical_pair_miss_rate':_avg([x['critical_miss'] for x in critical]),
        'downstream_union_sign_error_rate':_avg([x['sign_error'] for x in downstream]),
        'downstream_union_underresolution_rate':_avg([x['underresolution'] for x in downstream]),
        'downstream_union_miss_rate':_avg([x['critical_miss'] for x in downstream]),
        'downstream_union_resolved_sign_error_rate':_avg([x['sign_error'] for x in downstream_resolved]),
        'downstream_union_resolved_underresolution_rate':_avg([x['underresolution'] for x in downstream_resolved]),
        'downstream_union_resolved_miss_rate':_avg([x['critical_miss'] for x in downstream_resolved]),
        'downstream_union_contrast_n':len(downstream),
        'downstream_union_resolved_n':len(downstream_resolved),
        'pair_reference_risk_mae':_avg(errors),
        'near_boundary_pair_risk_mae':_avg([x for r in rows for x in r['pair_errors_near']]),
        'far_boundary_pair_risk_mae':_avg([x for r in rows for x in r['pair_errors_far']]),
        'downstream_union_reference_risk_mae':_avg([x for r in rows for x in r['down_errors']]),
        'matched_ego_safe_safety_ranking_regret':_avg(reg),
        'matched_ego_safe_safety_ranking_n':len(reg),
        'matched_ego_safe_pair_ranking_regret':_avg(pair_reg),
        'matched_ego_safe_pair_ranking_n':len(pair_reg),
        'predicted_ego_safe_decision_abstain_rate':_avg([float(r['decision_abstain']) for r in rows if r.get('decision_abstain') is not None]),
        'reference_any_pair_contact_flip_n':len(hits),
        'source_contact_flip_missed_reference_fraction':_avg([
            float(p['source_contact_flip']==0.) for p in hits]),
        'anchor_clearance_gt_2m_among_ref_flips_fraction':_avg([
            float(p['anchor_abs_clearance']>2.) for p in hits if p['anchor_abs_clearance'] is not None]),
        'boundary_warning':'Near=predicted anchor |clearance|<=2 m, not a validated physical guard interval'}


def _bootstrap(d1,d0,seed=51,n_boot=500):
    if len(d1)!=len(d0) or len(d1)<2:
        return {}
    rng=np.random.default_rng(seed);n=len(d1)
    keys=('critical_pair_sign_error_rate','reference_resolved_sign_error_rate',
          'critical_pair_underresolution_rate','critical_pair_miss_rate',
          'downstream_union_sign_error_rate','downstream_union_underresolution_rate',
          'downstream_union_miss_rate','downstream_union_resolved_sign_error_rate',
          'downstream_union_resolved_underresolution_rate','downstream_union_resolved_miss_rate',
          'matched_ego_safe_safety_ranking_regret','matched_ego_safe_pair_ranking_regret',
          'pair_reference_risk_mae','downstream_union_reference_risk_mae')
    samples={k:[] for k in keys}
    for _ in range(n_boot):
        idx=rng.integers(0,n,n)
        a=_aggregate([d1[i] for i in idx]);b=_aggregate([d0[i] for i in idx])
        for k in keys:
            if a[k] is not None and b[k] is not None:
                samples[k].append(a[k]-b[k])
    return {k:{'delta_method_minus_source':float(np.mean(v)),
               'bootstrap_95pct_ci':[float(np.quantile(v,.025)),float(np.quantile(v,.975))],
               'resamples_valid':len(v),
               'resamples_requested':int(n_boot),
               'resamples_valid_fraction':float(len(v)/n_boot) if n_boot else 0.0}
            for k,v in samples.items() if v}


def study(test_file,models,contrast=.20,detect=.05,bootstrap=2000,max_examples=8):
    """models: {label: checkpoint path}; reserves 'source' untrained baseline."""
    if not 0<detect<contrast<1 or bootstrap<0:
        raise ValueError('Invalid contrast/detection/bootstrap')
    records=list(read_records(test_file))
    if not records or any(r.get('experimental_split')!='test' for r in records):
        raise ValueError('CBC study requires frozen held-out --split test records')
    if len({r['scene_id'] for r in records})!=len(records):
        raise ValueError('Duplicate test scene')
    if len({tuple(r['audit_families']) for r in records})!=1:
        raise ValueError('Multiple reference-law definitions within test')
    models={'source':None,**models}
    predictions={};checksums={}
    # True one-anchor-only fast-reuse baseline: no candidate-conditioned reactor
    # prediction for any non-anchor action. Its limitations are measurable.
    anchor_copy=[]
    for r in records:
        view=r.get('anchor_view')
        if not view:raise ValueError('Test record lacks saved one-anchor evidence')
        pred={}
        for a in [x['action_id'] for x in r['actions']]:
            feats=[view['candidate_features'][f'{a}/{p}'] for p in r['pairs']]
            for p,feat in zip(r['pairs'],feats):pred[f'{a}/{p}']=float(feat[9])
            pred[f'{a}/down']=float(feats[0][15])
            pred[f'{a}/ego']=float(feats[0][16])
        anchor_copy.append(coherent_project(pred,[a['action_id'] for a in r['actions']],r['pairs']))
    predictions['anchor_copy']=anchor_copy
    for name,checkpoint in models.items():
        if name in ('source','anchor_copy') and checkpoint is not None:raise ValueError('source/anchor_copy are reserved baseline names')
        if checkpoint is None:
            predictions[name]=[r['predicted'] for r in records]
            continue
        model=load(checkpoint)
        tr_ids=set(model.meta.get('training_scene_sha256',[]))
        tr_groups=set(model.meta.get('training_group_sha256',[]))
        for r in records:
            if sha256(r['scene_id'].encode()).hexdigest() in tr_ids:
                raise ValueError('CBC test scene directly appeared in training checkpoint')
            gp=r.get('scene',{}).get('provenance',{}).get('overlap_group',r['scene_id'])
            if sha256(str(gp).encode()).hexdigest() in tr_groups:
                raise ValueError('CBC test overlap group appeared in training checkpoint')
        train_ids=sorted(model.meta.get('training_scene_sha256',[]))
        train_digest=(sha256('\\n'.join(train_ids).encode()).hexdigest() if train_ids else None)
        training_meta={key:model.meta.get(key) for key in (
            'epochs','optimizer_updates_per_epoch','optimizer_updates_total',
            'active_parameter_count','total_parameter_count','seed','contrast_weight','rank_weight',
            'contrastive_scenes_per_epoch','contrastive_pairs_per_epoch','contrast_threshold')}
        training_meta['training_scene_set_sha256']=train_digest
        checksums[name]={'checkpoint_sha256':sha256(Path(checkpoint).read_bytes()).hexdigest(),
                         'model_variant':model.variant,
                         'training_meta':training_meta}
        predictions[name]=[record_with_new_predictions(r,model)['predicted'] for r in records]
    details={name:[_score_prediction(row,p,contrast,detect) for row,p in zip(records,pred)]
             for name,pred in predictions.items()}
    aggregates={name:_aggregate(d) for name,d in details.items()}
    paired={name:_bootstrap(d,details['source'],n_boot=bootstrap) for name,d in details.items() if name!='source'}
    # Critical counterfactual controls for the *incremental* CBC claim:
    # an anchor-only graph must beat the no-adjustment anchor-copy at the
    # same single-anchor rollout budget. An event-feature head must beat the
    # matched full-model capacity control, not just the source reactor.
    paired_vs_anchor={name:_bootstrap(d,details['anchor_copy'],n_boot=bootstrap)
                      for name,d in details.items() if name not in ('anchor_copy','source')}
    paired_events=(_bootstrap(details['full'],details['no_events'],n_boot=bootstrap)
                   if 'full' in details and 'no_events' in details else {})
    paired_graph=(_bootstrap(details['anchor_graph'],details['anchor_full'],n_boot=bootstrap)
                   if 'anchor_graph' in details and 'anchor_full' in details else {})
    paired_event_local=(_bootstrap(details['anchor_graph_event_local'],details['anchor_graph'],n_boot=bootstrap)
                        if 'anchor_graph_event_local' in details and 'anchor_graph' in details else {})
    paired_delay_kernel=(_bootstrap(details['anchor_graph_delay_kernel'],details['anchor_graph'],n_boot=bootstrap)
                         if 'anchor_graph_delay_kernel' in details and 'anchor_graph' in details else {})
    paired_delay_kernel_rank=(_bootstrap(details['delay_kernel_rank'],details['anchor_graph_delay_kernel'],n_boot=bootstrap)
                              if 'delay_kernel_rank' in details and 'anchor_graph_delay_kernel' in details else {})
    paired_full_contrast=(_bootstrap(details['full_contrast'],details['full'],n_boot=bootstrap)
                          if 'full_contrast' in details and 'full' in details else {})
    paired_graph_contrast=(_bootstrap(details['anchor_graph_contrast'],details['anchor_graph'],n_boot=bootstrap)
                           if 'anchor_graph_contrast' in details and 'anchor_graph' in details else {})
    # Return only bounded-size, predeclared examples. No raw trajectories,
    # no source-world tensor dump, and no train/calibration target records.
    cases=[]
    for i,r in enumerate(records):
        base=details['source'][i]
        if not (base['critical'] or base['risk_reversal']):continue
        p=next((p for p in base['critical'] if p['pair']=='B|C'),None)
        cases.append({'scene_id':r['scene_id'], 'matched_proxy':bool(base['matched']),
                      'reference_risk_early_late':{
                          'early':r['audited'].get('early/down'),
                          'late':r['audited'].get('late/down')},
                      'BC_reference_pair_delta':p['true_delta'] if p else None,
                      'source_predicted_pair_delta':p['estimate'] if p else None})
        if len(cases)>=max_examples:break
    return {'protocol':'CBC_HELDOUT_SIMULATOR_CONDITIONAL_FALSIFICATION_V1',
            'candidate_actions':['early','late'],'reference_baseline_actions':['reference'],
            'test_file_sha256':sha256(Path(test_file).read_bytes()).hexdigest(),
            'test_scenes':len(records),'reference_families':records[0]['audit_families'],
            'reference_worlds_per_family':records[0]['audit_worlds_per_family'],
            'critical_pair_abs_delta_threshold':contrast,'predicted_detection_threshold':detect,
            'methods':aggregates,'paired_scene_bootstrap_method_minus_source':paired,
            'paired_scene_bootstrap_method_minus_anchor_copy':paired_vs_anchor,
            'paired_scene_bootstrap_full_minus_no_events':paired_events,
            'paired_scene_bootstrap_anchor_graph_minus_anchor_full':paired_graph,
            'paired_scene_bootstrap_anchor_graph_event_local_minus_anchor_graph':paired_event_local,
            'paired_scene_bootstrap_anchor_graph_delay_kernel_minus_anchor_graph':paired_delay_kernel,
            'paired_scene_bootstrap_delay_kernel_rank_minus_delay_kernel':paired_delay_kernel_rank,
            'paired_scene_bootstrap_full_contrast_minus_full':paired_full_contrast,
            'paired_scene_bootstrap_anchor_graph_contrast_minus_anchor_graph':paired_graph_contrast,
            'source_worlds_per_candidate':records[0].get('source_worlds'),
            'reference_worlds_per_candidate':len(records[0]['audit_families'])*int(records[0]['audit_worlds_per_family']),
            'checkpoints':checksums,'interesting_test_cases_bounded':cases,
            'identification_and_claim_limits':[
                'Reference outcomes are finite-MC model-conditional means, not human potential outcomes',
                'Training and reference reactors are distinct model families but are synthetic unless explicitly replaced',
                'Kinematic candidate matching does NOT verify map-law compliance or physical prefix safety',
                'Source reactor already runs candidate-conditioned full rollouts: this study has not yet benchmarked CBC anchor-reuse speedup',
                'Graph head is time-causal but physical guard envelope completeness is UNVERIFIED',
                'Paired bootstrap quantifies sampling across held-out test scenes, not unseen domain shifts',
                'If resolved critical pair count is zero, risk ranking evidence is not publication-conclusive']}
