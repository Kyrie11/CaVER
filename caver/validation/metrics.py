"""Auditable surrogate evaluation of CBC boundary discriminability.

All target probabilities are finite-K reference-reactor sample means, never
observed counterfactual outcomes or real-road causal effects.  Distinguish
signed ranking, union risk calibration and first-event margin diagnostics.
"""
from collections import defaultdict
import numpy as np


def _sign(x, threshold=.05):
    return 1 if x > threshold else -1 if x < -threshold else 0


def _avg(values):
    return float(np.mean(values)) if values else None


def _binned_reliability(points, bins=10):
    """Reference-mean reliability bins; NOT ordinary binary-label ECE."""
    rows=[]
    n=len(points)
    for i in range(bins):
        group=[(p,y) for p,y in points if i/bins <= p < (i+1)/bins or (i==bins-1 and p==1.)]
        if group:
            rows.append({'lower':i/bins,'upper':(i+1)/bins,'count':len(group),
                         'mean_prediction':_avg([p for p,_ in group]),
                         'reference_empirical_mean':_avg([y for _,y in group])})
    ece=sum((r['count']/n)*abs(r['mean_prediction']-r['reference_empirical_mean']) for r in rows) if n else None
    return {'reference_mean_ece':ece,'bins':rows,'label':'SURROGATE_FINITE_K_REFERENCE_MEANS'}


def rq_audit(rows, contrast_threshold=.20, task_progress_tolerance=3.0):
    """Report explicit RQ1/RQ2 falsification metrics; never claim field truth."""
    data=[]
    stats=defaultdict(list)
    pair_rel=[]; union_rel=[]; source_rel=[]
    for row in rows:
        pairs=row['pairs']; acts=[a['action_id'] for a in row['actions']]
        a,b='early','late'
        if a not in acts or b not in acts:
            continue
        pred=row['predicted'];ref=row['audited']
        diag=row.get('diagnostics',{})
        pair_rows=[]
        # Joint event labels are from a *single* simulator realization per world;
        # never reconstruct the union by assuming independent pair contacts.
        for act in acts:
            yp=float(ref[f'{act}/down']);pp=float(pred[f'{act}/down'])
            stats['union_abs_error'].append(abs(pp-yp))
            stats['union_sq_error'].append((pp-yp)**2)
            union_rel.append((pp,yp))
            if pairs and f'{act}/{pairs[0]}' in diag:
                source_p=float(diag[f'{act}/{pairs[0]}']['features'][15])
                stats['source_union_abs_error'].append(abs(source_p-yp))
                source_rel.append((source_p,yp))
            for pair in pairs:
                key=f'{act}/{pair}'
                py=float(pred[key]);ry=float(ref[key])
                pair_rel.append((py,ry))
                stats['pair_abs_error'].append(abs(py-ry))
                stats['pair_squared_error'].append((py-ry)**2)
                if key not in diag:
                    continue
                d=diag[key]
                baseline=d.get('baseline')
                target=d.get('target')
                if baseline is None or target is None:
                    continue
                stats['source_pair_abs_error'].append(abs(float(baseline[0])-ry))
                # Clearance target, mean over the same reference rollouts.
                if 'cbc_margin_prediction_m' in d:
                    stats['cbc_clearance_mae_m'].append(abs(float(d['cbc_margin_prediction_m'])-float(target[1])))
                    stats['source_clearance_mae_m'].append(abs(float(baseline[1])-float(target[1])))
                valid_fraction=float(target[3]);source_valid_fraction=float(baseline[3])
                stats['reference_event_valid_fraction'].append(valid_fraction)
                stats['source_event_valid_fraction'].append(source_valid_fraction)
                # A reference event must actually exist for event-slack error;
                # otherwise target[2] is a storage sentinel, not supervision.
                if valid_fraction > 0. and source_valid_fraction > 0.:
                    stats['source_event_slack_mae_m'].append(abs(float(baseline[2])-float(target[2])))
                    if 'cbc_event_slack_prediction_m' in d:
                        stats['cbc_event_slack_mae_m'].append(abs(float(d['cbc_event_slack_prediction_m'])-float(target[2])))
        for pair in pairs:
            truth=float(ref[f'{a}/{pair}'])-float(ref[f'{b}/{pair}'])
            estimate=float(pred[f'{a}/{pair}'])-float(pred[f'{b}/{pair}'])
            relevant=abs(truth)>=contrast_threshold
            sr=None
            if f'{a}/{pair}' in diag and f'{b}/{pair}' in diag:
                sr=float(diag[f'{a}/{pair}']['baseline'][0])-float(diag[f'{b}/{pair}']['baseline'][0])
            pair_rows.append({'pair':pair,'true_delta':truth,'pred_delta':estimate,
                              'source_delta':sr,'decision_relevant':bool(relevant),
                              'misranked':bool(relevant and _sign(truth)!=_sign(estimate)),
                              'underresolved':bool(relevant and abs(estimate)<.05)})
            if relevant:
                stats['critical_sign_error'].append(float(_sign(truth)!=_sign(estimate)))
                stats['critical_underdetection'].append(float(abs(estimate)<.05))
                if sr is not None:
                    stats['source_critical_sign_error'].append(float(_sign(truth)!=_sign(sr)))
                    stats['source_critical_underdetection'].append(float(abs(sr)<.05))
        downstream_diff=float(ref[f'{a}/down'])-float(ref[f'{b}/down'])
        ego_feasible=(float(ref[f'{a}/ego'])==0. and float(ref[f'{b}/ego'])==0.)
        # Progress and comfort under the reference reactor were not stored in
        # previous JSONL schema; no assertion of matched task execution.
        rq1=ego_feasible and abs(downstream_diff)>=contrast_threshold
        stats['rq1_surrogate_unmatched'].append(float(rq1))
        data.append({'scene_id':row['scene_id'],'candidate_pair':[a,b],
                     'simulated_ego_safe_downstream_risk_contrast_UNMATCHED':bool(rq1),
                     'downstream_delta':downstream_diff,'ego_safe_under_ref':bool(ego_feasible),
                     'task_comfort_progress_match':'NOT_VERIFIED','pair_risks':pair_rows})
    result={
        'scenes':len(data),
        'simulated_ego_safe_downstream_contrast_UNMATCHED':int(sum(r['simulated_ego_safe_downstream_risk_contrast_UNMATCHED'] for r in data)),
        'critical_pair_tests':len(stats['critical_sign_error']),
        'critical_pair_sign_error_rate':_avg(stats['critical_sign_error']),
        'source_critical_pair_sign_error_rate':_avg(stats['source_critical_sign_error']),
        'critical_pair_underresolution_rate':_avg(stats['critical_underdetection']),
        'source_critical_pair_underresolution_rate':_avg(stats['source_critical_underdetection']),
        'cbc_pair_reference_risk_mae':_avg(stats['pair_abs_error']),
        'source_pair_reference_risk_mae':_avg(stats['source_pair_abs_error']),
        'cbc_pair_reference_risk_mse':_avg(stats['pair_squared_error']),
        'cbc_downstream_union_reference_risk_mae':_avg(stats['union_abs_error']),
        'cbc_downstream_union_reference_risk_mse':_avg(stats['union_sq_error']),
        'source_downstream_union_reference_risk_mae':_avg(stats['source_union_abs_error']),
        'cbc_clearance_mae_m':_avg(stats['cbc_clearance_mae_m']),
        'source_clearance_mae_m':_avg(stats['source_clearance_mae_m']),
        'cbc_event_slack_mae_m_given_both_sources_valid':_avg(stats['cbc_event_slack_mae_m']),
        'source_event_slack_mae_m_given_both_sources_valid':_avg(stats['source_event_slack_mae_m']),
        'event_slack_valid_reference_fraction':_avg(stats['reference_event_valid_fraction']),
        'event_slack_valid_source_fraction':_avg(stats['source_event_valid_fraction']),
        'event_slack_jointly_valid_measurements':len(stats['source_event_slack_mae_m']),
        'cbc_pair_reliability':_binned_reliability(pair_rel),
        'cbc_union_reliability':_binned_reliability(union_rel),
        'source_union_reliability':_binned_reliability(source_rel),
        'rows':data,
        'task_progress_tolerance_requested_m':task_progress_tolerance,
        'warning':'All truth labels are Monte-Carlo surrogate-reactor empirical conditional means. RQ1 count is unmatched (task/progress/comfort not yet verified). Reference risk-MSE is not a real-world Brier score. Real causal outcomes and native simulator safety NOT established.'}
    return result
