"""Predeclared *evidence adequacy* check, not an automatic paper acceptance gate.

This does not replace independent external verification or confidence intervals.
It declines to judge if the critical matched, reference-resolved population is
small. Numerical policies are declared here, not fitted to test outcomes.
"""
import json
from pathlib import Path


def summarize(study_path, boundary_path=None, min_resolved=30,
              min_matched=50, min_scenes=100):
    report=json.loads(Path(study_path).read_text())
    methods=report.get('methods',{})
    baseline=methods.get('source',{})
    reuse=methods.get('anchor_copy',{})
    graph=methods.get('anchor_graph',{})
    n_scene=int(report.get('test_scenes',0))
    n_match=int(baseline.get('matched_kinematic_proxy_scenes') or 0)
    n_sign=int(baseline.get('critical_pair_ref_sign_resolved') or 0)
    n_ref=int(report.get('reference_worlds_per_family') or 0)*len(report.get('reference_families',[]))
    needed=[]
    if n_scene<min_scenes:needed.append('TEST_SCENES_BELOW_DECLARED_MINIMUM')
    if n_match<min_matched:needed.append('KINEMATIC_MANEUVERS_MATCHED_BELOW_DECLARED_MINIMUM')
    if n_sign<min_resolved:needed.append('REFERENCE_RESOLVED_CRITICAL_PAIRS_BELOW_DECLARED_MINIMUM')
    if n_ref<128:needed.append('REFERENCE_MONTE_CARLO_WORLD_COUNT_BELOW_RECOMMENDED_128')
    def key(m,k):return m.get(k)
    primary='reference_resolved_sign_error_rate'
    comparisons={}
    for name,m in methods.items():
        comparisons[name]={
            'resolved_critical_sign_error_rate':key(m,primary),
            'all_critical_underresolution_rate':key(m,'critical_pair_underresolution_rate'),
            'near_boundary_pair_risk_mae':key(m,'near_boundary_pair_risk_mae'),
            'union_risk_mae':key(m,'downstream_union_reference_risk_mae'),
            'matched_pair_ranking_regret':key(m,'matched_ego_safe_safety_ranking_regret')}
    boundary=None
    if boundary_path:
        d=json.loads(Path(boundary_path).read_text())
        boundary={'status':d.get('status'), 'heldout':d.get('heldout'),
                  'screen_is_verified_physical_certificate':False}
        held=d.get('heldout',{})
        if not held.get('reference_contact_flip_pairs'):
            needed.append('NO_HELDOUT_CONTACT_FLIP_PAIRS_FOR_SCREEN_EVALUATION')
        if d.get('status')!='FITTED_EMPIRICAL_SCREENER_NOT_CERTIFIED':
            needed.append('NO_DEVELOPMENT_SUPPORT_FOR_EMPIRICAL_REUSE_SCREEN')
    if needed:
        verdict='INSUFFICIENT_REFERENCE_EVIDENCE'
    elif not graph or graph.get(primary) is None or reuse.get(primary) is None:
        verdict='MISSING_PRIMARY_COMPARATORS'
    else:
        # Descriptive, not a significance claim. A claim of benefit requires
        # paired scene confidence intervals and new model families/domains.
        if (graph[primary]<reuse[primary] and
            graph.get('downstream_union_reference_risk_mae',float('inf')) <=
                baseline.get('downstream_union_reference_risk_mae',float('inf'))):
            verdict='DESCRIPTIVE_SIMULATOR_CONDITIONAL_SIGNAL_ONLY'
        else:
            verdict='NO_DESCRIPTIVE_CBC_CASCADE_ADVANTAGE_DEMONSTRATED'
    ns=int(report.get('test_scenes',0))
    actions=3
    cost={'description':'Per-scene SOURCE REACTOR rollout calls, NOT measured wall-clock latency',
          'source_candidate_recomputes_per_scene':actions,
          'source_rollouts_per_candidate':report.get('source_worlds_per_candidate'),
          'anchor_reuse_recomputes_per_scene':1,
          'single_anchor_rollouts_per_scene':report.get('source_worlds_per_candidate'),
          'source_rollout_reduction_fraction':(actions-1)/actions,
          'scope':'Counting only, no validated online event-boundary certificates or latency measurement'}
    return {'verdict':verdict,'protocol':'CBC_V2_PREDECLARED_EVIDENCE_ADEQUACY',
            'evidence_gaps':needed,'declared_thresholds':{
                'heldout_scenes':min_scenes,'kinematic_matched_scenes':min_matched,
                'reference_resolved_critical_pairs':min_resolved,'reference_worlds':128},
            'observed_sample':{'heldout_scenes':n_scene,'kinematic_matched_scenes':n_match,
                               'reference_resolved_critical_pairs':n_sign,'reference_worlds_per_action':n_ref},
            'method_comparison':comparisons,'empirical_boundary_screen':boundary,
            'model_rollout_effort_proxy':cost,
            'scientific_interpretation':[ 
                'No safety gain or human causal behavior proven by surrogate-reference outcomes',
                'Check paired-scene CI and finite Monte Carlo reference sign resolution before significance',
                'Anchor reuse is not safe online until the hybrid guard envelope and physical prefix are verified',
                'WOMD logged future is observational and cannot identify ego-action counterfactuals',
                'A promising simulator-conditional result must replicate in independently implemented reactive baselines']}
