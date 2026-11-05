"""Falsifiable *empirical* CBC anchor-reuse triage; NOT a physical certificate.

Thresholds fit on development outcomes only. On held-out reference reactors,
report both missed contact flips and recomputation cost proxy. Real-time reuse
must still default to full recomputation until complete verified bounds exist.
"""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import math
import numpy as np
from caver.experiment import read_records


def _collect(rows,contact_flip_min=0.):
    items=[]
    for r in rows:
        m=r.get('candidate_matching',{}).get('pairs',{}).get('early|late',{})
        if not m.get('matched_kinematic_proxy'):continue
        de=float(m['d_e'])
        for p,d in r.get('boundary_diagnostics',{}).items():
            t=d['reference']['contact_flip_rate']
            if not (0<=t<=1):raise ValueError('Invalid contact flip label')
            items.append({'scene':r['scene_id'],'pair':p,'true_flip':int(t>contact_flip_min),
                          'ref_flip_probability_sample':float(t),
                          'source_flip':int(d['source']['contact_flip_rate']>contact_flip_min),
                          'd_e':de,'margin':float(d['anchor_min_abs_clearance_m']),
                          'guard':float(d['anchor_min_abs_brake_guard_mps2'])})
    return items


def _screen(item,lam_margin,lam_guard):
    # A screening proposal, never a proof. A candidate whose anchor guard or
    # clearance is close to its candidate-dependent envelope is recomputed.
    return (item['margin']>lam_margin*item['d_e']
            and item['guard']>lam_guard*item['d_e'])


def _stats(items,lam_margin,lam_guard):
    accept=[x for x in items if _screen(x,lam_margin,lam_guard)]
    flips=[x for x in items if x['true_flip']]
    missed=[x for x in accept if x['true_flip']]
    return {'pair_comparisons':len(items),'reference_contact_flip_pairs':len(flips),
            'proposed_reused_pairs':len(accept),
            'reuse_fraction':len(accept)/len(items) if items else None,
            'missed_flip_pairs':len(missed),
            'miss_fraction_given_reference_flip':len(missed)/len(flips) if flips else None,
            'missed_flip_mass_sum':float(sum(x['ref_flip_probability_sample'] for x in missed)),
            'source_model_missed_reference_flip_fraction':(
                sum(x['source_flip']==0 for x in flips)/len(flips) if flips else None)}


def _cp_upper(k,n,confidence=.95):
    if n==0:return None
    from scipy.stats import beta
    return 1. if k==n else float(beta.ppf(confidence,k+1,n-k))


def study(dev_file,test_file,target_miss=.05):
    if not 0<=target_miss<1:raise ValueError('Invalid miss threshold')
    dev=list(read_records(dev_file));test=list(read_records(test_file))
    if not dev or not test or any(r.get('experimental_split') not in ('train','calibration') for r in dev) or any(r.get('experimental_split')!='test' for r in test):
        raise ValueError('Require independent train/calibration development and test records')
    if {x['scene_id'] for x in dev}&{x['scene_id'] for x in test}:
        raise ValueError('Train/test scene-ID overlap')
    def g(r):return r.get('scene',{}).get('provenance',{}).get('overlap_group',r['scene_id'])
    if {g(x) for x in dev}&{g(x) for x in test}:
        raise ValueError('Train/test log-map-group overlap')
    training=_collect(dev);evaluation=_collect(test)
    if not training or not evaluation:raise ValueError('No comparable pairs')
    parameters=[(lm,lg) for lm in (0.,.5,1.,2.,4.,8.,12.,24.,48.)
                for lg in (0.,.25,.5,1.,2.,4.,8.)]
    eligible=[]
    for lm,lg in parameters:
        st=_stats(training,lm,lg)
        if st['reference_contact_flip_pairs'] and st['miss_fraction_given_reference_flip']<=target_miss:
            eligible.append((st['reuse_fraction'],lm,lg))
    if eligible:
        _,lm,lg=max(eligible,key=lambda x:(x[0],-x[1],-x[2]))
        status='FITTED_EMPIRICAL_SCREENER_NOT_CERTIFIED'
    else:
        # Avoid quietly admitting uncertified reuse when no development event
        # flips were seen or every candidate threshold misses some.
        lm,lg=math.inf,math.inf
        status='NO_SUPPORTED_EMPIRICAL_REUSE_RULE'
    test_st=_stats(evaluation,lm,lg)
    dev_st=_stats(training,lm,lg)
    # This uncertainty quantifies sample variability of the *declared audit*
    # population only, not shifts or true driver counterfactuals.
    test_st['clopper_pearson_95pct_upper_miss_fraction']=_cp_upper(
        test_st['missed_flip_pairs'],test_st['reference_contact_flip_pairs'])
    return {'status':status,'development_split':sorted({r.get('experimental_split') for r in dev}),
            'prediction_scope':'EMPIRICAL_SCREEN_ONLY_CERTIFIED_BETA_DEFAULTS_ONE',
            'screen_definition':'DIMENSIONLESS_CLEARANCE_GUARD_HEURISTIC_NOT_LATENCY_GUARD',
            'training_pairs':len(training),'test_pairs':len(evaluation),
            'frozen_thresholds':{'clearance_multiplier':lm if math.isfinite(lm) else None,
                                 'brake_guard_multiplier':lg if math.isfinite(lg) else None},
            'development':dev_st,'heldout':test_st,
            'target_train_miss_fraction':target_miss,
            'train_sha256':sha256(Path(dev_file).read_bytes()).hexdigest(),
            'test_sha256':sha256(Path(test_file).read_bytes()).hexdigest(),
            'not_claimed':['No verified guard Lipschitz, complete path graph, or swept geometry tolerance',
                           'The fitted multipliers compare unlike units and are not a substitute for latency-adjusted occupancy margin',
                           'No real-world conditional safety or real-time speedup certificate',
                           'A source-model mode flip does not imply reference-model contact flip',
                           'Tuning thresholds on the held-out output invalidates this study']}
