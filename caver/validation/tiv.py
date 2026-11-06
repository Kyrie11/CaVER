"""TIV: query proposals, source audit provenance, and offline value estimation.

Acquisition never grants safety certification. Final calibration after adaptive
queries needs a separate, frozen release-state calibration protocol.
"""
from dataclasses import dataclass,asdict
from typing import Any
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn

@dataclass(frozen=True)
class Query:
    candidates: tuple[str,...]
    pairs: tuple[str,...]
    chain: tuple[str,...]
    window: tuple[float,float]
    source: str
    samples: int
    cost: float
    presumed_latency_ms: float

    def json(self):return asdict(self)

SOURCE_INFO={
    'more_source_samples':{'removes':'Monte Carlo noise','cannot_remove':'shared systematic model bias'},
    'full_candidate_reprediction':{'removes':'anchor reuse and local approximation error','cannot_remove':'base response law bias'},
    'independent_late_brake':{'removes':'some response structural disagreement','cannot_remove':'common missing driver regimes'},
    'independent_conservative':{'removes':'some response structural disagreement','cannot_remove':'common missing driver regimes'},
    'physics_envelope':{'removes':'uncertainty inside known bounded dynamics','cannot_remove':'unmodeled drivers or discontinuities'},
    'fresh_observation':{'removes':'latent driver-response uncertainty under a safe holding prefix','cannot_remove':'late observations / irreversible hazards'}
}

def propose(decision,pred,actions,pairs,budget=12,deadline_ms=100.,time_now=0.):
    """Predeclared greedy heuristic: near-boundary pairs for runner-up actions.

    Query optimizer is trained separately by the signed EVoV model below.
    The query is not run here, and its 'value' is NOT a risk certificate.
    """
    if decision.status=='DECISION-CERTIFIED' or decision.released:
        return []
    # LP witness priority: gate violations and plausible-safe regret competitor.
    # A pair absent from an LP witness must not be silently called causal.
    implicated=[]
    witness_candidates=set()
    for key,w in getattr(decision,'witnesses',{}).items():
        if '/gate/' in key and float(w.get('violation',0.))>0:
            witness_candidates.add(key.split('/')[0])
            typ=w.get('constraint','')
            if typ.startswith('increment:'):
                implicated.append(typ.split(':',1)[1])
        if key.endswith('/plausible_regret') and w.get('competitor'):
            witness_candidates.add(key.split('/')[0]);witness_candidates.add(w['competitor'])
    ranked=sorted(actions,key=lambda a:(a not in witness_candidates,
                     decision.margins.get(a,{}).get('downstream_upper',1.),a))
    cand=tuple(ranked[:min(2,len(ranked))])
    active=list(dict.fromkeys(p for p in implicated if p in pairs))
    if not active:
        # No validated event-onset time? Return a broad, explicitly unlocalized
        # candidate/pair query rather than manufacturing a detector onset.
        active=list(pairs)
    chain=tuple(['ego']+sorted({x for p in active for x in p.split('|')}))
    q=[]
    for source,samples,cost,latency in [
       ('more_source_samples',8,8.,18.),('full_candidate_reprediction',4,9.,40.),
       ('independent_late_brake',4,10.,65.),('independent_conservative',4,10.,68.)]:
        if cost<=budget and latency<=deadline_ms:
            q.append(Query(cand,tuple(active),chain,(time_now,max(time_now+1.,time_now+4.)),source,samples,cost,latency))
    return q

TIV_FEATURE_SCHEMA=(
    'pair_risk_max','pair_risk_min','pair_risk_spread',
    'downstream_upper_min','ego_upper_min','increment_upper_max',
    'regret_min','regret_spread','candidate_count','pair_count',
    'chain_count','sample_count','query_cost','query_latency','source_id',
    'witness_count')


class VerificationValueHead(nn.Module):
    def __init__(self,input_dim=len(TIV_FEATURE_SCHEMA)):
        super().__init__();self.input_dim=int(input_dim)
        self.net=nn.Sequential(nn.Linear(self.input_dim,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def query_features(decision,pred,q):
    # Missing coordinates are unresolved evidence. Treat them as the
    # conservative upper end used by the DSE LP instead of as zero risk.
    pp=[float(pred.get(f'{a}/{p}',1.)) for a in q.candidates for p in q.pairs]
    mar=[float(decision.margins.get(a,{}).get('downstream_upper',1.)) for a in q.candidates]
    ego=[float(decision.margins.get(a,{}).get('ego_upper',1.)) for a in q.candidates]
    inc=[float(decision.margins.get(a,{}).get('max_increment_upper',1.)) for a in q.candidates]
    regrets=[float(decision.worst_regret.get(a,5.)) for a in q.candidates]
    down=[float(pred.get(f'{a}/down',1.)) for a in q.candidates]
    src=['more_source_samples','full_candidate_reprediction','independent_late_brake','independent_conservative',
         'physics_envelope','fresh_observation'].index(q.source)
    return np.asarray([max(pp,default=1.),min(pp,default=1.),
                       max(pp,default=1.)-min(pp,default=1.),
                       min(mar,default=1.),min(ego,default=1.),max(inc,default=1.),
                       min(regrets,default=5.),max(regrets,default=5.)-min(regrets,default=5.),
                       len(q.candidates)/10,len(q.pairs)/10,len(q.chain)/10,
                       q.samples/50,q.cost/20,q.presumed_latency_ms/200,src/6,
                       len(getattr(decision,'witnesses',{}))/20],np.float32)

def learn_query_value(rows,output,epochs=120):
    """Rows from two independently evaluated *same scene* policy branches.

    Labels signed: [loss(no-query)-loss(query)-holding-cost-computation-cost].
    These labels are simulator-conditional and must be separated from final test.
    """
    if len(rows)<6:raise ValueError('Insufficient independent offline query audits')
    X=torch.tensor(np.asarray([r['features'] for r in rows],np.float32));Y=torch.tensor(np.asarray([r['signed_value'] for r in rows],np.float32))
    m=VerificationValueHead();opt=torch.optim.Adam(m.parameters(),lr=8e-4)
    for ep in range(epochs):
        pred=m(X);loss=torch.nn.functional.smooth_l1_loss(pred,Y)
        opt.zero_grad();loss.backward();opt.step()
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    torch.save({'weights':m.state_dict(),'data_scope':'OFFLINE_SIMULATED_SOURCE_AUDIT','n':len(rows),
                'feature_schema':list(TIV_FEATURE_SCHEMA)},output)
    return {'n':len(rows),'training_loss':float(loss.detach()),'output':str(output),
            'feature_schema':list(TIV_FEATURE_SCHEMA)}


def verify_source(scene,actions,q,seed=100):
    """Surrogate query execution; no physical-range proof source implemented."""
    import time
    from caver.sim.traffic import sample_world,rollout
    start=time.perf_counter()
    fam={'more_source_samples':'delayed_idm','full_candidate_reprediction':'delayed_idm',
         'independent_late_brake':'late_brake','independent_conservative':'conservative'}
    if q.source not in fam:
        return {'source':q.source,'status':'NOT_IMPLEMENTED_FOR_SCENE','can_recertify':False,
                'reason':'Requires independent physics envelope or real new observation'}
    actions={a.action_id:a for a in actions};outs=[]
    for a_id in q.candidates:
        for k in range(q.samples):
            r=rollout(scene,actions[a_id],sample_world(scene,seed+k,fam[q.source]))
            outs.append({'action':a_id,'seed':seed+k,'family':fam[q.source],
                         'downstream':r['downstream_collision'],
                         'pairs':{p:r['contact'].get(p,0) for p in q.pairs},
                         'margin':{p:r['min_clearance'].get(p) for p in q.pairs}})
    measured_ms=(time.perf_counter()-start)*1000.
    return {'source':q.source,'status':'SIMULATED_QUERY','can_recertify':False,'outcomes':outs,
            'measured_walltime_ms':measured_ms,'presumed_latency_ms':q.presumed_latency_ms,
            'source_independent_of_original_model':q.source in ('independent_late_brake','independent_conservative'),
            'structurally_independent_of_all_synthetic_reactors':False,
            'warning':'Query changes evidence transcript; final acceptance requires independent release-state calibration'}


def select_query(decision, pred, candidates, value_head=None, budget=12., deadline_ms=100.,
                 holding_prefix_verified=False, protocol_frozen=True):
    """One-shot adaptive query choice BEFORE terminal certification.

    A learned value head predicts signed execution-loss improvement; it NEVER
    certifies safety. If no separately checked hold/prefix, no query can delay
    the driving decision. A missing head is an audit-only ranking, NOT a VOI fit.
    """
    if not holding_prefix_verified or not protocol_frozen:
        return {'query':None,'reason':'holding prefix or frozen acquisition policy not verified'}
    if getattr(decision,'released',False):
        return {'query':None,'reason':'candidate already released'}
    allowed=[q for q in candidates if q.cost<=budget and q.presumed_latency_ms<=deadline_ms]
    if not allowed:
        return {'query':None,'reason':'deadline or budget disallows all queries'}
    if value_head is None:
        return {'query':None,'reason':'no offline learned signed query value; refuse entropy-only action'}
    scores=[]
    value_head.eval()
    with torch.no_grad():
        for q in allowed:
            features=torch.tensor(query_features(decision,pred,q)[None,:])
            score=float(value_head(features)[0].item())
            scores.append((score,q))
    score,q=max(scores,key=lambda x:x[0])
    if not np.isfinite(score) or score<=0:
        return {'query':None,'reason':'nonpositive predicted signed VOI','best_value':score}
    return {'query':q,'predicted_signed_value':score,
            'not_a_safety_certificate':True,'must_recalibrate_after_query':True}


def load_query_value(path):
    try: data=torch.load(path,map_location='cpu',weights_only=True)
    except TypeError: data=torch.load(path,map_location='cpu')
    if data.get('data_scope')!='OFFLINE_SIMULATED_SOURCE_AUDIT':
        raise ValueError('Unrecognized TIV acquisition model provenance')
    weight=data['weights'].get('net.0.weight')
    input_dim=int(weight.shape[1]) if weight is not None else len(TIV_FEATURE_SCHEMA)
    m=VerificationValueHead(input_dim=input_dim);m.load_state_dict(data['weights']);m.eval()
    return m


def audit_query_value(rows, value_head=None):
    """Score signed query-value predictions on a disjoint audit transcript.

    The labels are simulator-conditional execution deltas.  This diagnostic
    reports value estimation only and never converts a positive score into a
    safety or release certificate.
    """
    rows=list(rows)
    if value_head is None or not rows:
        return {'status':'NOT_MEASURED','n':len(rows),'reason':'missing value head or audit rows'}
    value_head.eval()
    X=torch.tensor(np.asarray([r['features'] for r in rows],np.float32))
    y=np.asarray([float(r['signed_value']) for r in rows],dtype=float)
    with torch.no_grad():
        pred=value_head(X).cpu().numpy().astype(float)
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(y)):
        return {'status':'INVALID','n':len(rows),'reason':'nonfinite value prediction or label'}
    mae=float(np.mean(np.abs(pred-y)))
    rmse=float(np.sqrt(np.mean((pred-y)**2)))
    corr=None
    if len(rows)>1 and np.std(pred)>0. and np.std(y)>0.:
        corr=float(np.corrcoef(pred,y)[0,1])
    # Grouped top-1 selection checks whether the learned head identifies the
    # best audited source within each scene, without claiming a causal gain.
    groups={}
    for i,row in enumerate(rows):
        groups.setdefault(str(row.get('scene_id',i)),[]).append(i)
    top1=[]
    for idx in groups.values():
        if len(idx)<2:continue
        top1.append(float(idx[int(np.argmax(pred[idx]))]==idx[int(np.argmax(y[idx]))]))
    return {'status':'MEASURED','n':len(rows),'scene_groups':len(groups),
            'signed_value_mae':mae,'signed_value_rmse':rmse,
            'signed_value_pearson':corr,
            'actual_positive_rate':float(np.mean(y>0.)),
            'predicted_positive_rate':float(np.mean(pred>0.)),
            'scene_top1_best_source_rate':float(np.mean(top1)) if top1 else None,
            'label_scope':'SIMULATOR_CONDITIONAL_OFFLINE_QUERY_AUDIT',
            'query_is_certificate':False}
