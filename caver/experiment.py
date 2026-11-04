"""Paired-action intervention audit and full CBC/DSE/TIV research harness."""
from pathlib import Path
from dataclasses import asdict
from hashlib import sha256
import json,math,time
import numpy as np
import torch
from caver.sim.traffic import standard_fourcar,candidates,Action,sample_world,rollout
from caver.models.cbc import all_downstream_pairs,pair_features,predict,predict_joint,coherent_project,training_loss,CascadeRiskHead,JointRiskHead,save,load
from caver.validation.evidence import conformal_calibrate,Calibration,choose
from caver.validation.tiv import propose,verify_source,query_features,learn_query_value


def make_scene(seed):
    rng=np.random.default_rng(seed)
    sc=standard_fourcar(offset=float(rng.normal(0,1.8)),speed=float(rng.uniform(13.,18.)))
    # Train, calibration and testing scene seeds are disjoint in the run protocol.
    for ag in sc.agents:
        if ag.agent_id=='E':ag.x+=float(rng.uniform(-3.,3.))
        if ag.agent_id=='A':ag.x+=float(rng.uniform(-6.,6.))
        if ag.agent_id=='C':ag.x+=float(rng.uniform(-4.,4.))
    sc.scene_id=f'synth-{seed:08d}'
    return sc

def _pair_target(rollouts,p):
    return [float(np.mean([r['contact'].get(p,0) for r in rollouts])),
            float(np.mean([r['min_clearance'].get(p,50.) for r in rollouts])),
            (float(np.mean([r['stop_slack_event'][p] for r in rollouts if r.get('stop_slack_valid',{}).get(p,False)]))
             if any(r.get('stop_slack_valid',{}).get(p,False) for r in rollouts) else 0.),
            float(np.mean([bool(r.get('stop_slack_valid',{}).get(p,False)) for r in rollouts]))]

def paired_scene(scene,actions=None,source_family='delayed_idm',audit_families=('late_brake','conservative'),
                 source_worlds=4,audit_worlds=16,seed=0,model=None,source_policy=None):
    """Produces ex ante features and HELD OUT simulated reference targets.

    Pair coupling is by identical seed+family only. Never interprets cross-world
    joint transitions as observed individual causal frequencies.
    """
    actions=actions or candidates();pairs=all_downstream_pairs(scene)
    source_by_action={};ref_by_action={};prediction={};reference={};diagnostics={}
    # Same latent world seeds for all candidate actions within each family.
    for a in actions:
        src=[rollout(scene,a,sample_world(scene,seed+k,source_family),response_net=source_policy) for k in range(source_worlds)]
        ref=[rollout(scene,a,sample_world(scene,100000+seed+k,fam)) for fam in audit_families for k in range(audit_worlds)]
        source_by_action[a.action_id]=src;ref_by_action[a.action_id]=ref
        prediction[f'{a.action_id}/ego']=float(np.mean([r['ego_collision'] for r in src]))
        prediction[f'{a.action_id}/down']=float(np.mean([r['downstream_collision'] for r in src]))
        reference[f'{a.action_id}/ego']=float(np.mean([r['ego_collision'] for r in ref]))
        reference[f'{a.action_id}/down']=float(np.mean([r['downstream_collision'] for r in ref]))
        for p in pairs:
            x=pair_features(scene,a,src,p)
            prediction[f'{a.action_id}/{p}']=predict(model,x)['p']
            reference[f'{a.action_id}/{p}']=_pair_target(ref,p)[0]
            diagnostics[f'{a.action_id}/{p}']={'features':x.tolist(),
                   'target':_pair_target(ref,p),'baseline':_pair_target(src,p)}
    prediction=coherent_project(prediction,[a.action_id for a in actions],pairs)
    return {'scene_id':scene.scene_id,'scene':scene.to_json(),'source_family':source_family,'audit_families':list(audit_families),
            'source_worlds':source_worlds,'audit_worlds_per_family':audit_worlds,
            'actions':[asdict(a) for a in actions],'pairs':pairs,'predicted':prediction,'audited':reference,
            'diagnostics':diagnostics,'reference_label':'SIMULATOR_CONDITIONAL_ONLY'}

def fit_cbc(records,output,epochs=75,seed=11):
    torch.manual_seed(seed)
    records=list(records)
    if any(r.get('experimental_split') not in ('train',None) for r in records):
        raise ValueError('CBC train data must not contain calibration/test rollouts')
    X=[];Y=[];scene_groups=[]
    for row in records:
        pairs=row['pairs']
        for a in [v['action_id'] for v in row['actions']]:
            features=[row['diagnostics'][f'{a}/{p}']['features'] for p in pairs]
            if features:
                scene_groups.append((torch.tensor(np.asarray(features,np.float32)),
                        torch.tensor([row['audited'][a+'/ego'],row['audited'][a+'/down']],dtype=torch.float32)))
        for v in row['diagnostics'].values():X.append(v['features']);Y.append(v['target'])
    if len(X)<6 or len(scene_groups)<2:raise ValueError('Need >=6 pairs and >=2 joint candidate labels')
    X=torch.tensor(np.array(X,np.float32));Y=torch.tensor(np.array(Y,np.float32))
    m=CascadeRiskHead();m.joint=JointRiskHead()
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
    for ep in range(epochs):
        perm=torch.randperm(len(X));groups=torch.randperm(len(scene_groups)).tolist();losses=[]
        for b in perm.split(128):
            z=m(X[b]);loss=training_loss(z,Y[b]);opt.zero_grad();loss.backward();
            torch.nn.utils.clip_grad_norm_(m.parameters(),3.);opt.step();losses.append(float(loss.detach()))
        # Joint events are supervised from complete independent rollouts.
        # Do NOT sum independent pair probabilities to create union labels.
        for sidx in groups:
            feat,y=scene_groups[sidx]
            output_joint=m.joint(feat)
            loss_joint=torch.nn.functional.binary_cross_entropy(output_joint.clamp(1e-6,1-1e-6),y)
            opt.zero_grad();loss_joint.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),3.);opt.step()
            losses.append(float(loss_joint.detach()))
    save(m,output,{'training_scenes':len(records),'example_count':len(X),'joint_examples':len(scene_groups),
                   'train_source':'simulator_reference','seed':seed,'slack_masked':True})
    return {'output':str(output),'train_scenes':len(records),'pairs':len(X),
            'joint_examples':len(scene_groups),'loss':float(np.mean(losses))}

def record_with_new_predictions(row,model):
    pred=dict(row['predicted'])
    diagnostics={k:dict(v) for k,v in row['diagnostics'].items()}
    names=[a['action_id'] for a in row['actions']]
    for a in names:
        feat=[]
        for pair in row['pairs']:
            v=row['diagnostics'][f'{a}/{pair}']
            feat.append(v['features'])
            estimate=predict(model,v['features'])
            pred[f'{a}/{pair}']=estimate['p']
            diagnostics[f'{a}/{pair}']['cbc_margin_prediction_m']=estimate['margin']
            diagnostics[f'{a}/{pair}']['cbc_event_slack_prediction_m']=estimate['slack']
        if feat:
            j=predict_joint(model,feat)
            pred[f'{a}/ego']=j['ego']
            pred[f'{a}/down']=j['down']
    pred=coherent_project(pred,names,row['pairs'])
    return {**row,'predicted':pred,'diagnostics':diagnostics,
            'calibrated_prediction_scope':'CBC_JOINT_POSTPROJECTION'}

def store_records(records,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w') as f:
        for r in records:f.write(json.dumps(r,allow_nan=False,separators=(',',':'))+'\n')

def read_records(path):
    with open(path) as f:
        for ln in f:
            if ln.strip():yield json.loads(ln)

def evaluate(records,calibration=None,eps_down=.7,eps_ego=.7,eps_inc=.8,eps_reg=8.,strict_prefix=False):
    outputs=[]
    for row in records:
        acts=[a['action_id'] for a in row['actions']]
        # Reference action deliberately low task progress: task cost controls tradeoff.
        task={a:(.5 if a=='reference' else 0.) for a in acts}
        d=choose(row['predicted'],acts,row['pairs'],'reference',calibration,
                 eps_down=eps_down,eps_ego=eps_ego,eps_inc=eps_inc,
                 eps_reg=eps_reg,task_cost=task,strict_prefix=strict_prefix)
        selected=d.chosen
        ref=row['audited']
        truth_safe={a for a in acts if ref[f'{a}/ego']<=eps_ego and ref[f'{a}/down']<=eps_down}
        if selected is not None:
            regret=max(0.,ref[f'{selected}/down']-min([ref[f'{a}/down'] for a in truth_safe] or [1.]))
        else:regret=None
        outputs.append({'scene_id':row['scene_id'],'status':d.status,'selected':selected,
                        'admitted':d.candidates,'gamma_R':d.gamma_robust,'gamma_P':d.gamma_plausible,
                        'worst_witnesses':d.witnesses,'regret_on_audited_reactor':regret,
                        'reference_downstream_risk':ref.get(f'{selected}/down') if selected is not None else None,
                        'accepted_wrong':bool(selected is not None and selected not in truth_safe),
                        'evidence_coverage':all((calibration.interval(row['predicted'][k])[0]<=ref[k]<=calibration.interval(row['predicted'][k])[1]) for k in ref) if calibration else None,
                        'reason':d.reason,'warnings':['simulator conditional','strict_prefix='+str(strict_prefix)]})
    n=len(outputs);accepted=[x for x in outputs if x['selected']]
    risk_admitted=[x for x in outputs if x['admitted']]
    decision_certified=[x for x in outputs if x['status']=='DECISION-CERTIFIED']
    return {'scenes':n,'accepted':len(accepted),'abstain':n-len(accepted),
            'risk_admissible_scene_coverage':len(risk_admitted)/n if n else None,
            'decision_certified_scene_coverage':len(decision_certified)/n if n else None,
            'risk_admissible_scenes':len(risk_admitted),'decision_certified_scenes':len(decision_certified),
            'audit_wrong_choice_rate_accepted':float(np.mean([r['accepted_wrong'] for r in accepted])) if accepted else None,
            'audit_decision_regret_accepted':float(np.mean([r['regret_on_audited_reactor'] for r in accepted])) if accepted else None,
            'joint_interval_coverage':float(np.mean([r['evidence_coverage'] for r in outputs])) if calibration else None,
            'rows':outputs,'independent_field_validation':False}

def synthetic_records(count=10,start=100,source_worlds=2,audit_worlds=4,model=None):
    return [paired_scene(make_scene(start+i),source_worlds=source_worlds,audit_worlds=audit_worlds,
                         seed=1000+i*23,model=model) for i in range(count)]

def single_demo(out,seed=10,worlds=4):
    Path(out).mkdir(parents=True,exist_ok=True)
    r=paired_scene(make_scene(seed),source_worlds=worlds,audit_worlds=worlds,seed=seed)
    (Path(out)/'demo_record.json').write_text(json.dumps(r,indent=2))
    evidence=choose(r['predicted'],[a['action_id'] for a in r['actions']],r['pairs'],calibration=None)
    qs=propose(evidence,r['predicted'],[a['action_id'] for a in r['actions']],r['pairs'])
    (Path(out)/'queries.json').write_text(json.dumps([q.json() for q in qs],indent=2))
    return {'demo':'SYNTHETIC_ONLY','out':str(out),'candidate_count':len(r['actions']),
            'pairs':r['pairs'],'candidate_risks':r['predicted'],'uncalibrated_gate':evidence.status,
            'queries_proposed':len(qs),'claims':'No original fourcar fixture or natural WOMD data were executed'}

def mediator_exchange(scene,world,action_a=None,action_b=None,mediator='B'):
    """Structural mediator swap inside frozen surrogate law, not real causal ID."""
    action_a,action_b=(action_a or candidates()[0]),(action_b or candidates()[1])
    x=rollout(scene,action_a,world);y=rollout(scene,action_b,world)
    idx=x['actor_ids'].index(mediator)
    xa=rollout(scene,action_a,world,override_states={mediator:y['trajectory'][:,idx,:]})
    yb=rollout(scene,action_b,world,override_states={mediator:x['trajectory'][:,idx,:]})
    def rec(r):return {'ego_contact':r['ego_collision'],'downstream_contact':r['downstream_collision'],
                      'BC_contact':r['contact'].get('B|C'),'BC_clearance':r['min_clearance'].get('B|C')}
    return {'base_a':rec(x),'base_b':rec(y),'a_with_b_mediator':rec(xa),'b_with_a_mediator':rec(yb),
            'source':'SIMULATOR_STRUCTURAL_INTERVENTION','physical_consistency_replay_not_guaranteed':True}

def frozen_challenge(path='configs/fourcar_challenge.json',outfile=None):
    from caver.sim.traffic import Driver,World
    from caver.data.schema import Scene
    config=json.loads(Path(path).read_text());scene=standard_fourcar()
    lookup={a.agent_id:a for a in scene.agents}
    for k,v in config['actors'].items():
        for field,x in v.items():setattr(lookup[k],field,x)
    scene.scene_id='fourcar-posthoc-mechanism'
    scene.provenance.update({'challenge_source':config['provenance'],'posthoc_iteration':config['search_iteration']})
    acts=[Action(**a) for a in config['actions']]
    # A is far ahead, but its policy is fixed across interventions.
    w=sample_world(scene,51)
    for k,p in config['policies'].items():w.driver[k]=Driver(**p)
    out=[rollout(scene,a,w,horizon=config['horizon_seconds'],dt=config['dt_seconds']) for a in acts]
    med=mediator_exchange(scene,w,acts[0],acts[1],'B')
    result={'fixture':config['provenance'],'config':str(path),
            'actions':{a.action_id:{'ego_collision':bool(r['ego_collision']),
                         'downstream_collision':bool(r['downstream_collision']),
                         'BC_collision':bool(r['contact']['B|C']),
                         'BC_clearance':r['min_clearance']['B|C'],
                         'ego_B_clearance':r['min_clearance']['B|E'],
                         'response_onset':r['response_onset']} for a,r in zip(acts,out)},
            'B_mediator_exchange':med,
            'action_max_position_distance_m':float(np.max(np.linalg.norm(out[0]['trajectory'][:,0,:2]-out[1]['trajectory'][:,0,:2],axis=-1))),
            'limitations':['posthoc manually searched surrogate-only fixture','no CARLA physics','no independent reference model test']}
    if outfile:
        Path(outfile).parent.mkdir(parents=True,exist_ok=True);Path(outfile).write_text(json.dumps(result,indent=2))
    return result
