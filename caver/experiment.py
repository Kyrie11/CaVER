"""Paired-action intervention audit and full CBC/DSE/TIV research harness."""
from pathlib import Path
from dataclasses import asdict
from hashlib import sha256
import json,math,time,gzip
import numpy as np
import torch
from caver.sim.traffic import standard_fourcar,candidates,Action,sample_world,rollout
from caver.models.cbc import FEATURES,all_downstream_pairs,pair_features,predict,predict_joint,coherent_project,training_loss,CascadeRiskHead,JointRiskHead,save,load
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



def stress_scene(seed, config='configs/fourcar_challenge.json'):
    """Pre-registered *stress law*: perturbs posthoc fixture before outcomes.

    Never filter by realized contact/choice labels. RQ1 natural prevalence
    must be computed on unselected WOMD, not on this synthetic stress law.
    """
    rng=np.random.default_rng(seed)
    cfg=json.loads(Path(config).read_text())
    sc=standard_fourcar()
    lookup={a.agent_id:a for a in sc.agents}
    for k,st in cfg['actors'].items():
        for attr,value in st.items():
            noise=1.8 if attr=='x' and k in ('E','C') else (.8 if attr=='x' else .4 if attr=='vx' else 0.)
            setattr(lookup[k],attr,float(value)+float(rng.normal(0,noise)))
    sc.scene_id=f'synth-stress-{seed:08d}'
    sc.provenance.update({'stress_sampled_before_outcomes':True,
       'stress_center':'POSTHOC_FROZEN_FOURCAR_FIXTURE',
       'causal_scope':'SYNTHETIC_POLICY_CONDITIONAL_NOT_NATURAL_FREQUENCY'})
    return sc.validate()


def stress_actions():
    return [Action('early',.5,1.6,1.8,accel_ramp_s=.40),
            Action('late',1.9,3.,1.8,accel_ramp_s=.40),
            Action('reference',9.,10.,0.,lane_offset=0.)]

def _pair_target(rollouts,p):
    return [float(np.mean([r['contact'].get(p,0) for r in rollouts])),
            float(np.mean([r['min_clearance'].get(p,50.) for r in rollouts])),
            (float(np.mean([r['stop_slack_event'][p] for r in rollouts if r.get('stop_slack_valid',{}).get(p,False)]))
             if any(r.get('stop_slack_valid',{}).get(p,False) for r in rollouts) else 0.),
            float(np.mean([bool(r.get('stop_slack_valid',{}).get(p,False)) for r in rollouts]))]



def action_match_audit(actions,source_by_action,ego_index,dt,
                       position_scale=5.,speed_scale=4.,accel_scale=3.,
                       d_max=1.8,task_diff_max=3.,jerk_limit=5.,acc_limit=3.,lateral_acc_limit=3.):
    """Pre-outcome task/dynamics checks from PRESCRIBED ego controls only.

    These statistics do not use other actors or reference contact labels.
    Traffic-law and map-topology feasibility remain separately UNVERIFIED.
    """
    profiles={}
    by_id={a.action_id:a for a in actions}
    for a in actions:
        tr=np.asarray(source_by_action[a.action_id][0]['trajectory'],float)[:,ego_index,:]
        v=tr[:,2]
        accel=np.diff(v)/dt
        j=np.diff(accel)/dt
        y=tr[:,1]
        lat_acc=np.diff(y,n=2)/(dt*dt)
        # Task cost deliberately independent of non-ego safety and audit outcomes.
        task=float(-(tr[-1,0]-tr[0,0])/10. + .04*np.sum(accel**2)*dt)
        profiles[a.action_id]={'progress_m':float(tr[-1,0]-tr[0,0]),
            'task_cost_ex_ante':task,'max_abs_long_jerk_mps3':float(np.max(np.abs(j))) if len(j) else 0.,
            'max_abs_long_accel_mps2':float(np.max(np.abs(accel))) if len(accel) else 0.,
            'max_abs_lateral_accel_mps2':float(np.max(np.abs(lat_acc))) if len(lat_acc) else 0.,
            'kinematics_feasible':bool((len(j)==0 or np.max(np.abs(j))<=jerk_limit+1e-4)
                and (len(accel)==0 or np.max(np.abs(accel))<=acc_limit+1e-4)
                and (len(lat_acc)==0 or np.max(np.abs(lat_acc))<=lateral_acc_limit+1e-4)),
            'map_law_route_feasibility':'UNVERIFIED'}
    matched={}
    for i,a in enumerate(actions):
        for b in actions[i+1:]:
            ta=np.asarray(source_by_action[a.action_id][0]['trajectory'])[:,ego_index,:]
            tb=np.asarray(source_by_action[b.action_id][0]['trajectory'])[:,ego_index,:]
            va,vb=np.diff(ta[:,2])/dt,np.diff(tb[:,2])/dt
            metric=float(np.mean(np.linalg.norm(ta[:,:2]-tb[:,:2],axis=-1)/position_scale
                      +np.abs(ta[:,2]-tb[:,2])/speed_scale
                      +np.r_[np.abs(va-vb)/accel_scale,0.]))
            comparable=bool(abs(a.lane_offset-b.lane_offset)<.1
                            and abs(a.lane_time-b.lane_time)<.1)
            task_diff=abs(profiles[a.action_id]['task_cost_ex_ante']-
                          profiles[b.action_id]['task_cost_ex_ante'])
            pair_key='|'.join(sorted((a.action_id,b.action_id)))
            matched[pair_key]={'same_maneuver_intent':comparable,'d_e':metric,
                 'delta_e_max':d_max,'task_cost_difference':task_diff,
                 'delta_J_max':task_diff_max,
                 'kinematic_feasible_both':profiles[a.action_id]['kinematics_feasible']
                                    and profiles[b.action_id]['kinematics_feasible'],
                 'matched_kinematic_proxy':bool(comparable and metric<=d_max
                     and task_diff<=task_diff_max and profiles[a.action_id]['kinematics_feasible']
                     and profiles[b.action_id]['kinematics_feasible']),
                 'lawful_route_match':'NOT_EVALUATED'}
    return {'profiles':profiles,'pairs':matched,'computed_from':'PRESCRIBED_EGO_ONLY_PRE_REFERENCE_LABELS'}



def paired_boundary_diagnostics(source_by_action,ref_by_action,pairs,actions,dt=.1):
    """Ex-post paired-world diagnostics, deliberately *not* CBC input.

    All differences use same latent world indexes per declared family. Audit
    world labels are never touched by source temporal graph construction.
    """
    if not {'early','late'} <= set(a.action_id for a in actions):
        return {}
    result={}
    for p in pairs:
        src0=source_by_action['early'];src1=source_by_action['late']
        ref0=ref_by_action['early'];ref1=ref_by_action['late']
        def digest(xs,ys):
            if len(xs)!=len(ys) or not xs:
                raise ValueError('Pairing requires same latent sample count')
            contact=[];guard_change=[];change_times=[]
            for x,y in zip(xs,ys):
                if x['world_seed']!=y['world_seed'] or x['family']!=y['family']:
                    raise ValueError('Response latent coupling was not preserved across actions')
                contact.append(int(x['contact'].get(p,0)!=y['contact'].get(p,0)))
                change=False;onset=None
                for agent in x['accelerations']:
                    if agent=='E':continue  # ego response is the prescribed action
                    ba=np.asarray(x['accelerations'][agent],float)<-.6
                    bb=np.asarray(y['accelerations'][agent],float)<-.6
                    disagree=np.flatnonzero(ba!=bb)
                    if len(disagree):
                        change=True
                        onset=float(disagree[0])*dt if onset is None else min(onset,float(disagree[0])*dt)
                guard_change.append(int(change))
                if onset is not None:change_times.append(onset)
            return {'contact_flip_rate':float(np.mean(contact)),
                    'any_brake_guard_flip_rate':float(np.mean(guard_change)),
                    'first_mode_disagreement_mean_s':float(np.mean(change_times)) if change_times else None,
                    'n_paired_latent_worlds':len(xs)}
        d0=src0[0]
        a=np.asarray([z['min_clearance'].get(p,50.) for z in src0],float)
        guarding=[]
        for r in src0:
            for agent,arr in r['accelerations'].items():
                if agent!='E':
                    guarding.extend(np.abs(np.asarray(arr,float)+.6).tolist())
        result[p]={'source':digest(src0,src1),'reference':digest(ref0,ref1),
             'anchor_min_abs_clearance_m':float(np.min(np.abs(a))) if len(a) else None,
             'anchor_min_abs_brake_guard_mps2':float(np.min(guarding)) if guarding else None,
             'event_sources':'PAIRED_STRUCTURAL_INTERVENTION_OUTCOME_LABELS_NOT_ONLINE_EVIDENCE'}
    return result

def paired_scene(scene,actions=None,source_family='delayed_idm',audit_families=('late_brake','conservative'),
                 source_worlds=4,audit_worlds=16,seed=0,model=None,source_policy=None):
    """Produces ex ante features and HELD OUT simulated reference targets.

    Pair coupling is by identical seed+family only. Never interprets cross-world
    joint transitions as observed individual causal frequencies.
    """
    actions=actions or candidates();pairs=all_downstream_pairs(scene)
    # This transcript is intentionally scoped to record generation.  It is
    # useful for finding which component dominates collection cost, but it is
    # not an online planning-callback latency claim.  The online benchmark
    # must capture the same fields at the planner callback boundary.
    t_total=time.perf_counter(); source_ms=0.; reference_ms=0.; feature_ms=0.; model_ms=0.
    source_calls=0; reference_calls=0
    source_by_action={};ref_by_action={};prediction={};reference={};diagnostics={};temporal={}
    # Same latent world seeds for all candidate actions within each family.
    for a in actions:
        t0=time.perf_counter()
        src=[rollout(scene,a,sample_world(scene,seed+k,source_family),response_net=source_policy) for k in range(source_worlds)]
        source_ms+=(time.perf_counter()-t0)*1000.; source_calls+=len(src)
        t0=time.perf_counter()
        ref=[rollout(scene,a,sample_world(scene,100000+seed+k,fam)) for fam in audit_families for k in range(audit_worlds)]
        reference_ms+=(time.perf_counter()-t0)*1000.; reference_calls+=len(ref)
        source_by_action[a.action_id]=src;ref_by_action[a.action_id]=ref
        # Encodes *source reactor predicted* time series only. Auditor target
        # traces remain inaccessible to CBC at training/inference input time.
        from caver.models.temporal import source_temporal_graph
        temporal[a.action_id]=source_temporal_graph(scene,src)
        prediction[f'{a.action_id}/ego']=float(np.mean([r['ego_collision'] for r in src]))
        prediction[f'{a.action_id}/down']=float(np.mean([r['downstream_collision'] for r in src]))
        reference[f'{a.action_id}/ego']=float(np.mean([r['ego_collision'] for r in ref]))
        reference[f'{a.action_id}/down']=float(np.mean([r['downstream_collision'] for r in ref]))
        for p in pairs:
            t0=time.perf_counter()
            x=pair_features(scene,a,src,p)
            feature_ms+=(time.perf_counter()-t0)*1000.
            t0=time.perf_counter()
            prediction[f'{a.action_id}/{p}']=predict(model,x)['p']
            model_ms+=(time.perf_counter()-t0)*1000.
            reference[f'{a.action_id}/{p}']=_pair_target(ref,p)[0]
            diagnostics[f'{a.action_id}/{p}']={'features':x.tolist(),
                   'target':_pair_target(ref,p),'baseline':_pair_target(src,p)}
    prediction=coherent_project(prediction,[a.action_id for a in actions],pairs)
    matching=action_match_audit(actions,source_by_action,[x.agent_id for x in scene.agents].index(scene.ego_id),scene.dt)
    boundary=paired_boundary_diagnostics(source_by_action,ref_by_action,pairs,actions,scene.dt)
    record={'scene_id':scene.scene_id,'scene':scene.to_json(),'source_family':source_family,'audit_families':list(audit_families),
            'source_worlds':source_worlds,'audit_worlds_per_family':audit_worlds,
            'actions':[asdict(a) for a in actions],'pairs':pairs,'predicted':prediction,'audited':reference,
            'diagnostics':diagnostics,'temporal':temporal,'candidate_matching':matching,
            'boundary_diagnostics':boundary,
            'reference_label':'SIMULATOR_CONDITIONAL_ONLY'}
    if 'early' in record['temporal']:
        from caver.models.anchor import build_anchor_view
        record['anchor_view']=build_anchor_view(record,'early')
    record['runtime']={
        'scope':'OFFLINE_RECORD_COLLECTION_WALLCLOCK_NOT_ONLINE_PLANNING_LATENCY',
        'total_ms':(time.perf_counter()-t_total)*1000.,
        'source_ms':float(source_ms),'reference_ms':float(reference_ms),
        'feature_ms':float(feature_ms),'model_ms':float(model_ms),
        'source_rollout_calls':int(source_calls),
        'reference_rollout_calls':int(reference_calls),
        'candidate_count':len(actions),
        'source_recompute_calls':len(actions),
        'anchor_recompute_calls':1,
        'callback_count':None,'reused':None,'fallback':None,
    }
    return record

def fit_cbc(records,output,epochs=75,seed=11,variant='full',device='auto',contrast_weight=0.):
    torch.manual_seed(seed)
    if device=='auto':device='cuda:0' if torch.cuda.is_available() else 'cpu'
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but torch.cuda.is_available() is false; install an appropriate CUDA PyTorch wheel')
    dev=torch.device(device)
    if contrast_weight<0:raise ValueError('contrast_weight must be nonnegative')
    records=list(records)
    if any(r.get('experimental_split') not in ('train',None) for r in records):
        raise ValueError('CBC train data must not contain calibration/test rollouts')
    X=[];Y=[];scene_groups=[]
    uses_anchor=variant.startswith('anchor_')
    uses_graph=variant in ('graph','anchor_graph')
    for row in records:
        pairs=row['pairs']
        if uses_anchor and 'anchor_view' not in row:
            raise ValueError('Anchor model requires one-anchor source evidence')
        for a in [v['action_id'] for v in row['actions']]:
            features=[(row['anchor_view']['candidate_features'][f'{a}/{p}'] if uses_anchor
                       else row['diagnostics'][f'{a}/{p}']['features']) for p in pairs]
            if features:
                scene_groups.append((torch.tensor(np.asarray(features,np.float32)),
                        torch.tensor([row['audited'][a+'/ego'],row['audited'][a+'/down']],dtype=torch.float32)))
        for k,v in row['diagnostics'].items():
            X.append(row['anchor_view']['candidate_features'][k] if uses_anchor else v['features'])
            Y.append(v['target'])
    if len(X)<6 or len(scene_groups)<2:raise ValueError('Need >=6 pairs and >=2 joint candidate labels')
    X=torch.tensor(np.array(X,np.float32),device=dev);Y=torch.tensor(np.array(Y,np.float32),device=dev)
    m=CascadeRiskHead(variant=variant);m.joint=JointRiskHead(suppress_events=(variant=='no_events'))
    if uses_graph:
        from caver.models.temporal import GraphCascadeRiskHead
        m.graph_model=GraphCascadeRiskHead(len(FEATURES))
    m=m.to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
    graph_groups=[]
    if uses_graph:
        for row in records:
            for a in [v['action_id'] for v in row['actions']]:
                pairs=row['pairs']
                graph=(row['anchor_view']['candidate_temporal'][a] if uses_anchor
                       else row.get('temporal',{}).get(a))
                if not graph:
                    raise ValueError('Graph variant requires source-only temporal traces in collected records')
                features=torch.tensor([(row['anchor_view']['candidate_features'][f'{a}/{p}'] if uses_anchor
                                        else row['diagnostics'][f'{a}/{p}']['features']) for p in pairs],dtype=torch.float32)
                targets=torch.tensor([row['diagnostics'][f'{a}/{p}']['target'] for p in pairs],dtype=torch.float32)
                joint=torch.tensor([row['audited'][a+'/ego'],row['audited'][a+'/down']],dtype=torch.float32)
                graph_groups.append((features,targets,joint,graph,pairs))
    for ep in range(epochs):
        perm=torch.randperm(len(X));groups=torch.randperm(len(scene_groups)).tolist();losses=[]
        if uses_graph:
            for idx in torch.randperm(len(graph_groups)).tolist():
                feat,y,joint,trace,pairs=graph_groups[idx]
                feat=feat.to(dev);y=y.to(dev);joint=joint.to(dev)
                pair_pred=m.graph_model(feat,trace,pairs)
                joint_pred=m.joint(feat)
                loss=training_loss(pair_pred,y)+torch.nn.functional.binary_cross_entropy(
                    joint_pred.clamp(1e-6,1-1e-6),joint)
                opt.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),3.);opt.step()
                losses.append(float(loss.detach()))
        else:
            for b in perm.split(128):
                z=m(X[b]);loss=training_loss(z,Y[b]);opt.zero_grad();loss.backward();
                torch.nn.utils.clip_grad_norm_(m.parameters(),3.);opt.step();losses.append(float(loss.detach()))
            # Joint events are supervised from complete independent rollouts.
            for sidx in groups:
                feat,y=scene_groups[sidx]
                feat=feat.to(dev);y=y.to(dev)
                output_joint=m.joint(feat)
                loss_joint=torch.nn.functional.binary_cross_entropy(output_joint.clamp(1e-6,1-1e-6),y)
                opt.zero_grad();loss_joint.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),3.);opt.step()
                losses.append(float(loss_joint.detach()))
        # Optional action-contrastive objective. Reference outcomes appear ONLY
        # in training supervision, never as candidate-selection or test inputs.
        # The same pre-outcome matching mask is used for every model variant.
        if contrast_weight>0:
            for row in records:
                matching=row.get('candidate_matching',{}).get('pairs',{}).get('early|late',{})
                if not matching.get('matched_kinematic_proxy',False):continue
                pair_ids=row['pairs'];deltas=[]
                for act in ('early','late'):
                    feat=torch.tensor(np.asarray([
                        (row['anchor_view']['candidate_features'][f'{act}/{pair}'] if uses_anchor
                        else row['diagnostics'][f'{act}/{pair}']['features']) for pair in pair_ids],np.float32),device=dev)
                    if uses_graph:
                        tr=(row['anchor_view']['candidate_temporal'][act] if uses_anchor
                            else row['temporal'][act])
                        risk=m.graph_model(feat,tr,pair_ids)[:,0]
                    else:
                        risk=m(feat)[:,0]
                    deltas.append(risk)
                estimate=deltas[0]-deltas[1]
                target=torch.tensor([row['audited']['early/'+p]-row['audited']['late/'+p]
                                     for p in pair_ids],dtype=torch.float32,device=dev)
                # Regression is defined even for zero-contrast cases. For
                # substantial reference differences, add a smooth ranking term.
                reg=torch.nn.functional.smooth_l1_loss(estimate,target,beta=.15)
                active=(torch.abs(target)>=.1).float()
                margin=(torch.nn.functional.softplus(.04-
                        torch.sign(target)*estimate)*active).sum()/active.sum().clamp_min(1.)
                loss_ctr=contrast_weight*(reg+.1*margin)
                opt.zero_grad();loss_ctr.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),3.);opt.step()
                losses.append(float(loss_ctr.detach()))
    m=m.to('cpu')
    save(m,output,{'training_scenes':len(records),'example_count':len(X),'joint_examples':len(scene_groups),
                   'train_source':'simulator_reference','variant':variant,'seed':seed,'slack_masked':True,
                   'training_device':str(dev),'contrast_weight':contrast_weight,
                   'prediction_source':'ONE_ANCHOR_REUSE' if uses_anchor else 'FULL_CANDIDATE_CONDITIONED_SOURCE',
                   'training_scene_sha256':[sha256(r['scene_id'].encode()).hexdigest() for r in records],
                   'training_group_sha256':[sha256(str(r.get('scene',{}).get('provenance',{}).get('overlap_group',r['scene_id'])).encode()).hexdigest() for r in records]})
    return {'output':str(output),'train_scenes':len(records),'pairs':len(X),
            'joint_examples':len(scene_groups),'variant':variant,'loss':float(np.mean(losses)),
            'training_device':str(dev),'contrast_weight':contrast_weight}

def record_with_new_predictions(row,model):
    uses_anchor=getattr(model,'variant','').startswith('anchor_')
    if uses_anchor and 'anchor_view' not in row:
        raise ValueError('Anchor prediction needs frozen source anchor evidence')
    pred=dict(row['predicted'])
    diagnostics={k:dict(v) for k,v in row['diagnostics'].items()}
    names=[a['action_id'] for a in row['actions']]
    if uses_anchor:
        cached=row['anchor_view']['candidate_features']
        for a in names:
            for p in row['pairs']:
                pred[f'{a}/{p}']=cached[f'{a}/{p}'][9]
    for a in names:
        feat=[(row['anchor_view']['candidate_features'][f'{a}/{pair}'] if uses_anchor
               else row['diagnostics'][f'{a}/{pair}']['features']) for pair in row['pairs']]
        graph=(row['anchor_view']['candidate_temporal'] if uses_anchor else row.get('temporal',{}))
        if getattr(model,'variant',None) in ('graph','anchor_graph'):
            if a not in graph:
                raise ValueError('Graph predictor has no source temporal evidence')
            with torch.no_grad():
                outs=model.graph_model(torch.tensor(np.asarray(feat),dtype=torch.float32),
                                       graph[a],row['pairs']).cpu().numpy()
            estimates=[{'p':float(o[0]),'margin':float(o[1]),'slack':float(o[2])} for o in outs]
        else:
            estimates=[predict(model,v) for v in feat]
        for pair,estimate in zip(row['pairs'],estimates):
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
    with (gzip.open(p,'wt',encoding='utf-8') if str(p).endswith('.gz') else p.open('w')) as f:
        for r in records:f.write(json.dumps(r,allow_nan=False,separators=(',',':'))+'\n')

def read_records(path):
    with (gzip.open(path,'rt',encoding='utf-8') if str(path).endswith('.gz') else open(path)) as f:
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
