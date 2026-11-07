"""Single entrypoint. Every command records whether it used real or surrogate data."""
from pathlib import Path
import argparse,json,sys,os,hashlib
import numpy as np
from caver.config import Paths

def report(obj,output=None):
    s=json.dumps(obj,ensure_ascii=False,indent=2,default=str)
    if output:
        p=Path(output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s+'\n')
    print(s)

def parser():
    p=argparse.ArgumentParser(prog='caver',description='CaVER CBC-DSE-TIV scientific research harness')
    p.add_argument('--paths',default=None,help='Path JSON, defaults to configs/paths.json-like user server paths')
    sub=p.add_subparsers(dest='cmd',required=True)
    s=sub.add_parser('doctor');s.add_argument('--output')
    s=sub.add_parser('womd-check');s.add_argument('--decode',type=int,default=0);s.add_argument('--split',default='training');s.add_argument('--format',choices=['scenario','tf_example'],default='scenario');s.add_argument('--output')
    s=sub.add_parser('womd-export');s.add_argument('--split',default='training');s.add_argument('--format',choices=['scenario','tf_example'],default='scenario');s.add_argument('--limit',type=int,default=30);s.add_argument('--output',required=True)
    s=sub.add_parser('womd-prep',help='Bounded streaming planning-only WOMD preparation')
    s.add_argument('--split',choices=['training','validation'],required=True)
    s.add_argument('--partition',choices=['train','calibration','test'],required=True)
    s.add_argument('--purpose',choices=['policy','corridor'],default='corridor')
    s.add_argument('--limit',type=int,default=120)
    s.add_argument('--max-scan',type=int,default=40000)
    s.add_argument('--max-shards',type=int,default=100)
    s.add_argument('--per-shard',type=int,default=400)
    s.add_argument('--shard-seed',type=int,default=17)
    s.add_argument('--max-output-mb',type=float,default=45.)
    s.add_argument('--output',required=True,help='MUST end in .jsonl.gz')
    s=sub.add_parser('waymax-read');s.add_argument('--split',default='training');s.add_argument('--limit',type=int,default=2);s.add_argument('--output',required=True)
    s=sub.add_parser('nuplan-check');s.add_argument('--split');s.add_argument('--db',help='Optional one DB strict schema audit');s.add_argument('--output')
    s=sub.add_parser('nuplan-export');s.add_argument('--db',required=True);s.add_argument('--map-root');s.add_argument('--limit',type=int,default=10);s.add_argument('--output',required=True)
    s=sub.add_parser('reactsim-check');s.add_argument('--export',help='Optional verified official simulator JSONL export');s.add_argument('--output')
    s=sub.add_parser('train-policy');s.add_argument('--input');s.add_argument('--womd-train-partition',action='store_true',help='Do not fit observational reactor on frozen calibration scenarios');s.add_argument('--limit',type=int,default=3000);s.add_argument('--epochs',type=int,default=15);s.add_argument('--device',default='auto');s.add_argument('--output',required=True)
    s=sub.add_parser('fourcar');s.add_argument('--out',required=True);s.add_argument('--seed',type=int,default=10);s.add_argument('--worlds',type=int,default=4)
    s=sub.add_parser('fourcar-challenge');s.add_argument('--config',default='configs/fourcar_challenge.json');s.add_argument('--output',required=True)
    s=sub.add_parser('fourcar-mediation');s.add_argument('--out',required=True);s.add_argument('--seed',type=int,default=10);s.add_argument('--world-seed',type=int,default=10);s.add_argument('--family',default='late_brake')
    s=sub.add_parser('fourcar-scan');s.add_argument('--out',required=True);s.add_argument('--scenes',type=int,default=100);s.add_argument('--worlds',type=int,default=3);s.add_argument('--seed',type=int,default=100)
    s=sub.add_parser('collect');s.add_argument('--source',choices=['synthetic','synthetic-stress','womd-jsonl','nuplan-jsonl'],default='synthetic');s.add_argument('--input');s.add_argument('--split',choices=['train','calibration','test'],required=True);s.add_argument('--scenes',type=int,default=20);s.add_argument('--source-worlds',type=int,default=4);s.add_argument('--audit-worlds',type=int,default=16);s.add_argument('--checkpoint');s.add_argument('--source-family',choices=['delayed_idm','learned_logged','stress_source'],default='delayed_idm');s.add_argument('--policy-checkpoint');s.add_argument('--candidate-policy',choices=['fixed','observed-adaptive'],default='fixed');s.add_argument('--stress-profile',choices=['default','boundary'],default='default');s.add_argument('--output',required=True)
    s=sub.add_parser('train-cbc');s.add_argument('--input',required=True);s.add_argument('--variant',choices=['full','no_events','graph','anchor_full','anchor_graph','anchor_graph_event','anchor_graph_event_local','anchor_graph_delay_kernel'],default='full');s.add_argument('--epochs',type=int,default=80);s.add_argument('--device',default='auto');s.add_argument('--contrast-weight',type=float,default=0.);s.add_argument('--rank-weight',type=float,default=0.);s.add_argument('--contrast-threshold',type=float,default=.20,help='Minimum resolved pair contrast used for the optional contrast loss');s.add_argument('--rank-threshold',type=float,default=None,help='Minimum resolved downstream-union contrast used for ranking loss; defaults to --contrast-threshold');s.add_argument('--reference-alpha',type=float,default=.05,help='Finite-MC resolution confidence used to filter optional contrast/ranking labels');s.add_argument('--seed',type=int,default=11);s.add_argument('--output',required=True)
    s=sub.add_parser('cbc-study',help='Compact held-out CBC baseline/ablation comparison')
    s.add_argument('--input',required=True)
    s.add_argument('--model',action='append',default=[],help='label=checkpoint.pt; repeat')
    s.add_argument('--contrast',type=float,default=.2)
    s.add_argument('--detect',type=float,default=.05)
    s.add_argument('--ranking-contrast',type=float,default=.05,
                   help='Subcritical held-out ranking threshold reported separately from the critical gate')
    s.add_argument('--ranking-detect',type=float,default=.01,
                   help='Prediction deadband for the separate ranking diagnostic')
    s.add_argument('--bootstrap',type=int,default=2000)
    s.add_argument('--output',required=True)
    s=sub.add_parser('cbc-boundary-study',help='Empirical non-certified anchor reuse screen')
    s.add_argument('--development',required=True);s.add_argument('--test',required=True)
    s.add_argument('--target-miss',type=float,default=.05);s.add_argument('--output',required=True)
    s=sub.add_parser('cbc-gate',help='Conservative evidence-adequacy report; not a safety certificate')
    s.add_argument('--study',required=True);s.add_argument('--boundary');s.add_argument('--output',required=True)
    s.add_argument('--min-resolved',type=int,default=30)
    s.add_argument('--min-matched',type=int,default=50)
    s.add_argument('--min-scenes',type=int,default=100)
    s=sub.add_parser('calibrate');s.add_argument('--input',required=True);s.add_argument('--checkpoint');s.add_argument('--alpha',type=float,default=.1);s.add_argument('--output',required=True)
    s=sub.add_parser('rq-audit');s.add_argument('--input',required=True);s.add_argument('--checkpoint');s.add_argument('--output',required=True);s.add_argument('--contrast',type=float,default=.2)
    s=sub.add_parser('extended-audit',help='Critical-pair, reuse, timing, yield and stratified calibration audit')
    s.add_argument('--input',required=True)
    s.add_argument('--checkpoint',help='Optional CBC checkpoint; predictions are recomputed before auditing')
    s.add_argument('--contrast',type=float,default=.2)
    s.add_argument('--detect',type=float,default=.05)
    s.add_argument('--bootstrap',type=int,default=2000)
    s.add_argument('--seed',type=int,default=51)
    s.add_argument('--reuse-clearance-multiplier',type=float)
    s.add_argument('--reuse-guard-multiplier',type=float)
    s.add_argument('--output',required=True)
    s=sub.add_parser('evaluate');s.add_argument('--input',required=True);s.add_argument('--checkpoint');s.add_argument('--calibration');s.add_argument('--eps-ego',type=float,default=.08);s.add_argument('--eps-down',type=float,default=.12);s.add_argument('--eps-inc',type=float,default=.1);s.add_argument('--eps-reg',type=float,default=1.5);s.add_argument('--surrogate-prefix-check',action='store_true',help='Simulation-only prefix check assumption; NOT a physical field certificate');s.add_argument('--output',required=True)
    s=sub.add_parser('audit-splits');s.add_argument('--train',required=True);s.add_argument('--calibration',required=True);s.add_argument('--test',required=True);s.add_argument('--output',required=True)
    s=sub.add_parser('tiv-audit');s.add_argument('--input',required=True);s.add_argument('--validation',help='Disjoint scene/overlap-group records required for a held-out value audit.');s.add_argument('--provisional-calibration',help='Optional frozen development calibration JSON used only to form LP witnesses and query features. It never certifies release.');s.add_argument('--allow-same-split-audit',action='store_true',help='Exploratory smoke mode only. Reusing train records is not a held-out value audit.');s.add_argument('--output',required=True);s.add_argument('--head',required=True);s.add_argument('--epochs',type=int,default=120);s.add_argument('--budget',type=float,default=12.)
    s=sub.add_parser('carla-check');s.add_argument('--host');s.add_argument('--port',type=int);s.add_argument('--timeout',type=float,default=8.);s.add_argument('--output')
    s=sub.add_parser('carla-run');s.add_argument('--host');s.add_argument('--port',type=int);s.add_argument('--town',default='Town04');s.add_argument('--out',required=True);s.add_argument('--step',type=float,default=.05);s.add_argument('--horizon',type=float,default=8.)
    return p

def run(args):
    cfg=Paths.load(args.paths)
    if args.cmd=='doctor':
        from caver.data.womd import inspect as wi
        from caver.data.nuplan import inspect as ni
        from caver.data.reactsim import inventory
        from importlib.util import find_spec
        available={k:bool(find_spec(k)) for k in ('tensorflow','waymo_open_dataset','waymax','nuplan','carla','torch','pytest')}
        report({'dependencies':available,'womd':wi(cfg.womd),'nuplan':ni(cfg.nuplan),'reactsim':inventory(cfg.reactsim,cfg.reactsim_repository,cfg.nuplan_maps),
                'nuplan_map_root':cfg.nuplan_maps,'reactsim_repository':cfg.reactsim_repository,
                'field_data_executed':False},args.output)
    elif args.cmd=='womd-check':
        from caver.data.womd import inspect,read
        info=inspect(cfg.womd)
        if args.decode:
            observed=[]
            for sc,fu in read(cfg.womd,format=args.format,split=args.split,limit=args.decode,with_future=True):
                observed.append({'id':sc.scene_id,'agents':len(sc.agents),'ego':sc.ego_id,
                                 'future_labels':len(fu),'source':sc.source,'steps':sc.provenance.get('num_track_states')})
            info['decoded_checks']=observed
        report(info,args.output)
    elif args.cmd=='womd-export':
        from caver.data.womd import export
        report(export(cfg.womd,args.format,args.split,args.output,args.limit))
    elif args.cmd=='womd-prep':
        from caver.data.preparation import prepare_womd
        result=prepare_womd(cfg.womd,args.split,args.partition,args.purpose,args.output,
                            limit=args.limit,max_scan=args.max_scan,
                            shard_seed=args.shard_seed,max_shards=args.max_shards,
                            per_shard=args.per_shard,max_output_mb=args.max_output_mb)
        report(result)
    elif args.cmd=='waymax-read':
        from caver.integrations.waymax_bridge import load_waymax_states,normalize_waymax_state
        from caver.data.schema import write_jsonl
        scenes=[]
        for state,shard,idx in load_waymax_states(cfg.womd,args.split,args.limit):scenes.append(normalize_waymax_state(state,shard,idx))
        write_jsonl(args.output,scenes)
        report({'scenes':len(scenes),'out':args.output,'waymax_native_closed_loop':False,
                'note':'Dataloader input only; reactive multi-agent rollout requires external official runner'})
    elif args.cmd=='nuplan-check':
        from caver.data.nuplan import inspect,inspect_db
        res=inspect(cfg.nuplan)
        if args.db:res['db_audit']=inspect_db(args.db)
        report(res,args.output)
    elif args.cmd=='nuplan-export':
        from caver.data.nuplan import export_scenarios,from_official_scenario
        from caver.data.schema import write_jsonl
        native=export_scenarios(args.db,args.map_root or cfg.nuplan_maps,args.limit)
        scenes=[from_official_scenario(s) for s in native]
        for sc in scenes:
            sc.provenance['dataset_split']=next((p for p in Path(args.db).parts if p.startswith('public_set_')), 'UNKNOWN')
        write_jsonl(args.output,scenes)
        report({'scenes':len(scenes),'out':args.output,'source':'NUPLAN_OFFICIAL_API','not_reactive_rollouts':True})
    elif args.cmd=='reactsim-check':
        from caver.data.reactsim import inventory,validate_export
        data=inventory(cfg.reactsim,cfg.reactsim_repository,cfg.nuplan_maps)
        if args.export:data['validated_rollout_records']=len(validate_export(args.export))
        report(data,args.output)
    elif args.cmd=='train-policy':
        from caver.data.schema import read_jsonl
        from caver.data.womd import read
        from caver.data.splits import training_partition
        from caver.models.policy import fit_logged_model
        scenes=read_jsonl(args.input,limit=args.limit) if args.input else read(cfg.womd,split='training',format='scenario',limit=args.limit)
        if args.womd_train_partition:
            # Freeze observational response training to the train partition.
            original=scenes
            def stream():
                for sc in original:
                    if sc.provenance.get('split')!='training':
                        raise ValueError('Observed policy partition requires official WOMD training')
                    if training_partition(sc)=='train':yield sc
            scenes=stream()
        report(fit_logged_model(scenes,args.output,epochs=args.epochs,device=args.device))
    elif args.cmd=='fourcar':
        from caver.experiment import single_demo
        report(single_demo(args.out,args.seed,args.worlds))
    elif args.cmd=='fourcar-challenge':
        from caver.experiment import frozen_challenge
        report(frozen_challenge(args.config,args.output))
    elif args.cmd=='fourcar-mediation':
        from caver.experiment import mediator_exchange,make_scene
        from caver.sim.traffic import sample_world
        sc=make_scene(args.seed)
        report(mediator_exchange(sc,sample_world(sc,args.world_seed,args.family)),args.out)
    elif args.cmd=='fourcar-scan':
        from caver.experiment import make_scene
        from caver.sim.traffic import candidates,sample_world,rollout
        results=[]
        for i in range(args.scenes):
            scene=make_scene(args.seed+i)
            all_actions={a.action_id:a for a in candidates()}
            for family in ('delayed_idm','late_brake'):
                for seed in range(args.worlds):
                    w=sample_world(scene,seed+1000*i,family)
                    r={a:rollout(scene,action,w) for a,action in all_actions.items()}
                    pair='B|C'; matched=(not r['early']['ego_collision'] and not r['late']['ego_collision'])
                    diff=r['early']['contact'].get(pair,0)!=r['late']['contact'].get(pair,0)
                    if matched and diff:
                        results.append({'scene_id':scene.scene_id,'world_seed':w.seed,'family':family,
                                        'early':r['early']['contact'],'late':r['late']['contact'],
                                        'clearance':{a:r[a]['min_clearance'][pair] for a in ('early','late')}})
        report({'searched':args.scenes*2*args.worlds,'positive_synthetic_cases':len(results),
                'examples':results[:30],'posthoc_search':True,'independent_confirmation':False},args.out)
    elif args.cmd=='collect':
        from caver.experiment import paired_scene,synthetic_records,store_records,make_scene,stress_scene,stress_actions
        from caver.models.cbc import load
        from caver.data.schema import read_jsonl
        from caver.data.corridor import corridor_subset
        from caver.data.splits import partition,training_partition
        model=load(args.checkpoint) if args.checkpoint else None
        policy=None
        if args.source_family=='learned_logged':
            if not args.policy_checkpoint:raise ValueError('learned_logged requires --policy-checkpoint')
            from caver.models.policy import load_logged_policy
            policy=load_logged_policy(args.policy_checkpoint)
        seeds={'train':10000,'calibration':300000,'test':600000}
        if args.source=='synthetic':
            scenes=(make_scene(seeds[args.split]+i) for i in range(args.scenes))
            stats={'input':'SYNTHETIC_ONLY'}
        elif args.source=='synthetic-stress':
            scenes=(stress_scene(seeds[args.split]+i) for i in range(args.scenes))
            stats={'input':'PREDECLARED_SYNTHETIC_STRESS_NOT_NATURAL_PREVALENCE'}
        else:
            if not args.input:raise ValueError('Source needs --input JSONL')
            def filter_scenes():
                scanned=0;kept=0
                for sc in read_jsonl(args.input):
                    # WOD validation windows can overlap. Partition by a
                    # conservative *static map group*, not unique scenario ID.
                    if args.source=='womd-jsonl':
                        tag=sc.provenance.get('split')
                        if args.split in ('train','calibration'):
                            if tag!='training' or training_partition(sc)!=args.split:continue
                        elif args.split=='test' and tag!='validation':continue
                    if args.source=='nuplan-jsonl':
                        tag=sc.provenance.get('dataset_split','')
                        if args.split=='train' and not tag.endswith('_train'):continue
                        if args.split in ('calibration','test'):
                            if tag!='public_set_val':continue
                            if partition(sc)!=args.split:continue
                    if kept>=args.scenes:break
                    scanned+=1
                    if sc.provenance.get('preparation_purpose')=='corridor':
                        # Womd-prep already normalized its local frame. Running
                        # corridor_subset again would rotate/reflect positions
                        # twice and reject genuinely eligible prepared scenes.
                        if sc.provenance.get('preparation_partition')!=args.split:
                            raise ValueError('Prepared partition does not match collect --split')
                        if (sc.provenance.get('corridor_filter')!='GEOMETRIC_NOT_MAP_CERTIFIED'
                                or {a.agent_id for a in sc.agents}!={'E','A','B','C'}):
                            raise ValueError('Prepared corridor roles are inconsistent')
                        c=sc
                    else:
                        c,reason=corridor_subset(sc)
                    if c is not None:
                        kept+=1;yield c
                if kept==0:raise ValueError(f'Zero eligible four-car corridor scenes in {scanned} input scenes')
            scenes=filter_scenes();stats={'input':args.input,'eligibility':'GEOMETRIC_ONLY'}
        if args.source=='synthetic-stress' and args.source_family=='learned_logged':
            raise ValueError('Synthetic stress requires a named non-logged surrogate source family')
        audit_families=(('stress_reference','stress_shifted') if args.source=='synthetic-stress'
                        else ('late_brake','conservative'))
        if args.candidate_policy=='observed-adaptive' and args.source not in ('womd-jsonl','nuplan-jsonl'):
            raise ValueError('Observed-adaptive candidate control policy is for observation-first corridor experiments')
        rec=[]
        for i,sc in enumerate(scenes):
            actions=(stress_actions(args.stress_profile) if args.source=='synthetic-stress' else None)
            candidate_meta={'policy':'FIXED_TEMPLATE_V1','map_law_route_feasibility':'UNVERIFIED'}
            if args.candidate_policy=='observed-adaptive':
                from caver.models.candidate_policy import choose_observation_only_actions
                actions,candidate_meta=choose_observation_only_actions(sc)
            row=paired_scene(sc,actions=actions,
                    source_family=args.source_family,source_worlds=args.source_worlds,
                    audit_families=audit_families,audit_worlds=args.audit_worlds,
                    seed=seeds[args.split]+i*73,model=model,source_policy=policy)
            row['experimental_split']=args.split
            row['candidate_selection_audit']=candidate_meta
            rec.append(row)
        store_records(rec,args.output)
        report({'split':args.split,'records':len(rec),'output':args.output,
                'reference_families':list(audit_families),
                'reference_type':'SIMULATED_NOT_REAL_CAUSAL_LABEL','candidate_policy':args.candidate_policy,
                'stress_profile':args.stress_profile if args.source=='synthetic-stress' else None,**stats})
    elif args.cmd=='train-cbc':
        from caver.experiment import fit_cbc,read_records
        report(fit_cbc(list(read_records(args.input)),args.output,epochs=args.epochs,variant=args.variant,device=args.device,contrast_weight=args.contrast_weight,rank_weight=args.rank_weight,seed=args.seed,contrast_threshold=args.contrast_threshold,rank_threshold=args.rank_threshold,reference_alpha=args.reference_alpha))
    elif args.cmd=='cbc-study':
        from caver.validation.cbc_study import study
        files={}
        for expr in args.model:
            if '=' not in expr:raise ValueError('Use --model LABEL=PATH')
            name,path=expr.split('=',1)
            if not name or name in ('source','anchor_copy') or not path or name in files:
                raise ValueError('Duplicate or invalid CBC model label')
            files[name]=path
        report(study(args.input,files,args.contrast,args.detect,args.bootstrap,
                     ranking_contrast=args.ranking_contrast,
                     ranking_detect=args.ranking_detect),args.output)
    elif args.cmd=='cbc-boundary-study':
        from caver.validation.boundary_study import study
        report(study(args.development,args.test,args.target_miss),args.output)
    elif args.cmd=='cbc-gate':
        from caver.validation.cbc_gate import summarize
        report(summarize(args.study,args.boundary,args.min_resolved,args.min_matched,args.min_scenes),args.output)
    elif args.cmd=='calibrate':
        from caver.experiment import read_records,record_with_new_predictions
        from caver.models.cbc import load
        from caver.validation.evidence import conformal_calibrate
        rec=list(read_records(args.input));model=load(args.checkpoint) if args.checkpoint else None
        if any(r.get('experimental_split')!='calibration' for r in rec):
            raise ValueError('Calibration requires independently collected --split calibration records')
        if model:rec=[record_with_new_predictions(row,model) for row in rec]
        digest=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() if args.checkpoint else 'SOURCE_ROLLOUT'
        cal=conformal_calibrate(rec,alpha=args.alpha,audit_worlds=rec[0]['audit_worlds_per_family']*len(rec[0]['audit_families']),
                                reference_families=rec[0]['audit_families'],predictor_digest=digest)
        cal.save(args.output)
        report({'calibration':args.output,'quantile':cal.quantile,'mc_buffer':cal.mc_buffer,'n':cal.n,
                'scope':'FROZEN_PREDICTOR_SIMULATED_REFERENCE_SCENE_EXCHANGEABILITY'})
    elif args.cmd=='rq-audit':
        from caver.experiment import read_records,record_with_new_predictions
        from caver.models.cbc import load
        from caver.validation.metrics import rq_audit
        rec=list(read_records(args.input))
        if args.checkpoint:
            model=load(args.checkpoint);rec=[record_with_new_predictions(r,model) for r in rec]
        report(rq_audit(rec,contrast_threshold=args.contrast),args.output)
    elif args.cmd=='extended-audit':
        from caver.experiment import read_records,record_with_new_predictions
        from caver.models.cbc import load
        from caver.validation.extended_metrics import extended_audit
        rec=list(read_records(args.input))
        checkpoint_digest=None; model_variant=None
        if args.checkpoint:
            model=load(args.checkpoint)
            checkpoint_digest=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
            model_variant=getattr(model,'variant',None)
            rec=[record_with_new_predictions(r,model) for r in rec]
        result=extended_audit(
            rec, contrast_threshold=args.contrast, detection_threshold=args.detect,
            bootstrap=args.bootstrap, seed=args.seed,
            reuse_clearance_multiplier=args.reuse_clearance_multiplier,
            reuse_guard_multiplier=args.reuse_guard_multiplier)
        result['provenance']={
            'input_sha256':hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
            'checkpoint_sha256':checkpoint_digest,
            'model_variant':model_variant,
            'records':len(rec),
            'evaluator_schema':result.get('schema'),
            'runtime_scope':result.get('runtime',{}).get('scope')}
        report(result,args.output)
    elif args.cmd=='evaluate':
        from caver.experiment import read_records,record_with_new_predictions,evaluate
        from caver.models.cbc import load
        from caver.validation.evidence import Calibration
        rec=list(read_records(args.input));model=load(args.checkpoint) if args.checkpoint else None
        if model:rec=[record_with_new_predictions(row,model) for row in rec]
        if any(r.get('experimental_split')!='test' for r in rec):
            raise ValueError('Final evaluation requires held-out --split test records')
        cal=Calibration.load(args.calibration) if args.calibration else None
        if cal is not None:
            digest=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() if args.checkpoint else 'SOURCE_ROLLOUT'
            if digest!=cal.predictor_digest:raise ValueError('Calibration/predictor checkpoint digest mismatch')
            if tuple(rec[0]['audit_families'])!=tuple(cal.reference_families):raise ValueError('Reference reactor mix differs from calibration')
        r=evaluate(rec,cal,args.eps_down,args.eps_ego,args.eps_inc,args.eps_reg,args.surrogate_prefix_check)
        report(r,args.output)
    elif args.cmd=='audit-splits':
        from caver.experiment import read_records
        from caver.data.splits import audit_record_splits
        parts={k:list(read_records(getattr(args,k))) for k in ('train','calibration','test')}
        result=audit_record_splits(parts)
        report(result,args.output)
        if not result['passed']:
            raise ValueError('Split leakage detected: duplicated scenario or shared map/log group')
    elif args.cmd=='tiv-audit':
        from caver.experiment import read_records
        from caver.validation.evidence import choose,Calibration
        from caver.validation.tiv import propose,verify_source,query_features,learn_query_value,load_query_value,audit_query_value
        from caver.data.schema import Scene
        from caver.sim.traffic import Action
        rec=list(read_records(args.input));training=[];audit=[]
        provisional_calibration=Calibration.load(args.provisional_calibration) if args.provisional_calibration else None
        if not rec or any(r.get('experimental_split')!='train' for r in rec):
            raise ValueError('TIV acquisition-head training requires --split train; never test/calibration reference labels')
        if args.validation is None and not args.allow_same_split_audit:
            raise ValueError('TIV requires a disjoint --validation split; use --allow-same-split-audit only for exploratory smoke tests')
        validation=list(read_records(args.validation)) if args.validation else rec
        if not validation or any(r.get('experimental_split') not in ('train','calibration','test') for r in validation):
            raise ValueError('TIV validation records must use train, calibration or test split')
        if args.validation and any(r.get('experimental_split')=='train' for r in validation):
            raise ValueError('TIV validation records must be disjoint from train records')
        if args.validation:
            train_ids={r.get('scene_id') for r in rec}
            val_ids={r.get('scene_id') for r in validation}
            if train_ids & val_ids:
                raise ValueError('TIV train/validation scene overlap')
            def group(row):
                return row.get('scene',{}).get('provenance',{}).get('overlap_group',row.get('scene_id'))
            if {group(r) for r in rec} & {group(r) for r in validation}:
                raise ValueError('TIV train/validation overlap-group overlap')

        def build_labels(rows):
            result=[]
            for row in rows:
                acts=[Action(**a) for a in row['actions']];names=[a.action_id for a in acts]
                d=choose(row['predicted'],names,row['pairs'],calibration=provisional_calibration)
                queries=propose(d,row['predicted'],names,row['pairs'],budget=args.budget,deadline_ms=110)
                scene=Scene.from_json(row['scene'])
                base=min(names,key=lambda a:row['predicted'][a+'/down']+(.5 if a=='reference' else 0.))
                oracle=lambda a:row['audited'][a+'/down']+(.5 if a=='reference' else 0.)
                for q in queries:
                    out=verify_source(scene,acts,q,seed=98174)
                    if out['status']!='SIMULATED_QUERY':continue
                    pv=dict(row['predicted'])
                    for a in q.candidates:
                        xs=[r['downstream'] for r in out['outcomes'] if r['action']==a]
                        if xs:pv[a+'/down']=float(np.mean(xs))
                    selected=min(names,key=lambda a:pv[a+'/down']+(.5 if a=='reference' else 0.))
                    gain=oracle(base)-oracle(selected)-.001*q.cost-.0002*q.presumed_latency_ms
                    result.append({'scene_id':row['scene_id'],'source':q.source,'before':base,'after':selected,
                                   'action_changed':bool(base!=selected),
                                   'signed_value':float(gain),'features':query_features(d,row['predicted'],q).tolist(),
                                   'evaluated_reference':row['audit_families'],'query_is_certificate':False,
                                   'query_contract':{'candidate_actions':list(q.candidates),
                                                     'affected_pairs':list(q.pairs),
                                                     'mediator_chain':list(q.chain),
                                                     'window':list(q.window),
                                                     'samples':q.samples,'cost':q.cost,
                                                     'presumed_latency_ms':q.presumed_latency_ms},
                                   'before_objective':float(oracle(base)),
                                   'after_objective':float(oracle(selected)),
                                   'query_measured_walltime_ms':out.get('measured_walltime_ms'),
                                   'query_outcome_count':len(out.get('outcomes',[]))})
            return result

        training=build_labels(rec)
        audit=build_labels(validation)
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text('\n'.join(json.dumps(a) for a in audit)+'\n')
        if len(training)>=6:
            status=learn_query_value(training,args.head,args.epochs)
            value_metrics=audit_query_value(audit,load_query_value(args.head))
        else:
            status={'trained':False,'reason':'Insufficient audited query labels','n':len(training)}
            value_metrics={'status':'NOT_MEASURED','n':len(audit),'reason':'head not trained'}
        def audit_summary(rows):
            vals=[float(r['signed_value']) for r in rows]
            return {'n':len(rows),
                    'action_change_rate':float(np.mean([r['action_changed'] for r in rows])) if rows else None,
                    'mean_signed_value':float(np.mean(vals)) if vals else None,
                    'positive_value_rate':float(np.mean([v>0 for v in vals])) if vals else None,
                    'source_counts':{s:sum(r['source']==s for r in rows) for s in sorted({r['source'] for r in rows})}}
        report({'training_records':len(training),'validation_records':len(audit),'audit':args.output,'head':status,
                'same_file_train_validation':not bool(args.validation),
                'heldout_value_audit':bool(args.validation),
                'provisional_calibration':args.provisional_calibration,
                'training_audit_summary':audit_summary(training),
                'validation_audit_summary':audit_summary(audit),
                'value_metrics':value_metrics,
                'warning':'Signed synthetic proxy acquisition targets only; no terminal release calibration or independent physical reference'})
    elif args.cmd=='carla-check':
        from caver.integrations.carla_runner import preflight
        result=preflight(args.host or cfg.carla_host,args.port or cfg.carla_port,args.timeout)
        report(result,args.output)
        if not result['rpc_connected']:
            raise RuntimeError('CARLA RPC connection failed; inspect carla-check report')
    elif args.cmd=='carla-run':
        from caver.integrations.carla_runner import run_server
        report(run_server(args.host or cfg.carla_host,args.port or cfg.carla_port,args.town,args.out,step=args.step,horizon=args.horizon))
    else:raise ValueError(args.cmd)

def main(argv=None):
    a=parser().parse_args(argv)
    try:run(a)
    except (RuntimeError,FileNotFoundError,ValueError,ImportError) as exc:
        print('CaVER error: '+str(exc),file=sys.stderr)
        raise SystemExit(2) from exc

if __name__=='__main__':main()
