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
    s=sub.add_parser('waymax-read');s.add_argument('--split',default='training');s.add_argument('--limit',type=int,default=2);s.add_argument('--output',required=True)
    s=sub.add_parser('nuplan-check');s.add_argument('--split');s.add_argument('--db',help='Optional one DB strict schema audit');s.add_argument('--output')
    s=sub.add_parser('nuplan-export');s.add_argument('--db',required=True);s.add_argument('--map-root');s.add_argument('--limit',type=int,default=10);s.add_argument('--output',required=True)
    s=sub.add_parser('reactsim-check');s.add_argument('--export',help='Optional verified official simulator JSONL export');s.add_argument('--output')
    s=sub.add_parser('train-policy');s.add_argument('--input');s.add_argument('--womd-train-partition',action='store_true',help='Do not fit observational reactor on frozen calibration scenarios');s.add_argument('--limit',type=int,default=3000);s.add_argument('--epochs',type=int,default=15);s.add_argument('--output',required=True)
    s=sub.add_parser('fourcar');s.add_argument('--out',required=True);s.add_argument('--seed',type=int,default=10);s.add_argument('--worlds',type=int,default=4)
    s=sub.add_parser('fourcar-challenge');s.add_argument('--config',default='configs/fourcar_challenge.json');s.add_argument('--output',required=True)
    s=sub.add_parser('fourcar-mediation');s.add_argument('--out',required=True);s.add_argument('--seed',type=int,default=10);s.add_argument('--world-seed',type=int,default=10);s.add_argument('--family',default='late_brake')
    s=sub.add_parser('fourcar-scan');s.add_argument('--out',required=True);s.add_argument('--scenes',type=int,default=100);s.add_argument('--worlds',type=int,default=3);s.add_argument('--seed',type=int,default=100)
    s=sub.add_parser('collect');s.add_argument('--source',choices=['synthetic','womd-jsonl','nuplan-jsonl'],default='synthetic');s.add_argument('--input');s.add_argument('--split',choices=['train','calibration','test'],required=True);s.add_argument('--scenes',type=int,default=20);s.add_argument('--source-worlds',type=int,default=4);s.add_argument('--audit-worlds',type=int,default=16);s.add_argument('--checkpoint');s.add_argument('--source-family',choices=['delayed_idm','learned_logged'],default='delayed_idm');s.add_argument('--policy-checkpoint');s.add_argument('--output',required=True)
    s=sub.add_parser('train-cbc');s.add_argument('--input',required=True);s.add_argument('--epochs',type=int,default=80);s.add_argument('--output',required=True)
    s=sub.add_parser('calibrate');s.add_argument('--input',required=True);s.add_argument('--checkpoint');s.add_argument('--alpha',type=float,default=.1);s.add_argument('--output',required=True)
    s=sub.add_parser('rq-audit');s.add_argument('--input',required=True);s.add_argument('--checkpoint');s.add_argument('--output',required=True);s.add_argument('--contrast',type=float,default=.2)
    s=sub.add_parser('evaluate');s.add_argument('--input',required=True);s.add_argument('--checkpoint');s.add_argument('--calibration');s.add_argument('--eps-ego',type=float,default=.08);s.add_argument('--eps-down',type=float,default=.12);s.add_argument('--eps-inc',type=float,default=.1);s.add_argument('--eps-reg',type=float,default=1.5);s.add_argument('--surrogate-prefix-check',action='store_true',help='Simulation-only prefix check assumption; NOT a physical field certificate');s.add_argument('--output',required=True)
    s=sub.add_parser('audit-splits');s.add_argument('--train',required=True);s.add_argument('--calibration',required=True);s.add_argument('--test',required=True);s.add_argument('--output',required=True)
    s=sub.add_parser('tiv-audit');s.add_argument('--input',required=True);s.add_argument('--output',required=True);s.add_argument('--head',required=True);s.add_argument('--epochs',type=int,default=120);s.add_argument('--budget',type=float,default=12.)
    s=sub.add_parser('carla-run');s.add_argument('--town',default='Town04');s.add_argument('--out',required=True);s.add_argument('--step',type=float,default=.05);s.add_argument('--horizon',type=float,default=8.)
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
        report(fit_logged_model(scenes,args.output,epochs=args.epochs))
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
        from caver.experiment import paired_scene,synthetic_records,store_records,make_scene
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
        if args.source=='synthetic':scenes=(make_scene(seeds[args.split]+i) for i in range(args.scenes)); stats={'input':'SYNTHETIC_ONLY'}
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
                    c,reason=corridor_subset(sc)
                    if c is not None:
                        kept+=1;yield c
                if kept==0:raise ValueError(f'Zero eligible four-car corridor scenes in {scanned} input scenes')
            scenes=filter_scenes();stats={'input':args.input,'eligibility':'GEOMETRIC_ONLY'}
        rec=[paired_scene(sc,source_family=args.source_family,source_worlds=args.source_worlds,audit_worlds=args.audit_worlds,
                          seed=seeds[args.split]+i*73,model=model,source_policy=policy) for i,sc in enumerate(scenes)]
        for row in rec:row['experimental_split']=args.split
        store_records(rec,args.output)
        report({'split':args.split,'records':len(rec),'output':args.output,
                'reference_families':['late_brake','conservative'],
                'reference_type':'SIMULATED_NOT_REAL_CAUSAL_LABEL',**stats})
    elif args.cmd=='train-cbc':
        from caver.experiment import fit_cbc,read_records
        report(fit_cbc(list(read_records(args.input)),args.output,epochs=args.epochs))
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
        from caver.validation.evidence import choose
        from caver.validation.tiv import propose,verify_source,query_features,learn_query_value
        from caver.data.schema import Scene
        from caver.sim.traffic import Action
        rec=list(read_records(args.input));training=[];audit=[]
        if not rec or any(r.get('experimental_split')!='train' for r in rec):
            raise ValueError('TIV acquisition-head training requires --split train; never test/calibration reference labels')
        for row in rec:
            acts=[Action(**a) for a in row['actions']];names=[a.action_id for a in acts]
            d=choose(row['predicted'],names,row['pairs'],calibration=None)
            queries=propose(d,row['predicted'],names,row['pairs'],budget=args.budget,deadline_ms=110)
            scene=Scene.from_json(row['scene'])
            # Loss is an explicitly stated *proxy* selector loss on held-out
            # structural reactors; not a valid certified post-query DSE update.
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
                item={'scene_id':row['scene_id'],'source':q.source,'before':base,'after':selected,
                      'signed_value':float(gain),'features':query_features(d,row['predicted'],q).tolist(),
                      'evaluated_reference':row['audit_families'],'query_is_certificate':False}
                training.append(item);audit.append(item)
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text('\n'.join(json.dumps(a) for a in audit)+'\n')
        if len(training)>=6:status=learn_query_value(training,args.head,args.epochs)
        else:status={'trained':False,'reason':'Insufficient audited query labels'}
        report({'records':len(training),'audit':args.output,'head':status,
                'warning':'Signed synthetic proxy acquisition targets only; no terminal release calibration or independent physical reference'})
    elif args.cmd=='carla-run':
        from caver.integrations.carla_runner import run_server
        report(run_server(cfg.carla_host,cfg.carla_port,args.town,args.out,step=args.step,horizon=args.horizon))
    else:raise ValueError(args.cmd)

def main(argv=None):
    a=parser().parse_args(argv)
    try:run(a)
    except (RuntimeError,FileNotFoundError,ValueError,ImportError) as exc:
        print('CaVER error: '+str(exc),file=sys.stderr)
        raise SystemExit(2) from exc

if __name__=='__main__':main()
