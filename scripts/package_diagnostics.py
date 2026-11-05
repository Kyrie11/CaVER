#!/usr/bin/env python3
"""Make a deliberately small user-return bundle from *reports*, not datasets.

No raw observations, source future predictions, target records, checkpoints,
private maps or CARLA trajectory dumps are included. Each file is checksummed.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import zipfile

REPORT_NAMES={
    'cbc_study.json','cbc_boundary_study.json','cbc_gate.json',
    'rq_audit.json','split_audit.json','fourcar_challenge.json',
    'womd_training_check.json','womd_validation_check.json','doctor.json',
    'nuplan_inventory.json','reactsim_inventory.json',
    'carla_check.json','carla_summary.json','pytest_summary.txt',
    'calibration.json','evaluation.json','cbc_summary_table.md','cbc_summary_table.csv'
}


def main():
    parser=argparse.ArgumentParser(description='Compact diagnostic ZIP: reports only')
    parser.add_argument('folders',nargs='+',help='One or more output directories')
    parser.add_argument('--output',default='outputs/CaVER_CBC_report_bundle.zip')
    parser.add_argument('--log',action='append',default=[],help='Optional stderr log (last 200 KB)')
    parser.add_argument('--max-mb',type=float,default=20.)
    args=parser.parse_args()
    if not 0<args.max_mb<50:raise ValueError('Bundle max must be >0 and <50 MB')
    limit=int(args.max_mb*1024*1024)
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    entries=[];excluded=[];names=set();total=0
    for f in args.folders:
        root=Path(f)
        if not root.is_dir():
            excluded.append({'path':str(root),'reason':'FOLDER_MISSING'})
            continue
        for file in sorted(root.rglob('*')):
            if not file.is_file():continue
            if file.name not in REPORT_NAMES and not file.name.endswith('.manifest.json'):continue
            if file.name.startswith('carla_rollouts'):continue
            if file.stat().st_size>3*1024*1024:
                excluded.append({'path':str(file),'reason':'REPORT_ABOVE_3_MB'})
                continue
            entry=f'{root.name}/{file.relative_to(root).as_posix()}'
            if entry in names:continue
            data=file.read_bytes()
            if total+len(data)>limit-1024*1024:
                excluded.append({'path':str(file),'reason':'BUNDLE_BUDGET'})
                continue
            entries.append((entry,data))
            names.add(entry);total+=len(data)
    for path in args.log:
        p=Path(path)
        if not p.is_file():
            excluded.append({'path':str(p),'reason':'LOG_MISSING'})
            continue
        # Errors are not hidden, but no full raw log is copied by default.
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-200000));data=f.read()
        entry='logs/'+p.name
        if entry not in names and total+len(data)<limit-1024*1024:
            entries.append((entry,data));total+=len(data);names.add(entry)
    if not entries:
        raise ValueError('No whitelisted reports. Provide completed output folders.')
    manifest={
      'schema':'CAVER_COMPACT_DIAGNOSTICS_V1',
      'safety':'REPORTS_ONLY_NO_TRAINING_DATA_NO_CHECKPOINTS_NO_PRIVATE_MAPS',
      'files':[{'name':name,'bytes':len(data),'sha256':sha256(data).hexdigest()}
               for name,data in entries],
      'excluded':excluded,
      'limitations':['Reference labels in CBC studies are simulator-conditional unless explicitly replaced',
                     'Do not infer real counterfactual effects from WOMD observational futures',
                     'A CBC no-reuse screen is not a physical certificate']}
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=8) as z:
        for name,data in entries:z.writestr(name,data)
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
        z.writestr('README_CN.txt',
            'CaVER CBC 诊断包：仅包含小体积 JSON 报告、校验与可选错误日志。\n'
            '没有 WOMD JSONL、未来标签样本、神经网络 checkpoint 或地图数据。\n'
            '所有 CBC 参考标签都来自被声明的结构反应模型，除非另行提供独立官方闭环证据。\n')
    if target.stat().st_size>limit:raise RuntimeError('Unexpected diagnostic bundle over budget')
    print(json.dumps({'output':str(target),'compressed_bytes':target.stat().st_size,
                      'included_files':len(entries),'excluded_files':len(excluded)},indent=2))

if __name__=='__main__':main()
