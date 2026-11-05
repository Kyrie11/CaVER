#!/usr/bin/env python3
"""Small observation-only risk-context manifest for WOMD corridor records."""
import argparse,json
from pathlib import Path
from caver.validation.observation_triage import observation_audit

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True,help='womd_*_corridor.jsonl.gz')
    p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    result=observation_audit(args.input,args.limit)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print('Wrote observation-only:',out,'scenes:',result['observed_scenes'])

if __name__=='__main__':main()
