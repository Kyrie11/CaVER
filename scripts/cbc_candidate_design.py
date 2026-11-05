#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from caver.validation.candidate_design import study

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True)
    p.add_argument('--output',required=True);p.add_argument('--limit',type=int)
    a=p.parse_args()
    result=study(a.input,a.limit)
    o=Path(a.output);o.parent.mkdir(parents=True,exist_ok=True)
    o.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('CBC candidate-design triage:',o,'N=',result['scenes'])

if __name__=='__main__':main()
