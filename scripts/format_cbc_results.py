#!/usr/bin/env python3
"""Create a small markdown/CSV table from CBC falsification metrics.

The table explicitly states which candidate matching and independent MC
reference test sizes apply. Output is never a proof of efficacy by itself.
"""
import argparse
import csv
import json
from pathlib import Path

def show(x):return 'NA' if x is None else f'{float(x):.4f}'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--study',required=True)
    p.add_argument('--output-prefix',required=True)
    args=p.parse_args()
    d=json.loads(Path(args.study).read_text())
    chosen=('source','anchor_copy','no_events','full','anchor_full','anchor_graph',
            'anchor_graph_delay_kernel','delay_kernel_rank')
    cols=['Method','Critical N','Resolved N','Sign error resolved','Underresolution',
          'Near-boundary pair MAE','Union-risk MAE','Union sign resolved',
          'Union underresolution resolved','Matched safety regret']
    mapping=['critical_pair_tests','critical_pair_ref_sign_resolved',
             'reference_resolved_sign_error_rate','critical_pair_underresolution_rate',
             'near_boundary_pair_risk_mae','downstream_union_reference_risk_mae',
             'downstream_union_resolved_sign_error_rate',
             'downstream_union_resolved_underresolution_rate',
             'matched_ego_safe_safety_ranking_regret']
    lines=[]
    for name in chosen:
        if name not in d['methods']:continue
        m=d['methods'][name]
        lines.append([name]+[str(m.get(k)) if idx<2 else show(m.get(k))
                              for idx,k in enumerate(mapping)])
    out=Path(args.output_prefix);out.parent.mkdir(parents=True,exist_ok=True)
    with open(str(out)+'.csv','w',newline='') as f:
        writer=csv.writer(f);writer.writerow(cols);writer.writerows(lines)
    with open(str(out)+'.md','w') as f:
        f.write('# CBC model-conditional heldout audit (NOT real causal identification)\n\n')
        f.write('| '+' | '.join(cols)+' |\n')
        f.write('| '+' | '.join(['---']*len(cols))+' |\n')
        for row in lines:f.write('| '+' | '.join(row)+' |\n')
        f.write('\nReference families: '+', '.join(d.get('reference_families',[]))+'; '
                +'Worlds per family/action: '+str(d.get('reference_worlds_per_family'))+'; '
                +'Matched and resolved sample counts must be checked before interpretation.\n')
    print(json.dumps({'csv':str(out)+'.csv','markdown':str(out)+'.md','methods':len(lines)}))

if __name__=='__main__':main()
