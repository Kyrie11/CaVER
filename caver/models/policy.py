"""WOMD logged behavior pretraining (observed policy regression, not counterfactual data).

The lightweight model predicts longitudinal acceleration from *past* nearby
kinematics. A pretrained SMART checkpoint is NOT bundled or impersonated.
"""
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn

class LoggedPolicyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(5,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def history_rows(scene):
    """Each row uses consecutive *historical* frames only (no future leakage)."""
    agents=[a for a in scene.agents if a.history and len(a.history)>=4]
    rows=[]
    for a in agents:
        tr=np.asarray(a.history,float)
        for t in range(2,len(tr)-1):
            if not np.all(tr[t-2:t+2,5]>0):continue
            vx=np.linalg.norm(tr[t,:2]-tr[t-1,:2])/.1
            newv=np.linalg.norm(tr[t+1,:2]-tr[t,:2])/.1
            val=(newv-vx)/.1
            if not np.isfinite(val) or abs(val)>12:continue
            front=[]
            for b in agents:
                if a.agent_id==b.agent_id:continue
                oth=np.asarray(b.history,float)
                if len(oth)!=len(tr) or not oth[t,5]:continue
                hd=tr[t,4];ux,uy=math.cos(hd),math.sin(hd)
                dx,dy=oth[t,:2]-tr[t,:2];dist=dx*ux+dy*uy
                lat=abs(-uy*dx+ux*dy)
                if dist>0 and lat<2.4:
                    speed_other=np.linalg.norm(oth[t,:2]-oth[t-1,:2])/.1
                    front.append((dist-.5*(a.length+b.length),vx-speed_other))
            gap,closing=min(front,key=lambda z:z[0]) if front else (70.,0.)
            rows.append((np.array([vx/25.,np.clip(gap,0,70)/70.,closing/15.,math.sin(tr[t,4]),math.cos(tr[t,4])],np.float32),
                         np.float32(np.clip(val,-10,5))))
    return rows

def fit_logged_model(scenes,output,epochs=10,batch=2048,lr=1e-3,max_rows=250000,seed=7,device='auto'):
    rng=np.random.default_rng(seed)
    if device=='auto':device='cuda:0' if torch.cuda.is_available() else 'cpu'
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; install a CUDA-enabled PyTorch wheel')
    dev=torch.device(device)
    samples=[]
    for scene in scenes:
        samples.extend(history_rows(scene))
        if len(samples)>=max_rows:break
    if len(samples)<20:raise ValueError('Need >=20 valid logged history samples; current scenes have insufficient 10Hz history')
    if len(samples)>max_rows:samples=[samples[i] for i in rng.choice(len(samples),size=max_rows,replace=False)]
    X=torch.tensor(np.stack([r[0] for r in samples]));Y=torch.tensor(np.array([r[1] for r in samples]))
    X=X.to(dev);Y=Y.to(dev)
    m=LoggedPolicyHead().to(dev);opt=torch.optim.Adam(m.parameters(),lr=lr)
    for ep in range(epochs):
        perm=torch.randperm(len(X));losses=[]
        for ids in perm.split(batch):
            loss=torch.nn.functional.smooth_l1_loss(m(X[ids]),Y[ids]);opt.zero_grad();loss.backward();opt.step()
            losses.append(loss.item())
    m=m.to('cpu')
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    torch.save({'weights':m.state_dict(),'model':'LOGGED_OBSERVATIONAL_POLICY','samples':len(samples)},output)
    return {'samples':len(samples),'training_loss':float(np.mean(losses)),'claim':'observational acceleration only; not an ego-conditioned world model','training_device':str(dev)}


def load_logged_policy(path):
    try:data=torch.load(path,map_location='cpu',weights_only=True)
    except TypeError:data=torch.load(path,map_location='cpu')
    if data.get('model')!='LOGGED_OBSERVATIONAL_POLICY':raise ValueError('Wrong model type')
    model=LoggedPolicyHead();model.load_state_dict(data['weights']);model.eval()
    return model
