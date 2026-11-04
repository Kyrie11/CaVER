"""Conservative WOMD/nuPlan corridor filtering for the 4-car cascade task.

No assertion of lane-map consistency: a geometric eligibility filter only.
Unsupported intersections and ambiguous lane roles are SKIPPED WITH REASONS.
"""
from dataclasses import replace
import math
from .schema import Scene,Agent

def corridor_subset(sc):
    ego=next(a for a in sc.agents if a.agent_id==sc.ego_id)
    cy,sy=math.cos(ego.yaw),math.sin(ego.yaw)
    def local(a):
        dx,dy=a.x-ego.x,a.y-ego.y
        vx,vy=a.vx,a.vy
        history=None
        if a.history is not None:
            history=[]
            for hx,hy,hvx,hvy,hyaw,valid in a.history:
                if not valid:
                    history.append([0.,0.,0.,0.,0.,0]);continue
                ddx,ddy=hx-ego.x,hy-ego.y
                history.append([ddx*cy+ddy*sy,-ddx*sy+ddy*cy,
                                hvx*cy+hvy*sy,-hvx*sy+hvy*cy,
                                ((hyaw-ego.yaw+math.pi)%(2*math.pi)-math.pi),1])
        return replace(a,x=dx*cy+dy*sy,y=-dx*sy+dy*cy,
                       vx=vx*cy+vy*sy,vy=-vx*sy+vy*cy,
                       yaw=((a.yaw-ego.yaw+math.pi)%(2*math.pi)-math.pi),history=history)
    E=local(ego)
    others=[local(a) for a in sc.agents if a.agent_id!=sc.ego_id]
    veh=[a for a in others if a.kind=='vehicle' and abs(a.yaw)<.24 and abs(a.vy)<2.5]
    # A target-lane group with two trailing vehicles and one front vehicle.
    for sign in (-1,1):
        lane_y=3.5*sign
        grouped=sorted([a for a in veh if abs(a.y-lane_y)<1.0],key=lambda x:x.x)
        behind=[a for a in grouped if -60<a.x<-6]
        front=[a for a in grouped if 18<a.x<100]
        mid=[a for a in grouped if -7<a.x<12]
        if behind and mid and front:
            C=behind[-1];B=mid[0];A=front[0]
            if B.x-C.x<12 or A.x-B.x<15:continue
            # Reflect negative target lanes so ego always changes toward +Y.
            # This is only a geometric local-frame normalization.
            normalized=[]
            for agent,ident in ((E,'E'),(A,'A'),(B,'B'),(C,'C')):
                history=([[h[0],h[1]*sign,h[2],h[3]*sign,h[4]*sign,h[5]] for h in agent.history]
                         if agent.history is not None else None)
                normalized.append(replace(agent,agent_id=ident,y=agent.y*sign,vy=agent.vy*sign,
                                          yaw=agent.yaw*sign,history=history))
            chosen=normalized
            return Scene(sc.scene_id,sc.source+'_CORRIDOR_FILTER',sc.dt,'E',chosen,
                         {**sc.provenance,'original_roles':{'E':sc.ego_id,'A':A.agent_id,'B':B.agent_id,'C':C.agent_id},
                          'corridor_filter':'GEOMETRIC_NOT_MAP_CERTIFIED','action_target_lane':3.5}).validate(),None
    return None,'NO_ELIGIBLE_ADJACENT_CHAIN'
