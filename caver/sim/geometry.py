"""Oriented box separating axes and conservative sampling diagnostics.

The OBB intersection test is based on the original RIDE geometry.py, cleaned
and exercised with extra tests. A discrete rollout is NOT a continuous collision
certificate; a speed-bound inflation can be requested by the caller.
"""
import math
import numpy as np

def obb_signed_clearance(a, b, inflation=0.0):
    """Largest signed separating-axis gap; positive => disjoint; negative => overlap.

    Negative value is not a Euclidean penetration depth. At overlap, all axes
    overlap. At separation, any positive axis certifies disjoint OBBs.
    """
    def axes(h):return [(math.cos(h),math.sin(h)),(-math.sin(h),math.cos(h))]
    va=axes(float(a[2]));vb=axes(float(b[2]));dx=float(b[0]-a[0]);dy=float(b[1]-a[1])
    gaps=[]
    for u in va+vb:
        ra=sum(.5*float(dim)*abs(v[0]*u[0]+v[1]*u[1]) for dim,v in zip(a[3:5],va))
        rb=sum(.5*float(dim)*abs(v[0]*u[0]+v[1]*u[1]) for dim,v in zip(b[3:5],vb))
        gaps.append(abs(dx*u[0]+dy*u[1])-ra-rb-inflation)
    return max(gaps)

def trajectory_clearance(A,B,dt=0.1,max_speed=0.,strict=False):
    """Min sampled OBB separating-axis gap for arrays [x,y,yaw,length,width].

    Uses four separating axes in parallel for all sample times, equivalent to
    `obb_signed_clearance` up to floating-point rounding. This sampled margin
    is not a continuous collision certificate. Strict continuous-time bounds
    require a certified upper bound supplied by the caller; center speed alone
    is not enough for rotating vehicles and should include footprint motion.
    """
    if strict and max_speed<=0:
        raise ValueError('No continuous-time motion bound supplied')
    A=np.asarray(A,dtype=float)
    B=np.asarray(B,dtype=float)
    if A.ndim!=2 or B.ndim!=2 or A.shape!=B.shape or A.shape[1]!=5 or not len(A):
        raise ValueError('OBB traces must have identical nonempty (T,5) shape')
    if not (np.isfinite(A).all() and np.isfinite(B).all()):
        raise ValueError('non-finite OBB trace')
    if np.any(A[:,3:5]<=0) or np.any(B[:,3:5]<=0):
        raise ValueError('invalid OBB dimensions')
    ha=A[:,2]
    hb=B[:,2]
    ua=np.stack([np.cos(ha),np.sin(ha)],axis=-1)
    va=np.stack([-np.sin(ha),np.cos(ha)],axis=-1)
    ub=np.stack([np.cos(hb),np.sin(hb)],axis=-1)
    vb=np.stack([-np.sin(hb),np.cos(hb)],axis=-1)
    axes=np.stack([ua,va,ub,vb],axis=1)
    direction=(B[:,None,:2]-A[:,None,:2])
    centre_gap=np.abs(np.sum(direction*axes,axis=-1))
    def support(C,u,v):
        return (.5*C[:,3,None]*np.abs(np.sum(u[:,None,:]*axes,axis=-1))
                +.5*C[:,4,None]*np.abs(np.sum(v[:,None,:]*axes,axis=-1)))
    inflation=(dt*max_speed/2) if max_speed>0 else 0.
    signed_gaps=centre_gap-support(A,ua,va)-support(B,ub,vb)-inflation
    return float(np.min(np.max(signed_gaps,axis=1)))

def stop_slack(gap,v_follower,v_leader,delay,preacc,brake_follower,brake_leader):
    """Longitudinal RSS-inspired diagnostic; valid for fixed aligned following.
    Returns -inf for invalid actuator envelope. Never a 2D collision certificate.
    """
    if min(brake_follower,brake_leader)<=0 or delay<0:return -math.inf
    d=v_follower*delay+.5*preacc*delay**2+(max(0.,v_follower+preacc*delay))**2/(2*brake_follower)-max(0.,v_leader)**2/(2*brake_leader)
    return gap-max(0.,d)

def crossing_bounds(pa,pb):
    """Sharp Fréchet bounds; does not identify a cross-world joint law."""
    if not 0<=pa<=1 or not 0<=pb<=1:raise ValueError('probabilities outside [0,1]')
    return {'flip_lower':abs(pa-pb),'flip_upper':min(pa+pb,2-pa-pb),
            'harm_lower':max(0.,pa-pb),'harm_upper':min(pa,1-pb)}
