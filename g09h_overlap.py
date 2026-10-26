"""Generic receding horizon; reobserve after a gripper transition, without task IDs."""
from collections import deque
import numpy as np


def consensus(chunks):
    a=np.asarray(chunks)
    if a.ndim!=3 or a.shape[1:]!=(20,7) or len(a) not in (1,3) or a.dtype!=np.float32 or not np.isfinite(a).all():
        raise ValueError('Invalid chunks')
    a=np.clip(a,-1,1)
    if len(a)==1:
        return a[0].copy()
    distance=np.mean(np.abs(a[:,None,:10,:6]-a[None,:,:10,:6]),axis=(2,3))
    ref=int(np.argmin(distance.sum(1))); result=a[ref].copy()
    for t in range(20):
        same=np.sign(a[:,t,6])==np.sign(a[ref,t,6])
        result[t,:6]=a[same,t,:6].mean(0)
    return result.astype(np.float32)


class OverlapActions:
    def __init__(self, old_weight=.25, horizon=5, smooth=False, ema_alpha=1., transition_replan=False):
        if horizon not in (1,3,5,10) or not 0 <= old_weight <= 1 or not 0 < ema_alpha <= 1:
            raise ValueError('Invalid inference configuration')
        self.weight=np.float32(old_weight); self.horizon=horizon
        self.smooth=bool(smooth); self.alpha=np.float32(ema_alpha)
        self.transition=bool(transition_replan); self.reset()

    def reset(self):
        self.queue=deque(); self.previous=None; self.last=None
        self.consumed=0; self.predictions=0; self.actions=0; self.transition_replans=0

    def needs_chunk(self):
        return not self.queue

    def push(self, chunk):
        if self.queue:
            raise ValueError('Actions are still queued')
        a=np.asarray(chunk)
        if a.shape != (20,7) or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError('Invalid chunk')
        a=np.clip(a,-1,1).copy(); out=a[:self.horizon].copy()
        if self.smooth:
            for t in range(1,self.horizon):
                if np.sign(a[t-1,6]) == np.sign(a[t,6]) == np.sign(a[t+1,6]):
                    out[t,:6]=.25*a[t-1,:6]+.5*a[t,:6]+.25*a[t+1,:6]
        if self.previous is not None and self.weight:
            old=self.previous[self.consumed:self.consumed+self.horizon]
            if len(old) != len(out):
                raise ValueError('Chunk overlap ran past the previous prediction')
            same=np.sign(old[:,6]) == np.sign(out[:,6])
            out[same,:6]=(np.float32(1)-self.weight)*out[same,:6]+self.weight*old[same,:6]
        if self.transition:
            prior=None if self.last is None else np.sign(self.last[6])
            for t,x in enumerate(out):
                sign=np.sign(x[6])
                if prior is not None and sign != prior:
                    # Execute the transition once, then obtain a new observation.
                    if t+1 < len(out):
                        out=out[:t+1]; self.transition_replans+=1
                    break
                prior=sign
        self.previous=a; self.consumed=0; self.predictions+=1
        self.queue.extend(x.copy() for x in out)

    def pop(self):
        if not self.queue:
            raise ValueError('Prediction required')
        a=self.queue.popleft().copy()
        if self.last is not None and self.alpha < 1 and np.sign(a[6]) == np.sign(self.last[6]):
            a[:3]=self.alpha*a[:3]+(np.float32(1)-self.alpha)*self.last[:3]
        self.last=a.copy(); self.consumed+=1; self.actions+=1
        return a

# tampered

# tampered

# tampered
