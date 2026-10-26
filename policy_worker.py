"""Local inference worker. Observations in; seven learned actions out."""
from __future__ import annotations
import base64
import contextlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import traceback

def emit(value):
    sys.stdout.write('PAI_RPC '+json.dumps(value,separators=(',',':'))+'\n');sys.stdout.flush()

def main():
    runtime=Path(sys.argv[1]).resolve();implementation=Path(sys.argv[2]).resolve()
    os.environ.update(PAI_POLICY_RUNTIME=str(runtime),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',WANDB_DISABLED='true')
    os.chdir(runtime);sys.path[:0]=[str(implementation.parent),str(runtime/'vendor'),str(runtime)]
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import numpy as np
            spec=importlib.util.spec_from_file_location('pai_recorded_policy',implementation)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            policy=module.MyPolicy()
            if not policy.torch.cuda.is_available():raise RuntimeError('An allocated HPC GPU is required')
        emit({'ok':True,'ready':True,'torch_version':str(policy.torch.__version__)})
        for line in sys.stdin:
            request=json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if request['kind']=='reset':
                    parameters=inspect.signature(policy.reset).parameters
                    kwargs={'instruction':request['instruction']}
                    if 'seed' in parameters:kwargs['seed']=2026
                    policy.reset(**kwargs);overlap=getattr(policy,'_g09h_overlap',None)
                    horizon=getattr(overlap,'horizon',None)
                    response={'ok':True,'horizon':int(horizon) if isinstance(horizon,int) else None}
                elif request['kind']=='action':
                    obs={}
                    for key,value in request['observation'].items():
                        if isinstance(value,dict):
                            raw=base64.b64decode(value['data'],validate=True)
                            obs[key]=np.frombuffer(raw,dtype=np.dtype(value['dtype'])).reshape(value['shape']).copy()
                        else:obs[key]=np.asarray(value,dtype=np.float32)
                    action=np.asarray(policy.get_action(obs),dtype=np.float32)
                    if action.shape!=(7,) or not np.isfinite(action).all() or np.abs(action).max()>1.00001:raise ValueError('Invalid learned action')
                    response={'ok':True,'action':action.tolist()}
                else:raise ValueError('Unknown inference request')
            emit(response)
    except Exception:
        traceback.print_exc(file=sys.stderr);emit({'ok':False,'error':'Inference failed; inspect the private worker log on HPC.'});return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
