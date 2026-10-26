"""Record an observation-dependent R07 rollout on public LIBERO base tasks."""
from __future__ import annotations
import argparse
import base64
import contextlib
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid
import zipfile
from common import MODEL,OBS_KEYS,SCHEMA,WEIGHTS_SHA,export_policy,png,read,runtime_from_zip,sha,write

HERE=Path(__file__).resolve().parent
PUBLIC_SUITES=('libero_goal','libero_spatial','libero_object')

def task_slug(value):
    return re.sub(r'[^a-z0-9]+','_',str(value).casefold()).strip('_')

def resolve_task(tasks,requested):
    """Resolve registered metadata, never a guessed replacement or edit distance."""
    target=task_slug(requested)
    if not target:raise ValueError('Task name must not be empty')
    def names(task):return (task_slug(task.name),task_slug(Path(task.bddl_file).stem))
    tiers=(('name',lambda task:target in names(task)),
           ('instruction',lambda task:task_slug(task.language)==target),
           ('scene_prefix',lambda task:any(value.endswith('_'+target) for value in names(task))))
    for kind,predicate in tiers:
        matches=[task for task in tasks if predicate(task)]
        if len(matches)==1:return matches[0],kind
        if len(matches)>1:return None,'AMBIGUOUS_'+kind.upper()
    return None,'UNAVAILABLE'

def plan_tasks(benchmark,paths,config):
    """Check all requested public tasks before starting the expensive policy."""
    initial_source=config.get('initial_state_source','saved_init')
    if initial_source not in ('saved_init','seeded_reset'):raise ValueError('Unknown initial state source')
    registry=benchmark.get_benchmark_dict();cache={};catalogue={'suites':{},'requests':[]};planned=[]
    for spec in config['tasks']:
        suite_name=spec['suite']
        if suite_name not in PUBLIC_SUITES:raise ValueError('Use public base task suites only')
        request={'suite':suite_name,'requested_task':spec['task']};catalogue['requests'].append(request)
        try:
            if suite_name not in cache:
                if suite_name not in registry:raise LookupError('Public suite not registered')
                suite=registry[suite_name]();cache[suite_name]=[suite.get_task(i) for i in range(suite.n_tasks)]
                catalogue['suites'][suite_name]=[
                    {'name':task.name,'bddl_file':Path(task.bddl_file).name,'instruction':task.language}
                    for task in cache[suite_name]]
            task,kind=resolve_task(cache[suite_name],spec['task'])
            request['match']=kind
            if task is None:
                request['status']='SKIPPED';continue
            bddl=Path(paths['bddl_files'])/task.problem_folder/task.bddl_file
            init_file=(Path(paths['init_states'])/task.problem_folder/Path(task.init_states_file).name
                       if initial_source=='saved_init' else None)
            required=[('BDDL',bddl)]+([('init_states',init_file)] if init_file is not None else [])
            missing=[label for label,path in required if not path.is_file()]
            if missing:
                request.update(status='SKIPPED',error='Missing local public assets: '+', '.join(missing));continue
            request.update(status='RESOLVED',task=task.name,instruction=task.language,initial_state_source=initial_source)
            planned.append({'suite':suite_name,'requested_task':spec['task'],'task':task.name,
                            'instruction':task.language,'bddl':bddl,'init_file':init_file})
        except Exception as error:
            request.update(status='SKIPPED',error_type=type(error).__name__,error=str(error))
    return planned,catalogue

def redact_diagnostic(text,forbidden=()):
    text=re.sub(r'/(?:rsrch\d+/(?:home|scratch)|Users|home)/[^\s\"\'<>]+',
                lambda match:'[LOCAL_PATH]/'+match.group(0).rsplit('/',1)[-1],text)
    text=re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b','[REDACTED_EMAIL]',text)
    text=re.sub(r'\b(?:hf_|ghp_|github_pat_)[A-Za-z0-9_]{10,}\b','[REDACTED_TOKEN]',text)
    for token in forbidden:
        if token:text=re.sub(re.escape(token),'[PRIVATE]',text,flags=re.IGNORECASE)
    return text

@contextlib.contextmanager
def deadline(seconds):
    def expired(*_):raise TimeoutError('Simulator initialization deadline exceeded')
    previous=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)

class Worker:
    def __init__(self,python,runtime,implementation,log):
        self.log=Path(log).open('ab',buffering=0)
        self.process=subprocess.Popen([str(python),'-u',str(HERE/'policy_worker.py'),str(runtime),str(implementation)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,env=os.environ.copy(),bufsize=0)
        self.selector=selectors.DefaultSelector();self.selector.register(self.process.stdout,selectors.EVENT_READ)
        try:self.info=self.receive(600)
        except Exception:self.close();raise
    def receive(self,timeout):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if not self.selector.select(max(0,end-time.monotonic())):raise TimeoutError('Inference worker timed out')
            line=self.process.stdout.readline(1024*1024)
            if not line:raise RuntimeError('Inference worker stopped; inspect its private log')
            if not line.startswith(b'PAI_RPC '):self.log.write(line);continue
            result=json.loads(line[8:])
            if not result.get('ok'):raise RuntimeError(result.get('error','Inference failed'))
            return result
        raise TimeoutError('Inference worker timed out')
    def call(self,request):
        raw=(json.dumps(request,separators=(',',':'))+'\n').encode();offset=0
        while offset<len(raw):
            written=self.process.stdin.write(raw[offset:])
            if not written:raise BrokenPipeError('Inference worker pipe closed')
            offset+=written
        self.process.stdin.flush();return self.receive(60)
    def action(self,obs):
        import numpy as np
        encoded={}
        for key in OBS_KEYS:
            value=np.asarray(obs[key])
            if key.endswith('_image'):
                if value.shape!=(128,128,3) or value.dtype!=np.uint8:raise ValueError('Model camera contract must remain uint8 128x128 RGB')
                encoded[key]={'data':base64.b64encode(value.tobytes()).decode('ascii'),'shape':list(value.shape),'dtype':str(value.dtype)}
            else:encoded[key]=value.astype(np.float32).tolist()
        action=np.asarray(self.call({'kind':'action','observation':encoded})['action'],dtype=np.float32)
        if action.shape!=(7,) or not np.isfinite(action).all() or np.abs(action).max()>1.00001:raise ValueError('Invalid action received')
        return action
    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.selector.close()
        if self.process.stdin:self.process.stdin.close()
        if self.process.stdout:self.process.stdout.close()
        self.log.close()

def configure_libero(repo,run,initial_source='saved_init'):
    import yaml
    repo=Path(repo).resolve();package=repo/'libero/libero'
    if not (package/'envs').is_dir():raise FileNotFoundError('LIBERO package missing; check --libero-repo')
    old_file=Path(os.environ.get('LIBERO_CONFIG_PATH',str(Path.home()/'.libero')))/'config.yaml'
    old=(yaml.safe_load(old_file.read_text()) or {}) if old_file.is_file() else {}
    paths={'benchmark_root':str(package),'bddl_files':str(package/'bddl_files'),'init_states':str(package/'init_files'),
           'assets':str(package/'assets'),'datasets':str(Path(run)/'private/unused_datasets')}
    for key in ('bddl_files','init_states','assets'):
        if not Path(paths[key]).is_dir() and Path(old.get(key,'/nonexistent')).is_dir():paths[key]=old[key]
        if key=='init_states' and initial_source=='seeded_reset' and not Path(paths[key]).is_dir():
            paths[key]=str(Path(run)/'private/unused_init_states');Path(paths[key]).mkdir(parents=True,exist_ok=True)
        if not Path(paths[key]).is_dir():raise FileNotFoundError('Missing local LIBERO asset directory: '+key)
    Path(paths['datasets']).mkdir(parents=True,exist_ok=True)
    config=Path(run)/'private/libero_config';config.mkdir()
    (config/'config.yaml').write_text(yaml.safe_dump(paths),encoding='utf-8')
    os.environ['LIBERO_CONFIG_PATH']=str(config);sys.path.insert(0,str(repo))
    import pdb
    def reject_debugger(*_,**__):raise RuntimeError('Simulator requested an interactive debugger')
    pdb.set_trace=reject_debugger
    return paths

def display_frame(env,obs,rotation):
    import numpy as np
    images=[]
    for camera,key,size in (('agentview','agentview_image',512),('robot0_eye_in_hand','robot0_eye_in_hand_image',256)):
        try:rgb=np.asarray(env.sim.render(camera_name=camera,width=size,height=size,depth=False))
        except Exception:
            rgb=np.asarray(obs[key]);factor=size//rgb.shape[0];rgb=np.repeat(np.repeat(rgb,factor,0),factor,1)
        if rgb.shape!=(size,size,3) or rgb.dtype!=np.uint8:raise ValueError('Invalid rendered camera')
        images.append(np.ascontiguousarray(np.rot90(rgb,k=rotation//90)))
    if float(images[0].std())<1:raise ValueError('Main camera appears blank')
    canvas=np.zeros((512,768,3),dtype=np.uint8);canvas[:,:512]=images[0];canvas[:256,512:]=images[1]
    return canvas

def episode(env,worker,state,instruction,folder,config):
    import numpy as np
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    initial_source='seeded_reset' if state is None else 'saved_init'
    reset_seed=int(config.get('reset_seed',config['environment_seed'])) if state is None else None
    with deadline(120):
        # Use one underlying simulator reset instead of the wrapper's unbounded retry loop.
        if state is None:
            np.random.seed(reset_seed);obs=env.env.reset()
        else:
            env.env.reset();obs=env.set_init_state(state)
        for _ in range(config['warmup_actions']):obs,_,_,_=env.step([0.,0.,0.,0.,0.,0.,-1.])
    if env.check_success():raise ValueError('Initial state already satisfies the goal')
    horizon=worker.call({'kind':'reset','instruction':instruction}).get('horizon');frames=[]
    def save(t):
        name=f'frame_{len(frames):04d}.png';png(folder/name,display_frame(env,obs,config['display_rotation_degrees']));frames.append({'file':name,'t':float(t)})
    started=time.monotonic();save(0.);success=False
    for step in range(1,config['max_policy_actions']+1):
        action=worker.action(obs);obs,_,done,_=env.step(action.tolist())
        elapsed=time.monotonic()-started;save(elapsed);success=bool(env.check_success())
        if success or done or elapsed>=config['max_episode_wall_seconds']:break
    result={'origin':'live_libero','initial_goal':False,'success':success,'action_replay':False,'steps':step,'policy_calls':step,
            'initial_state_source':initial_source,'reset_seed':reset_seed,
            'wall_seconds':frames[-1]['t'],'horizon':horizon,'frames':frames,'warmup_actions_before_clip':config['warmup_actions'],
            'model_initialized_before_clip':True,'collision_metric':'not measured','timing':'recorded wall time, including inference waits'}
    write(folder/'episode.json',result);return result

def seal_return(output,run,source,attempts,selected,config,stage=None,forbidden=()):
    payload=Path(run)/'return_payload';payload.mkdir()
    if source.is_dir():shutil.copytree(source,payload/'policy_code')
    recorder=payload/'recorder_code';recorder.mkdir()
    for name in ('capture.py','common.py','policy_worker.py'):
        shutil.copyfile(HERE/name,recorder/name)
    write(recorder/'project.json',config)
    if selected:
        selected=dict(selected);folder=Path(selected.pop('folder'));shutil.copytree(folder,payload/'capture');selected['capture']='capture'
    versions={}
    for name in ('numpy','torch','robosuite','mujoco','PyYAML'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    write(payload/'environment_versions.json',versions)
    catalogue=Path(run)/'private/task_catalogue.json'
    if catalogue.is_file():
        text=redact_diagnostic(catalogue.read_text(encoding='utf-8'),forbidden)
        target=payload/'diagnostics/task_catalogue.json';target.parent.mkdir(exist_ok=True);target.write_text(text,encoding='utf-8')
    errors=[Path(run)/'private/fatal_error.txt',*sorted((Path(run)/'private/attempts').glob('*/private_error.txt'))]
    if any(path.is_file() for path in errors):
        diagnostic=payload/'diagnostics';diagnostic.mkdir(exist_ok=True)
        for path in [*errors,Path(run)/'private/worker.log']:
            if path.is_file():
                name=path.relative_to(Path(run)/'private').as_posix().replace('/','__')
                with path.open('rb') as stream:
                    stream.seek(0,2);stream.seek(max(0,stream.tell()-65536));text=stream.read().decode('utf-8',errors='replace')
                (diagnostic/name).write_text(redact_diagnostic(text,forbidden),encoding='utf-8')
    manifest={'schema':SCHEMA,'run_id':Path(run).name,'origin':'live_libero','status':'RECORDING_READY' if selected else 'NO_SUCCESS',
        'weights_sha256':WEIGHTS_SHA,'new_training':False,'attempts':attempts,'selected':selected,'config':config,
        'failure_stage':None if selected else stage,
        'selection_rule':'first successful complete clip fitting the 1-3 minute video at up to 2x playback',
        'files_sha256':{p.relative_to(payload).as_posix():sha(p) for p in payload.rglob('*') if p.is_file()}}
    write(payload/'recording_manifest.json',manifest)
    archive=Path(run)/'recording_return.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(payload.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(payload))
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise RuntimeError('Recording ZIP CRC failure')
    target=Path(output)/'recording_return.zip';temporary=target.with_suffix('.zip.partial');shutil.copyfile(archive,temporary);temporary.replace(target)
    (Path(output)/'recording_return.zip.sha256').write_text(sha(target)+'  recording_return.zip\n')
    print('PAI_STATUS='+manifest['status'],flush=True);print('PAI_RETURN='+str(target),flush=True);print('PAI_PRIVATE_RUN='+str(run),flush=True)
    return bool(selected)

def main(argv=None):
    with contextlib.ExitStack() as resources:
        return run_capture(argv,resources)

def run_capture(argv,resources):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--settings',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--runtime',type=Path);p.add_argument('--policy-python');p.add_argument('--libero-repo',type=Path)
    p.add_argument('--project',type=Path,default=HERE/'project.json');p.add_argument('--new-run',action='store_true');a=p.parse_args(argv)
    settings=read(a.settings) if a.settings else {};output=Path(a.output or settings.get('hpc_output','recording_output')).resolve()
    output.mkdir(parents=True,exist_ok=True);lock=resources.enter_context((output/'run.lock').open('w'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'recording_return.zip').exists() and not a.new_run:
        with zipfile.ZipFile(output/'recording_return.zip') as z:prior= json.loads(z.read('recording_manifest.json'))
        if prior.get('status')=='RECORDING_READY' and prior.get('schema')==SCHEMA and prior.get('weights_sha256')==WEIGHTS_SHA:
            if sha(output/'recording_return.zip')!=(output/'recording_return.zip.sha256').read_text().split()[0]:raise ValueError('Cached return ZIP changed')
            if prior.get('config')!=read(a.project):raise ValueError('Project settings changed; use --new-run or another --output')
            print('PAI_STATUS=RECORDING_READY (existing recording preserved)');print('PAI_RETURN='+str(output/'recording_return.zip'));return 0
    run=output/'runs'/uuid.uuid4().hex[:12];(run/'private').mkdir(parents=True)
    config=read(a.project)
    if not 1<=config['max_policy_actions']<=300:raise ValueError('Action budget must be 1..300')
    python=a.policy_python or settings.get('policy_python');repo=a.libero_repo or settings.get('libero_repo')
    if not python or not repo:raise ValueError('Specify policy Python and LIBERO repository')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',WANDB_DISABLED='true',
        TOKENIZERS_PARALLELISM='false',PYTHONBREAKPOINT='0',MUJOCO_GL=os.environ.get('MUJOCO_GL','egl'),PYOPENGL_PLATFORM=os.environ.get('PYOPENGL_PLATFORM','egl'))
    source=run/'private/policy_code';attempts=[];selected=None;worker=None;stage='runtime_verification'
    try:
        if a.runtime:
            runtime=a.runtime.resolve()
            if sha(runtime/MODEL)!=WEIGHTS_SHA:raise ValueError('Runtime weights differ from R07')
            identity=read(HERE/'policy_identity.json');source.mkdir()
            for name,digest in identity['files_sha256'].items():
                if sha(HERE/name)!=digest:raise ValueError('Public inference source changed')
                shutil.copyfile(HERE/name,source/name)
            shutil.copyfile(HERE/'policy_identity.json',source/'policy_identity.json');implementation=source/'policy_impl.py'
        else:
            runtime=runtime_from_zip(settings,run);implementation=export_policy(runtime,source,settings.get('forbidden_public_tokens',()))
        initial_source=config.get('initial_state_source','saved_init')
        stage='libero_import';paths=configure_libero(repo,run,initial_source)
        import numpy as np
        import torch
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        stage='task_preflight';np.random.seed(config['environment_seed'])
        with (run/'private/task_preflight_stdout.log').open('w') as stream,contextlib.redirect_stdout(stream):
            planned,catalogue=plan_tasks(benchmark,paths,config)
        write(run/'private/task_catalogue.json',catalogue)
        for request in catalogue['requests']:
            print('PAI_TASK_PREFLIGHT='+request['status']+' '+request['suite']+'/'+request.get('task',request['requested_task']),flush=True)
        if not planned:raise LookupError('None of the requested public tasks resolved. See diagnostics/task_catalogue.json in the return ZIP.')
        print('PAI_INITIAL_STATE='+initial_source,flush=True)
        stage='policy_startup'
        worker=Worker(python,runtime,implementation,run/'private/worker.log')
        stage='task_rollout'
        for spec in planned:
            try:states=(torch.load(spec['init_file'],map_location='cpu',weights_only=False) if spec['init_file'] is not None else None)
            except Exception:
                folder=run/'private/attempts'/f'attempt_{len(attempts)+1:03d}';folder.mkdir(parents=True,exist_ok=True)
                (folder/'private_error.txt').write_text(traceback.format_exc())
                attempts.append({'suite':spec['suite'],'task':spec['task'],'status':'TECHNICAL_FAILURE','success':False,'stage':'init_state_load'})
                continue
            for init_id in config['init_ids']:
                episode_seed=config['environment_seed']+init_id
                attempt={'suite':spec['suite'],'task':spec['task'],'init_id':init_id,'initial_state_source':initial_source,
                         'seed':episode_seed,'status':'TECHNICAL_FAILURE','success':False};attempts.append(attempt)
                folder=run/'private/attempts'/f'attempt_{len(attempts):03d}';env=None
                try:
                    if states is not None and init_id>=len(states):raise IndexError('Public init unavailable')
                    np.random.seed(episode_seed)
                    with deadline(120):
                        env=OffScreenRenderEnv(bddl_file_name=str(spec['bddl']),camera_heights=128,camera_widths=128,control_freq=config['control_frequency_hz'],horizon=1000)
                        env.seed(episode_seed)
                    state=np.asarray(states[init_id]).copy() if states is not None else None
                    result=episode(env,worker,state,spec['instruction'],folder,{**config,'reset_seed':episode_seed})
                    eligible=result['success'] and result['wall_seconds']<=240
                    attempt.update(status='COMPLETE',success=result['success'],video_eligible=eligible,steps=result['steps'],wall_seconds=result['wall_seconds'])
                    if eligible:selected={'suite':spec['suite'],'task':spec['task'],'instruction':spec['instruction'],'init_id':init_id,
                                          'initial_state_source':initial_source,'seed':episode_seed,'folder':str(folder)}
                except Exception:
                    folder.mkdir(parents=True,exist_ok=True);(folder/'private_error.txt').write_text(traceback.format_exc())
                finally:
                    if env is not None:
                        try:env.close()
                        except Exception:(run/'private/close_error.txt').write_text(traceback.format_exc())
                    write(run/'private/attempts.json',attempts)
                print(f"PAI_ATTEMPT={len(attempts)} {attempt['status']} success={int(attempt['success'])}",flush=True)
                if selected:break
            if selected:break
    except Exception:
        (run/'private/fatal_error.txt').write_text(traceback.format_exc());attempts.append({'status':'TECHNICAL_FAILURE','success':False,'stage':stage,'error':'See diagnostics/fatal_error.txt in the return ZIP.'})
    finally:
        if worker:worker.close()
    if selected:
        for f in Path(selected['folder']).glob('private_*'):f.unlink()
    return 0 if seal_return(output,run,source,attempts,selected,config,stage,settings.get('forbidden_public_tokens',())) else 2

if __name__=='__main__':raise SystemExit(main())
