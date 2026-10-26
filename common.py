"""Shared, path-independent utilities for the course demonstration."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
import struct
import zipfile
import zlib

WEIGHTS_SHA = '96c0a1836a231a811122da19a355a571f7cf332f1209b4c51944e8ce9cc76a1f'
ZIP_SHA = '4e8284e32e0124def96744ae88db43cc9e827b2fdcc38100e108e55515397e5f'
MODEL = 'model_weights/pi05_libero_finetuned_v044/model.safetensors'
SCHEMA = 'pai_live_capture_v2'
OBS_KEYS = ('agentview_image','robot0_eye_in_hand_image','robot0_joint_pos',
            'robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')
PUBLIC_FILES = ('capture.py','common.py','policy_worker.py','presentation.py','requirements_presentation.txt','project.json')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def safe_extract(archive,destination):
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target=(destination/item.filename).resolve()
            if target!=destination and destination not in target.parents:raise ValueError('Unsafe ZIP path')
            if stat.S_ISLNK(item.external_attr>>16):raise ValueError('ZIP symlinks are not supported')
        damaged=z.testzip()
        if damaged:raise ValueError('ZIP CRC failure: '+damaged)
        z.extractall(destination)

def privacy_findings(text,forbidden=()):
    patterns=(r'/rsrch\d+/(?:home|scratch)/[^\s\"\'<>]+',r'/(?:Users)/[A-Za-z0-9_.-]+',
              r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
              r'\b(?:hf_|ghp_|github_pat_)[A-Za-z0-9_]{12,}\b')
    found=[pattern for pattern in patterns if re.search(pattern,text)]
    found.extend('private token' for token in forbidden if token and len(token)>=5 and token.casefold() in text.casefold())
    return found

def png(path,image):
    import numpy as np
    a=np.ascontiguousarray(image,dtype=np.uint8)
    if a.ndim!=3 or a.shape[2]!=3:raise ValueError('Expected RGB pixels')
    height,width,_=a.shape
    def chunk(kind,payload):
        return struct.pack('>I',len(payload))+kind+payload+struct.pack('>I',zlib.crc32(kind+payload)&0xffffffff)
    raw=b''.join(b'\0'+row.tobytes() for row in a)
    data=(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))
          +chunk(b'IDAT',zlib.compress(raw,3))+chunk(b'IEND',b''))
    Path(path).write_bytes(data)

def export_policy(runtime,destination,forbidden=()):
    """Preserve MyPolicy and its bootstrap, and omit the HTTP endpoints."""
    runtime=Path(runtime);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    source=(runtime/'policy_server.py').read_text(encoding='utf-8');tree=ast.parse(source)
    original=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MyPolicy')
    lines=source.splitlines(keepends=True);prefix=''.join(lines[:original.end_lineno])
    for n in tree.body:
        if n.lineno>=original.lineno:break
        if isinstance(n,ast.If) and '__main__' in ast.get_source_segment(source,n):
            raise ValueError('Unexpected executable entry point before MyPolicy')
    futures=[n for n in tree.body if isinstance(n,ast.ImportFrom) and n.module=='__future__']
    for n in reversed(futures):prefix=prefix.replace(''.join(lines[n.lineno-1:n.end_lineno]),'',1)
    future_text=''.join(ast.get_source_segment(source,n)+'\n' for n in futures)
    generated=(future_text+"import os as _pai_os\n"
               "__file__ = _pai_os.path.join(_pai_os.environ['PAI_POLICY_RUNTIME'], 'policy_server.py')\n"+prefix)
    after=next(n for n in ast.parse(generated).body if isinstance(n,ast.ClassDef) and n.name=='MyPolicy')
    if ast.dump(original,include_attributes=False)!=ast.dump(after,include_attributes=False):raise ValueError('Inference class changed during export')
    modules={p.stem:p for p in runtime.glob('*.py') if p.name!='policy_server.py'}
    sources={'policy_impl.py':generated};pending=[generated]
    while pending:
        for node in ast.walk(ast.parse(pending.pop())):
            names=[a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else [node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                path=modules.get(name)
                if path and path.name not in sources:
                    text=path.read_text(encoding='utf-8');sources[path.name]=text;pending.append(text)
    for name,text in sources.items():
        if privacy_findings(text,forbidden):raise ValueError('Inference source needs privacy cleanup: '+name)
        (destination/name).write_text(text,encoding='utf-8')
    write(destination/'policy_identity.json',{'model':'participant-trained pi0.5 R07','weights_sha256':WEIGHTS_SHA,
        'original_server_sha256':sha(runtime/'policy_server.py'),'original_MyPolicy_ast_preserved':True,
        'files_sha256':{name:sha(destination/name) for name in sources}})
    return destination/'policy_impl.py'

def runtime_from_zip(settings,run):
    archive=Path(settings['r07_zip'])
    if not archive.is_file():raise FileNotFoundError('Missing original R07 ZIP: '+str(archive))
    print('PAI_VERIFYING_R07_ZIP',flush=True)
    if sha(archive)!=ZIP_SHA:raise ValueError('Original R07 ZIP SHA-256 mismatch')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();servers=[n for n in names if n.endswith('policy_server.py') and '/vendor/' not in '/'+n]
        server=min(servers,key=lambda n:len(Path(n).parts));prefix=server[:-len('policy_server.py')]
        def valid(runtime):
            if not (runtime/MODEL).is_file() or sha(runtime/MODEL)!=WEIGHTS_SHA:return False
            for name in names:
                if not name.startswith(prefix):continue
                relative=name[len(prefix):];path=Path(relative)
                if ((len(path.parts)==1 and path.suffix=='.py') or
                    (relative.startswith('model_weights/pi05_libero_finetuned_v044/') and path.suffix=='.json')):
                    target=runtime/relative
                    if not target.is_file() or target.read_bytes()!=z.read(name):return False
            return True
        runtime=Path(settings['r07_runtime']) if settings.get('r07_runtime') else Path(run)/'unused'
        if not valid(runtime):
            size=sum(item.file_size for item in z.infolist())
            if shutil.disk_usage(run).free<size+512*1024**2:raise OSError('Insufficient free space for R07 extraction')
            base=Path(run)/'private/runtime';safe_extract(archive,base);runtime=base/prefix
            if not valid(runtime):raise ValueError('Extracted R07 runtime verification failed')
    return runtime

def verify_capture(root,test_mode=False):
    root=Path(root).resolve();m=read(root/'recording_manifest.json')
    origin='test_fixture' if test_mode else 'live_libero'
    if m.get('schema')!=SCHEMA or m.get('origin')!=origin or m.get('status')!='RECORDING_READY':raise ValueError('Successful live capture required')
    if m.get('weights_sha256')!=WEIGHTS_SHA:raise ValueError('Capture model identity differs')
    for name,digest in m['files_sha256'].items():
        path=(root/name).resolve()
        if root not in path.parents or not path.is_file() or sha(path)!=digest:raise ValueError('Capture checksum mismatch: '+name)
    selected=m['selected'];capture=(root/selected['capture']).resolve()
    if root not in capture.parents:raise ValueError('Invalid capture directory')
    ep=read(capture/'episode.json');frames=ep['frames'];times=[f['t'] for f in frames]
    if selected.get('initial_state_source')=='seeded_reset':
        config=m.get('config',{});expected=config.get('environment_seed',0)+selected['init_id']
        if (config.get('initial_state_source')!='seeded_reset' or ep.get('initial_state_source')!='seeded_reset'
                or selected.get('seed')!=expected or ep.get('reset_seed')!=expected):
            raise ValueError('Reset seed or initial state source differs from the recording')
    import math
    if ep.get('origin')!=origin or not ep.get('success') or ep.get('action_replay') or ep.get('initial_goal') is not False:raise ValueError('Invalid rollout provenance')
    if not 1<=ep['steps']<=300 or ep['policy_calls']!=ep['steps'] or len(frames)!=ep['steps']+1:raise ValueError('Action/frame mismatch')
    if times[0]!=0 or not all(math.isfinite(t) for t in times) or any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Invalid timestamps')
    if abs(ep['wall_seconds']-times[-1])>1e-6:raise ValueError('Duration mismatch')
    for frame in frames:
        path=(capture/frame['file']).resolve()
        if capture not in path.parents or path.relative_to(root).as_posix() not in m['files_sha256']:raise ValueError('Unverified frame')
    wins=[a for a in m['attempts'] if a['status']=='COMPLETE' and a['success']]
    if not any(a['suite']==selected['suite'] and a['task']==selected['task'] and a['init_id']==selected['init_id'] for a in wins):raise ValueError('Selected attempt differs from ledger')
    return m,ep,capture
