"""Create a course MP4, PDF and public code from a verified live capture."""
from __future__ import annotations
import argparse
import bisect
import importlib.metadata
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import textwrap
import zipfile
from xml.sax.saxutils import escape
from common import privacy_findings,read,sha,verify_capture,write

HERE=Path(__file__).resolve().parent
WIDTH,HEIGHT,FPS=1024,576,12

def ffmpeg_binary():
    binary=shutil.which('ffmpeg')
    if binary:return binary
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as error:raise RuntimeError('Install requirements_presentation.txt on the Mac') from error

def fonts():
    from PIL import ImageFont
    options=('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/System/Library/Fonts/Supplemental/Arial.ttf','/Library/Fonts/Arial.ttf')
    path=next((p for p in options if Path(p).is_file()),None)
    return {size:ImageFont.truetype(path,size) if path else ImageFont.load_default(size=size) for size in (16,18,22,34)}

def card(title,lines,account,test_mode=False):
    from PIL import Image,ImageDraw
    f=fonts();image=Image.new('RGB',(WIDTH,HEIGHT),'#10283d');draw=ImageDraw.Draw(image)
    draw.rectangle((0,0,WIDTH,12),fill='#55c0bd')
    draw.text((45,40),'PAI FINAL ASSIGNMENT | '+account,font=f[18],fill='#bdd7e6');y=108
    for part in textwrap.wrap(title,39):draw.text((45,y),part,font=f[34],fill='white');y+=46
    y+=18
    for line in lines:
        for part in textwrap.wrap(line,68):draw.text((48,y),part,font=f[22],fill='#dce9f1');y+=32
        y+=12
    if test_mode:
        draw.rectangle((0,HEIGHT-36,WIDTH,HEIGHT),fill='#a31529')
        draw.text((15,HEIGHT-32),'TEST ONLY - NOT A REAL ROLLOUT',font=f[18],fill='white')
    return image

def make_video(capture,m,ep,account,target,test_mode=False):
    from PIL import Image,ImageDraw
    duration=ep['wall_seconds'];speed=1 if duration<=120 else 2
    demo_duration=duration/speed+3.;intro_duration,method_duration=10,15
    # PAI_VIDEO_FRAME_TIMING_FIX_V1: fill the minimum length in whole frames.
    demo_frames=math.ceil(demo_duration*FPS)
    end_frames=max(10*FPS,60*FPS-(intro_duration+method_duration)*FPS-demo_frames)
    total_frames=(intro_duration+method_duration)*FPS+demo_frames+end_frames
    expected=total_frames/FPS
    if not 60<=expected<=180:raise ValueError('Complete rollout cannot fit the 1-3 minute video at 1x/2x playback')
    completed=[a for a in m['attempts'] if a['status']=='COMPLETE'];successes=sum(a['success'] for a in completed)
    errors=len(m['attempts'])-len(completed)
    intro=card('Camera-guided tabletop tidying',[
        'A participant-trained vision-language-action policy controls a robot.',
        'The robot uses camera images, robot state and a language instruction.',
        'Simulation demonstration; no new training was performed.'
    ],account,test_mode)
    method=card('Observe, predict, act, observe again',[
        'The original R07 learned policy and preprocessing are preserved.',
        'Short action chunks are followed by updated camera observations.',
        'No saved action trajectory is played back.',
        'The model is initialized before recording begins.'
    ],account,test_mode)
    ending=card('Result and scope',[
        f'Goal achieved in {ep["steps"]} learned policy actions.',
        (f'Initial state: simulator reset, seed {m["selected"]["seed"]}.'
         if m['selected'].get('initial_state_source')=='seeded_reset' else f'Saved benchmark init: {m["selected"]["init_id"]}.'),
        f'Completed attempts: {len(completed)}; successes: {successes}; technical errors: {errors}.',
        'The first eligible success is shown; collection then stops.',
        'This demonstration is not a generalization benchmark.'
    ],account,test_mode)
    f=fonts();frames=ep['frames'];times=[frame['t'] for frame in frames];previous=None;pixels=None
    def actual_frames():
        nonlocal previous,pixels
        for j in range(demo_frames):
            t=min(duration,j/FPS*speed);index=min(len(frames)-1,bisect.bisect_right(times,t)-1)
            if index!=previous:
                with Image.open(capture/frames[index]['file']) as image:pixels=image.convert('RGB').copy()
                if pixels.size!=(768,512):raise ValueError('Recorded image layout differs')
                previous=index
            image=Image.new('RGB',(WIDTH,HEIGHT),'#10283d');draw=ImageDraw.Draw(image)
            draw.text((16,6),f'Actual rollout | {speed}x recorded wall time',font=f[22],fill='white')
            instruction=m['selected']['instruction']
            draw.text((16,37),instruction,font=f[16],fill='#bdd7e6')
            image.paste(pixels,(0,64))
            for k,line in enumerate(('R07 / pi0.5','Live camera input',f'Action {index}/{ep["steps"]}',f'Wall time {t:.1f}s',f'Playback: {speed}x')):
                draw.text((790,105+k*45),line,font=f[18],fill='white')
            draw.text((525,350),'WRIST CAMERA',font=f[18],fill='#bdd7e6')
            draw.text((525,390),'Model initialized',font=f[16],fill='#bdd7e6');draw.text((525,418),'before this clip.',font=f[16],fill='#bdd7e6')
            if t>=duration:
                draw.text((790,395),'GOAL ACHIEVED',font=f[18],fill='#55c0bd');draw.text((790,428),'Final frame held',font=f[16],fill='white')
            if test_mode:
                draw.rectangle((0,HEIGHT-32,WIDTH,HEIGHT),fill='#a31529');draw.text((15,HEIGHT-29),'TEST ONLY - NOT A REAL ROLLOUT',font=f[16],fill='white')
            yield image
    target=Path(target);partial=target.with_suffix('.mp4.partial');log=target.with_suffix('.ffmpeg.log')
    command=[ffmpeg_binary(),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{WIDTH}x{HEIGHT}',
        '-r',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-map_metadata','-1',
        '-metadata','title=PAI final assignment '+account,'-movflags','+faststart','-f','mp4',str(partial)]
    with log.open('wb') as err:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=err)
        try:
            for image,seconds in ((intro,intro_duration),(method,method_duration)):
                raw=image.tobytes()
                for _ in range(seconds*FPS):process.stdin.write(raw)
            for image in actual_frames():process.stdin.write(image.tobytes())
            raw=ending.tobytes()
            for _ in range(end_frames):process.stdin.write(raw)
            process.stdin.close()
            if process.wait()!=0:raise RuntimeError('Video encoding failed: '+str(log))
        except Exception:
            if process.poll() is None:process.kill();process.wait()
            raise
    if not partial.is_file() or partial.stat().st_size<1000:raise RuntimeError('Video output is incomplete')
    partial.replace(target)
    check=subprocess.run([ffmpeg_binary(),'-hide_banner','-i',str(target),'-frames:v','1','-f','null','-'],capture_output=True,text=True)
    match=re.search(r'Duration: (\d+):(\d+):(\d+(?:\.\d+)?)',check.stderr)
    if check.returncode or not match:raise RuntimeError('Encoded MP4 could not be decoded')
    measured=int(match[1])*3600+int(match[2])*60+float(match[3])
    if not 60<=measured<=180 or abs(measured-expected)>.2:raise ValueError('MP4 duration differs from requirements')
    log.unlink(missing_ok=True)
    return {'duration_seconds':measured,'playback_speed':speed,'recorded_wall_seconds':duration,'final_state_hold_seconds':3,'fps':FPS,'resolution':[WIDTH,HEIGHT]}

def make_report(capture,m,ep,video,account,target,test_mode=False):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Image,PageBreak
    font='PAIReportSans'
    font_file=HERE/'assets/fonts/PAIReportSans-Regular.ttf'
    if not font_file.is_file():raise FileNotFoundError('Bundled Japanese font is missing: assets/fonts/PAIReportSans-Regular.ttf')
    pdfmetrics.registerFont(TTFont(font,str(font_file)))
    body=ParagraphStyle('body',fontName=font,fontSize=9.5,leading=15,wordWrap='CJK',spaceAfter=7)
    heading=ParagraphStyle('heading',parent=body,fontSize=12,leading=18,spaceBefore=8,textColor=colors.HexColor('#235a84'))
    title=ParagraphStyle('title',parent=body,fontSize=19,leading=27,spaceAfter=10)
    def p(text,style=body):return Paragraph(escape(str(text)),style)
    def grid(rows):
        table=Table([[p(a),p(b)] for a,b in rows],colWidths=[82,420],hAlign='LEFT')
        table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(0,-1),colors.HexColor('#edf3f8')),
            ('GRID',(0,0),(-1,-1),.4,colors.HexColor('#d5dfe9')),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),4)]));return table
    completed=[a for a in m['attempts'] if a['status']=='COMPLETE'];successes=sum(a['success'] for a in completed);errors=len(m['attempts'])-len(completed)
    selected=m['selected'];story=[]
    initial_label=(f'reset trial {selected["init_id"]} / seed {selected["seed"]}'
                   if selected.get('initial_state_source')=='seeded_reset' else f'init {selected["init_id"]}')
    if test_mode:story.append(p('TEST ONLY - 実際のロボット実行結果ではありません',heading))
    story.extend([p('カメラで見直す卓上片付け',title),p('PAI最終課題 / OmniCampus ID: '+account),
        p('1. 目的',heading),p('カメラ画像と自然言語の指示から操作を生成し、シミュレータ上のロボットアームで物体を把持・配置する。以前に追加学習した方策を再利用し、最新の観測に基づく実行を示す。'),
        p('2. 使用したAIと実装',heading),
        grid([('方策','参加者が以前にLoRA追加学習したpi0.5（R07）。今回の新規学習はなし。'),
              ('入力','主カメラ・手先カメラ画像、robot state、自然言語指示。'),
              ('出力','7次元action。保存したaction列の再生は使用しない。'),
              ('実行方法','元の画像処理・正規化・推論クラスを保持し、短い行動区間ごとに最新観測から再計画する。'+(f' 実行長は{ep["horizon"]} actions。' if ep.get('horizon') else ''))]),
        p('3. 実行結果',heading),
        grid([('公開タスク',selected['task']),('指示',selected['instruction']),
              ('試行の内訳',f'完了{len(completed)}試行、成功{successes}試行、技術失敗{errors}件。固定順で試し、動画条件を満たす初成功で終了した。'),
              ('掲載例',f'{initial_label} / {ep["steps"]} policy actions / goal達成。'),
              ('動画',f'{video["duration_seconds"]:.1f}秒。実行部分は記録したwall timeの{video["playback_speed"]}倍速。モデル起動と初期化は動画外。')]),
        PageBreak(),p('4. 開始状態と終了状態',heading)])
    first=capture/ep['frames'][0]['file'];last=capture/ep['frames'][-1]['file']
    screenshots=Table([[Image(str(first),width=242,height=242*2/3),Image(str(last),width=242,height=242*2/3)],[p('開始状態'),p('goal達成後の状態')]],colWidths=[251,251])
    story.extend([screenshots,p('5. 工夫した点',heading),
        p('以前に追加学習した重みと、短い区間で観測を見直す実行方法を再利用した。推論環境とシミュレータを別プロセスで接続し、モデルの前処理やactionの意味を変えずに録画機能を追加した。'),
        p('推論の待ち時間を含めた実行を記録し、成功例の選択方法、試行数、再生速度を明記した。公開用コードと、個人パス・ログ・重みを分離した。録画処理と提出用コードの整理には生成AIの支援を利用した。'),
        p('6. 限界と今後',heading),
        p('動画は固定順の候補から選んだ成功例であり、未知条件での成功率を推定する実験ではない。実機と安全性は未検証。今後は別の初期配置・視点で評価し、接触操作の失敗に対する教師を追加したい。'),
        p('7. 出典と再現',heading),
        p('環境はLIBERO、基盤はLeRobot版pi0.5。公開コードのREADMEに実行手順と学習済み重みのSHA-256を記載する。重みと外部runtimeのvendorは同梱していない。'),
        p('https://github.com/Lifelong-Robot-Learning/LIBERO'),p('https://huggingface.co/lerobot/pi05_libero_finetuned_v044')])
    def footer(canvas,document):
        canvas.setAuthor('');canvas.setTitle('PAI final assignment '+account);canvas.setFont('Helvetica',8)
        canvas.setFillColor(colors.HexColor('#607080'));canvas.drawString(45,23,'PAI final assignment | '+account);canvas.drawRightString(A4[0]-45,23,str(document.page))
        if test_mode:
            canvas.setFillColor(colors.HexColor('#a31529'));canvas.drawCentredString(A4[0]/2,12,'TEST ONLY - NOT A REAL ROLLOUT')
    doc=SimpleDocTemplate(str(target),pagesize=A4,leftMargin=45,rightMargin=45,topMargin=35,bottomMargin=36,title='PAI final assignment '+account,author='')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)

def make_public_code(root,destination,m,account,forbidden=()):
    destination.mkdir()
    # Export the recorder sources that actually produced this capture.
    for name in ('capture.py','common.py','policy_worker.py','project.json'):
        shutil.copyfile(root/'recorder_code'/name,destination/name)
    for name in ('presentation.py','requirements_presentation.txt'):shutil.copyfile(HERE/name,destination/name)
    identity=read(root/'policy_code/policy_identity.json')
    for name,digest in identity['files_sha256'].items():
        source=root/'policy_code'/name
        if sha(source)!=digest:raise ValueError('Recorded inference source changed')
        shutil.copyfile(source,destination/name)
    write(destination/'policy_identity.json',identity)
    shutil.copyfile(root/'environment_versions.json',destination/'environment_versions.json')
    shutil.copytree(HERE/'assets',destination/'assets')
    s=m['selected']
    initial_text=(f'The initial state comes from the simulator reset with seed {s["seed"]}. '
                  'The JSON init_id is a reset trial offset, not an index into saved benchmark init states.'
                  if s.get('initial_state_source')=='seeded_reset' else f'Saved benchmark init index: {s["init_id"]}.')
    text=f'''# Camera-guided tabletop tidying

PAI final assignment / account ID: {account}

## Method

The robot uses two camera images, robot state and a language instruction.
The participant-trained pi0.5 R07 weights and the original inference class
are reused. Short execution chunks are followed by updated observations.
No saved action trajectory is played back and no new training is performed.

Public task: `{s['suite']}/{s['task']}`.
{initial_text}
The video is the first successful complete clip eligible for 1-3 minutes
at up to 2x playback. This selected demonstration is not a benchmark.

## Reproduce

Use an allocated Linux GPU, a local LIBERO installation and the original
inference dependencies. `environment_versions.json` records the simulator
environment. Model weights and third-party vendor packages are external
assets and are not included. The participant-trained weights have SHA-256:

`{m['weights_sha256']}`

The runtime directory must contain the original model, tokenizer and
processors under `model_weights/pi05_libero_finetuned_v044/`, and `vendor/`
when the inference dependencies were vendored. Supply local paths below:

```bash
python capture.py --runtime "$PAI_POLICY_RUNTIME" --policy-python "$PAI_POLICY_PYTHON" --libero-repo "$LIBERO_REPO" --output ./recording
```

`policy_impl.py` and its helper modules are the inference source used in
the recording. The original MyPolicy AST and preprocessing are preserved;
the HTTP endpoints are unnecessary for this local pipe-based recorder.

To build the presentation on a computer used for video editing:

```bash
python -m pip install -r requirements_presentation.txt
python -m zipfile -e recording/recording_return.zip ./recording_extracted
python presentation.py --input ./recording_extracted --account-id YOUR_ACCOUNT_ID --output ./submission
```

## Sources

- https://github.com/Lifelong-Robot-Learning/LIBERO
- https://github.com/huggingface/lerobot
- https://huggingface.co/lerobot/pi05_libero_finetuned_v044
- https://github.com/Physical-Intelligence/openpi

External models and dependencies retain their upstream terms. This repository
contains the course recorder, presentation builder and participant inference
source. It contains no private evaluation registry, evaluation infrastructure,
model weights, personal paths or participant logs. Generative AI assisted
the implementation and code organization.

The bundled PAI Report Sans is a renamed report-character subset of Noto Sans JP.
Its source, modifications and SIL Open Font License are in `assets/fonts/`.
'''
    (destination/'README.md').write_text(text,encoding='utf-8')
    for path in destination.rglob('*'):
        # Preserve the upstream font license verbatim; its public attribution is not participant data.
        if path.is_file() and path.suffix in ('.py','.md','.json','.txt') and path.name!='OFL.txt':
            if privacy_findings(path.read_text(encoding='utf-8'),forbidden):raise ValueError('Public file needs privacy cleanup: '+path.name)

def build(root,account,output,test_mode=False,forbidden=()):
    if not re.fullmatch('[A-Za-z0-9_-]{1,48}',account) or account in ('YOUR_ACCOUNT_ID','ACCOUNT_ID'):raise ValueError('Use your OmniCampus account ID')
    if privacy_findings(account,forbidden):raise ValueError('Use an account ID without your personal name')
    root=Path(root).resolve();m,ep,capture=verify_capture(root,test_mode)
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Existing output preserved; choose a new --output directory')
    output.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.'+output.name+'.building_',dir=output.parent))
    try:
        filename='PAI最終課題_'+account;video_path=stage/(filename+'.mp4');pdf_path=stage/('説明資料_PAI最終課題_'+account+'.pdf')
        video=make_video(capture,m,ep,account,video_path,test_mode);make_report(capture,m,ep,video,account,pdf_path,test_mode)
        make_public_code(root,stage/'github_public',m,account,forbidden)
        with zipfile.ZipFile(stage/'github_public_code.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted((stage/'github_public').rglob('*')):
                if p.is_file():z.write(p,'github_public/'+p.relative_to(stage/'github_public').as_posix())
        (stage/'SUBMIT.txt').write_text(f'''PAI最終課題 / アカウントID: {account}

1. 動画とPDFを開き、指示・物体・最後の配置が正しく見えることを確認。
2. PDF: {pdf_path.name} をomnicampusへ提出。
3. github_public/ の内容をGitHubのpublicリポジトリへ公開し、そのURLをomnicampusへ提出。
4. {video_path.name} をYouTubeへ限定公開または公開でアップロード。
   タイトル: PAI最終課題_{account}
   実行部分の再生速度: {video['playback_speed']}倍速（動画内にも表示）。
   子ども向け・年齢制限は「いいえ」。採点終了まで削除・非公開化しない。
5. YouTube URLをomnicampusへ提出。

GitHub URL: （公開後に記入）
YouTube URL: （アップロード後に記入）

自分の氏名・所属・メール・個人パスを追加しない。
開発ZIP、private_settings.json、HPCログ、モデル重みは公開用ではない。
コードからの公開・アップロード・提出は自動実行しない。
''',encoding='utf-8')
        write(stage/'SUBMISSION_MANIFEST.json',{'ready_for_manual_submission':not test_mode,'test_only':test_mode,'account_id':account,
            'input_manifest_sha256':sha(root/'recording_manifest.json'),'weights_sha256':m['weights_sha256'],'video':video,'pdf':pdf_path.name,
            'files_sha256':{p.relative_to(stage).as_posix():sha(p) for p in stage.rglob('*') if p.is_file()}})
        stage.rename(output)
    except Exception:shutil.rmtree(stage,ignore_errors=True);raise
    print('PAI_SUBMISSION_READY='+('TEST_ONLY' if test_mode else 'YES'));print('PAI_SUBMISSION_DIR='+str(output));return output

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--account-id',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--privacy-settings',type=Path);a=p.parse_args()
    forbidden=read(a.privacy_settings).get('forbidden_public_tokens',()) if a.privacy_settings else ()
    build(a.input,a.account_id,a.output,forbidden=forbidden)

if __name__=='__main__':main()
