"""Real API/GPU acceptance using only synthetic images; never uses personal files."""
import argparse
import base64
import io
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8790")
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / "validation-output"
root.mkdir(exist_ok=True)


def api(path, body=None, method=None):
    request = urllib.request.Request(args.url + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def encode(image):
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def run(name, payload):
    payload = dict(payload, request_id=f"qa-{name}-{time.time_ns()}")
    started = time.monotonic()
    task = api("/v1/images/edits" if payload.get("images") else "/v1/images/generations", payload)
    task_id = task['id']
    print(f"{name}: submitted {task_id}", flush=True)
    last_print = 0
    while task['status'] not in {'succeeded', 'failed', 'cancelled'}:
        if time.monotonic() - started > 7200:
            raise TimeoutError(f"{name} exceeded 2 hours; task {task_id} remains queryable")
        if time.monotonic() - last_print > 20:
            print(f"{name}: {task['status']} progress={task['progress']} elapsed={time.monotonic()-started:.0f}s", flush=True)
            last_print = time.monotonic()
        time.sleep(2)
        task = api('/v1/tasks/' + task_id)
    outputs = []
    for index, image in enumerate(task.get('images', []), 1):
        path = root / f"{name}-{index}.png"
        with urllib.request.urlopen(args.url + image['url'], timeout=60) as response:
            path.write_bytes(response.read())
        with Image.open(path) as output:
            extrema = output.getchannel('A').getextrema() if 'A' in output.getbands() else None
            outputs.append({'file': path.name, 'size': list(output.size), 'mode': output.mode, 'alpha_range': extrema})
            outputs[-1]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {'name': name, 'task_id': task_id, 'status': task['status'], 'error': task.get('error'),
              'seconds': round(time.monotonic()-started, 2), 'outputs': outputs,
              'health_after': api('/health'),
              'parameters': {k: v for k, v in payload.items() if k not in {'images','mask'}},
              'receipt': task}
    print(f"{name}: {result['status']} {result['seconds']}s {result['error'] or ''}", flush=True)
    return result


source = Image.new('RGB', (512, 512), '#e8e3d9')
draw = ImageDraw.Draw(source)
draw.ellipse((128, 128, 384, 384), fill='#d63630')
second = Image.new('RGB', source.size, '#e8e3d9')
ImageDraw.Draw(second).rectangle((128, 128, 384, 384), fill='#315abc')
mask = Image.new('L', source.size, 0)
ImageDraw.Draw(mask).rectangle((128, 128, 384, 384), fill=255)
base = {'width': 512, 'height': 512, 'steps': 4 if args.quick else 12, 'seed': 42}
cases = [
    ('text-to-image', {'prompt': 'A ceramic red teapot on a light wooden table, soft morning window light, product photography.',
                      'width': 512 if args.quick else 1024, 'height': 512 if args.quick else 1024,
                      'steps': 4 if args.quick else 40, 'seed': 42}),
    ('image-edit', dict(base, prompt='Turn the red circle into a realistic red apple. Keep the light background.', images=[encode(source)])),
    ('multi-reference', dict(base, prompt='Place the red round object from image 1 beside the blue square object from image 2 on a light background.', images=[encode(source),encode(second)])),
    ('transparent', dict(base, prompt='A single cheerful yellow star sticker, clean edges, isolated.', background='transparent')),
    ('mask-edit', dict(base, prompt='Replace the red circle with a shiny green sphere.', images=[encode(source)], mask=encode(mask))),
    ('batch', dict(base, prompt='A small blue ceramic cup on a white background.', n=2)),
    ('seed-a', dict(base, prompt='A small orange fruit on a white table.', steps=4)),
    ('seed-b', dict(base, prompt='A small orange fruit on a white table.', steps=4)),
    ('negative-and-no-cache', dict(base, prompt='A clean blue cup on a white table.', steps=4,
        negative_prompt='blurry, noise', guidance_scale=1.5, use_kv_cache=False)),
    ('ten-references', dict(base, prompt='Arrange the red circles and blue squares in two rows on a light background.',
        steps=2, width=256, height=256, images=[encode(source),encode(second)]*5)),
    ('2k-memory-smoke', dict(base, prompt='A simple blue sphere on a white background.', width=2048,height=2048,steps=1)),
]
report = {'health': api('/health'), 'capabilities': api('/v1/capabilities'), 'quick': args.quick, 'cases': []}
for name, payload in cases:
    result = run(name, payload)
    report['cases'].append(result)
    (root / 'gpu-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if result['status'] != 'succeeded':
        raise SystemExit(1)
    assert len(result['outputs']) == payload.get('n',1), name+' output count'
    assert all(output['size']==[payload['width'],payload['height']] for output in result['outputs']), name+' dimensions'
seed_a=next(case for case in report['cases'] if case['name']=='seed-a')
seed_b=next(case for case in report['cases'] if case['name']=='seed-b')
report['same_seed_identical']=seed_a['outputs'][0]['sha256']==seed_b['outputs'][0]['sha256']
transparent=next(case for case in report['cases'] if case['name']=='transparent')['outputs'][0]
report['transparent_alpha_verified']=bool(transparent['alpha_range'] and transparent['alpha_range'][0]<255 and transparent['alpha_range'][1]>0)
first=report['cases'][0]
report['real_idempotency_verified']=api('/v1/images/generations',first['parameters'])['id']==first['task_id']
assert report['real_idempotency_verified']
mask_result=next(case for case in report['cases'] if case['name']=='mask-edit')
with Image.open(root/mask_result['outputs'][0]['file']) as edited:
    report['mask_preserves_outside']=all(a==b for a,b,m in zip(edited.convert('RGBA').getdata(),source.convert('RGBA').getdata(),mask.getdata()) if m==0)
assert report['mask_preserves_outside']
cancel_task=api('/v1/images/generations',dict(base,prompt='A small green cup.',steps=100))
deadline=time.monotonic()+300
while cancel_task['status'] not in {'succeeded','failed','cancelled'} and not (cancel_task['status']=='running' and cancel_task['step']>=1):
    if time.monotonic()>deadline:raise TimeoutError('Cancellation probe did not start')
    time.sleep(1);cancel_task=api('/v1/tasks/'+cancel_task['id'])
api('/v1/tasks/'+cancel_task['id'],method='DELETE')
while cancel_task['status'] not in {'succeeded','failed','cancelled'}:
    if time.monotonic()>deadline:raise TimeoutError('Cancellation did not settle')
    time.sleep(1);cancel_task=api('/v1/tasks/'+cancel_task['id'])
report['real_cancellation']={'status':cancel_task['status'],'task_id':cancel_task['id'],'images':len(cancel_task['images'])}
assert cancel_task['status']=='cancelled' and not cancel_task['images']
report['after_cancellation']=run('after-cancellation',dict(base,prompt='A small red cup.',steps=2))
assert report['after_cancellation']['status']=='succeeded'
(root/'gpu-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('Real GPU/API suite completed; inspect output images before publishing quality claims.', flush=True)
