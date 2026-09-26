"""A plain HTTP client: no Dreamer, Torch or service-module dependency."""
import argparse
import base64
import json
from pathlib import Path
import time
import urllib.request
import uuid

parser = argparse.ArgumentParser()
parser.add_argument('prompt')
parser.add_argument('--url', default='http://127.0.0.1:8790')
parser.add_argument('--image', type=Path, action='append', default=[])
parser.add_argument('--mask', type=Path)
parser.add_argument('--transparent', action='store_true')
parser.add_argument('--width', type=int, default=1024)
parser.add_argument('--height', type=int, default=1024)
parser.add_argument('--steps', type=int, default=40)
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--output', type=Path, default=Path('generated.png'))
args = parser.parse_args()


def api(path, payload=None):
    request = urllib.request.Request(args.url.rstrip('/') + path,
        data=json.dumps(payload).encode() if payload else None,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


payload = dict(prompt=args.prompt, width=args.width, height=args.height,
               steps=args.steps, seed=args.seed, request_id=uuid.uuid4().hex,
               background='transparent' if args.transparent else 'default',
               images=[base64.b64encode(path.read_bytes()).decode() for path in args.image])
if args.mask:
    payload['mask'] = base64.b64encode(args.mask.read_bytes()).decode()
task = api('/v1/images/edits' if args.image else '/v1/images/generations', payload)
print('Task ID:', task['id'], flush=True)
while task['status'] not in {'succeeded', 'failed', 'cancelled'}:
    print(task['status'], f"{task['progress']:.0%}", flush=True)
    time.sleep(3)
    task = api('/v1/tasks/' + task['id'])
if task['status'] != 'succeeded':
    raise SystemExit(task.get('error') or task['status'])
with urllib.request.urlopen(args.url.rstrip('/') + task['images'][0]['url'], timeout=60) as response:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(response.read())
print(args.output.resolve())
