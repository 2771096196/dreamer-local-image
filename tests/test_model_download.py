import hashlib
import http.server
from pathlib import Path
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_download import download_file, download_segmented, verify


@pytest.mark.parametrize('honor_range', [True, False])
def test_resume_and_ignore_range_servers(tmp_path, honor_range):
    payload = b'official-model-test-bytes' * 10000
    seen = []
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requested = self.headers.get('Range')
            seen.append(requested)
            offset = int(requested.split('=')[1].split('-')[0]) if requested and honor_range else 0
            self.send_response(206 if requested and honor_range else 200)
            if requested and honor_range:
                self.send_header('Content-Range', f'bytes {offset}-{len(payload)-1}/{len(payload)}')
            self.send_header('Content-Length', str(len(payload)-offset))
            self.end_headers()
            self.wfile.write(payload[offset:])
        def log_message(self, *_):
            pass
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    destination = tmp_path / 'weight.safetensors'
    partial = destination.with_name(destination.name + '.part')
    partial.write_bytes(payload[:123])
    entry = {'path': destination.name, 'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
    try:
        download_file(f'http://127.0.0.1:{server.server_port}/weight', destination, entry, lambda _: None)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert seen == ['bytes=123-']
    assert destination.read_bytes() == payload
    assert not partial.exists()
    assert verify(destination, entry)


def test_non_lfs_git_blob_hash(tmp_path):
    content = b'{"model": "test"}\n'
    path = tmp_path / 'config.json'
    path.write_bytes(content)
    entry = {'size': len(content), 'git_sha1': hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()}
    assert verify(path, entry)
    path.write_bytes(b'x' * len(content))
    assert not verify(path, entry)


def test_segmented_download_reuses_prefix_and_ranges(tmp_path):
    payload = bytes(range(256)) * 1000
    seen = []
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            start, end = map(int, self.headers['Range'].split('=')[1].split('-'))
            seen.append((start,end))
            self.send_response(206)
            self.send_header('Content-Range',f'bytes {start}-{end}/{len(payload)}')
            self.send_header('Content-Length',str(end-start+1))
            self.end_headers()
            self.wfile.write(payload[start:end+1])
        def log_message(self, *_):
            pass
    server = http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread = threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    path=tmp_path/'weights'
    path.with_name('weights.part').write_bytes(payload[:12345])
    chunks=path.with_name('weights.chunks');chunks.mkdir()
    (chunks/'16384').write_bytes(payload[16384:20000])
    entry={'path':'weights','size':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
    try:
        download_segmented(f'http://127.0.0.1:{server.server_port}/weights',path,entry,lambda _:None,workers=4,chunk_size=16384)
    finally:
        server.shutdown();server.server_close();thread.join()
    assert path.read_bytes()==payload
    assert (20000,32767) in seen
    assert not list(chunks.iterdir())
