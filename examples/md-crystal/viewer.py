"""Local UI bridge. All simulation steps run in the lowered native Host binary."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading


def serve(command, env, port):
    process = subprocess.Popen([*command, '--interactive'], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, text=True, env=env, bufsize=1)
    lock = threading.Lock()
    initial = process.stdout.readline()
    if not initial:
        raise RuntimeError('native simulation failed to initialize')
    state = json.loads(initial)
    html = Path(__file__).with_name('viewer.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, content, mime):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.path == '/':
                self.reply(200, html, 'text/html; charset=utf-8')
            elif self.path == '/state':
                with lock:
                    data = json.dumps(state).encode()
                self.reply(200, data, 'application/json')
            else:
                self.reply(404, b'Not found', 'text/plain')

        def do_POST(self):
            nonlocal state
            try:
                if self.path != '/step':
                    raise ValueError('unknown endpoint')
                # Browser requests must originate from this local viewer.
                origin = self.headers.get('Origin')
                if origin and origin != f'http://127.0.0.1:{port}':
                    raise ValueError('invalid origin')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1024:
                    raise ValueError('invalid request size')
                request = json.loads(self.rfile.read(length))
                if request.get('reset'):
                    line = 'reset\n'
                else:
                    count = int(request.get('steps', 8))
                    hands = request.get('hands', [[-100, -100, -100], [-100, -100, -100]])
                    values = [float(v) for hand in hands for v in hand]
                    amplitude = float(request.get('amplitude', 0.03))
                    import math
                    if len(values) != 6 or not all(math.isfinite(v) for v in values) or not 0 <= count <= 100 or not 0 <= amplitude <= 0.15:
                        raise ValueError('invalid simulation parameters')
                    line = 'step ' + ' '.join(map(str, [count, *values, amplitude])) + '\n'
                with lock:
                    process.stdin.write(line)
                    process.stdin.flush()
                    result = process.stdout.readline()
                    if not result:
                        raise RuntimeError('native process stopped; see terminal')
                    state = json.loads(result)
                    data = json.dumps(state).encode()
                self.reply(200, data, 'application/json')
            except (ValueError, TypeError, RuntimeError, BrokenPipeError) as error:
                self.reply(400, json.dumps(dict(error=str(error))).encode(), 'application/json')

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'結晶を触って加熱: http://127.0.0.1:{port}/ (Ctrl+Cで終了)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        process.terminate()
        process.wait(timeout=10)
