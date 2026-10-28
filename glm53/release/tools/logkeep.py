#!/usr/bin/env python3
"""logkeep.py: the logs of this release, in memory, one keeper a node and lane. tf-glm53 starts it; nothing goes to disk.

  logkeep.py --sock NAME serve [--listen HOST:PORT] [--keep MB]   the keeper (stays on): streams in memory
  logkeep.py --sock NAME start [--listen HOST:PORT] [--keep MB] [--renew]
                                          the keeper in the background unless one of this version answers (another
                                          version's is replaced); --renew: one with other settings is replaced too
                                          (the replaced keeper's lines are lost)
  logkeep.py --sock NAME send STREAM      stdin into STREAM (a writer opening STREAM moves its lines to STREAM.prev)
  logkeep.py --sock NAME open STREAM...   each STREAM's lines to STREAM.prev now (before a start: no line of the start
                                          before is read as the new one's; its writer then finds STREAM empty)
  logkeep.py --sock NAME cat STREAM       STREAM's lines held now
  logkeep.py --sock NAME follow STREAM    STREAM's lines, then each new one (Ctrl-C ends it)
  logkeep.py --sock NAME ping             exit 0 when the keeper answers; its settings to stdout
  logkeep.py --sock NAME stop             the keeper ends (its lines are lost)

NAME is an abstract unix socket (no file: nothing for logind to remove at logout, nothing on disk). An abstract socket
has no file permissions, so both ends check the other's uid: the keeper serves the user it runs as only, and the
commands talk to their own user's keeper only (another user's under the same name: refused). Each stream keeps
its last MB (--keep, 8 by default); older lines go. With --listen the keeper also answers, read-only and with no key,
GET /streams (the node, the keeper's start time, each stream's first and last line numbers and size) and
GET /logs?stream=S&after=N&max=M (the lines after line N, as [number, unix time, text]), for a monitor that keeps the
logs elsewhere. Line numbers run across the streams and start again at 1 with a new keeper (its start time changes). `send` never stops
reading its input while the keeper is away (lines meanwhile are dropped), so the writer is never blocked by it."""
import collections, json, os, socket, socketserver, struct, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

MAX_LINE = 16384  # a longer line is cut
VERSION = 2       # of the protocol: start replaces a keeper of another (2: open, started)


def settings(listen, keep_mb):
    return f"v{VERSION} listen={listen or '-'} keep={keep_mb:g}"


class Stream:
    def __init__(self, keep):
        self.lines, self.bytes, self.keep = collections.deque(), 0, keep

    def add(self, seq, text):
        self.lines.append((seq, time.time(), text))
        self.bytes += len(text) + 32
        while self.bytes > self.keep and len(self.lines) > 1:
            _, _, old = self.lines.popleft()
            self.bytes -= len(old) + 32


class Keeper:
    def __init__(self, keep):
        self.keep, self.lock, self.seq, self.streams, self.started = keep, threading.Lock(), 0, {}, round(time.time(), 3)
        self.changed = threading.Condition(self.lock)

    def open(self, name):
        with self.lock:
            if name in self.streams and self.streams[name].lines:
                self.streams[name + ".prev"] = self.streams[name]
            self.streams[name] = Stream(self.keep)

    def add(self, name, text):
        with self.lock:
            self.seq += 1
            self.streams.setdefault(name, Stream(self.keep)).add(self.seq, text[:MAX_LINE])
            self.changed.notify_all()

    def after(self, name, after=0, limit=None):
        with self.lock:
            s = self.streams.get(name)
            out = [x for x in s.lines if x[0] > after] if s else []
        return out[:limit] if limit else out

    def summary(self):
        with self.lock:
            return [{"stream": k, "first": v.lines[0][0] if v.lines else 0, "last": v.lines[-1][0] if v.lines else 0,
                     "lines": len(v.lines), "bytes": v.bytes} for k, v in sorted(self.streams.items())]


def handler(keeper, me):
    class H(socketserver.StreamRequestHandler):
        def handle(self):
            if peer_uid(self.request) != os.getuid():     # another user: nothing (no line in, none out)
                return
            head = self.rfile.readline().decode("utf-8", "replace").split()
            if not head:
                return
            op, name = head[0], head[1] if len(head) > 1 else ""
            if op in ("W", "A"):                           # a writer (A: the same one again, appending): its lines
                if op == "W":
                    keeper.open(name)
                for raw in self.rfile:
                    keeper.add(name, raw.decode("utf-8", "replace").rstrip("\n"))
            elif op in ("R", "F"):                         # a reader: the lines held, and with F each new one
                last = 0
                while True:
                    for seq, _, text in keeper.after(name, last):
                        self.wfile.write((text + "\n").encode())
                        last = seq
                    self.wfile.flush()
                    if op == "R":
                        return
                    with keeper.changed:
                        keeper.changed.wait(1.0)
            elif op == "O":                                # streams opened anew before their writers come
                for n in head[1:]:
                    keeper.open(n)
                self.wfile.write(b"ok\n")
            elif op == "P":
                self.wfile.write(f"ok {me}\n".encode())
            elif op == "Q":
                self.wfile.write(b"bye\n"); self.wfile.flush()
                os._exit(0)
    return H


class UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def handle_error(self, request, address):            # a reader gone mid-line: nothing to say (stderr is closed)
        pass


def http_handler(keeper):
    host = socket.gethostname()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(u.query).items()}
            if u.path == "/streams":
                body = {"node": host, "started": keeper.started, "streams": keeper.summary()}
            elif u.path == "/logs" and q.get("stream"):
                try:
                    after, limit = int(q.get("after", 0)), min(int(q.get("max", 5000)), 50000)
                except ValueError:
                    return self.reply(400, {"error": "after and max are numbers"})
                lines = keeper.after(q["stream"], after, limit)
                body = {"node": host, "started": keeper.started, "stream": q["stream"], "lines": [[s, round(t, 3), x] for s, t, x in lines],
                        "last": lines[-1][0] if lines else after}
            else:
                return self.reply(404, {"error": "GET /streams or /logs?stream=S&after=N&max=M"})
            self.reply(200, body)

        def reply(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    return H


def peer_uid(s):
    return struct.unpack("3i", s.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[1]


def connect(sock, first):
    """A connection to this user's keeper; PermissionError when the name is another user's keeper."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect("\0" + sock)
    if peer_uid(s) != os.getuid():
        s.close()
        raise PermissionError(f"{sock}: another user's keeper")
    s.sendall((first + "\n").encode())
    return s


def serve(sock, listen, keep_mb):
    """Both sockets are bound before either answers, so a keeper that answers ping has its HTTP too."""
    keeper = Keeper(int(keep_mb * 1024 * 1024))
    try:
        unix = UnixServer("\0" + sock, handler(keeper, settings(listen, keep_mb)))
    except OSError:
        print(f"logkeep: {sock}: a keeper answers already", file=sys.stderr)
        return 1
    http = None
    if listen:
        host, port = listen.rsplit(":", 1)
        try:
            http = ThreadingHTTPServer((host.strip("[]"), int(port)), http_handler(keeper))
        except (OSError, ValueError) as e:
            print(f"logkeep: --listen {listen}: {e}", file=sys.stderr)
            return 1
        http.daemon_threads = True
        threading.Thread(target=http.serve_forever, daemon=True).start()
    threading.Thread(target=unix.serve_forever, daemon=True).start()
    print(f"logkeep: {sock} on {socket.gethostname()}{', read-only at http://' + listen if listen else ''}", flush=True)
    null = os.open(os.devnull, os.O_WRONLY)              # started in the background, nobody reads these any more
    os.dup2(null, 1); os.dup2(null, 2)
    while True:
        time.sleep(3600)


def ask(sock, op):
    """The keeper's one-line answer to op, "" from another user's keeper (it closes at once), None when none answers."""
    try:
        s = connect(sock, op)
        s.settimeout(5)
        return s.makefile("rb").readline().decode().strip()
    except (PermissionError, ConnectionResetError):
        return ""
    except OSError:
        return None


def start(sock, listen, keep_mb, renew):
    have = ask(sock, "P")
    if have == "":                                     # a keeper of another user holds the name
        print(f"logkeep: {sock}: held by another user's keeper", file=sys.stderr)
        return 1
    if have is not None:
        same_version = have.split()[1:2] == [f"v{VERSION}"]
        if have == "ok " + settings(listen, keep_mb) or (same_version and not renew):
            return 0
        ask(sock, "Q")
        for _ in range(50):
            if ask(sock, "P") is None:
                break
            time.sleep(0.1)
    argv = [sys.executable, os.path.abspath(__file__), "--sock", sock, "serve", "--keep", f"{keep_mb:g}"]
    p = subprocess.Popen(argv + (["--listen", listen] if listen else []), stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
    for _ in range(100):
        if p.poll() is not None:
            print((p.stderr.read().decode().strip() or f"logkeep: the keeper ended ({p.returncode})"), file=sys.stderr)
            return 1
        if ask(sock, "P") is not None:
            return 0
        time.sleep(0.1)
    print(f"logkeep: {sock}: the keeper does not answer", file=sys.stderr)
    return 1


def send(sock, stream):
    """stdin to the keeper; when it is away, the input is still read (and dropped) so the writer never blocks, and the
    keeper is asked again every few seconds (a new connection opens the stream anew only the first time)."""
    s, opened, retry = None, False, 0.0
    src = sys.stdin.buffer
    for raw in src:
        if s is None and time.time() >= retry:
            try:
                s = connect(sock, f"W {stream}" if not opened else f"A {stream}")
                opened = True
            except OSError:
                s, retry = None, time.time() + 5
        if s is not None:
            try:
                s.sendall(raw if raw.endswith(b"\n") else raw + b"\n")
            except OSError:
                s, retry = None, time.time() + 5
    return 0


def read(sock, stream, follow):
    try:
        s = connect(sock, f"{'F' if follow else 'R'} {stream}")
    except PermissionError:
        print(f"logkeep: {sock}: refused (another user's keeper)", file=sys.stderr)
        return 1
    except OSError:
        print(f"logkeep: no keeper {sock} on this node", file=sys.stderr)
        return 1
    try:
        while True:
            data = s.recv(65536)
            if not data:
                return 0
            sys.stdout.buffer.write(data)
            sys.stdout.flush()
    except KeyboardInterrupt:
        return 0
    except BrokenPipeError:                                # a reader that stopped early (grep -q, head)
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except ConnectionResetError:
        print(f"logkeep: {sock}: refused (another user's keeper)", file=sys.stderr)
        return 1


def main(argv):
    if len(argv) < 3 or argv[0] != "--sock":
        print(__doc__.split("\n\n")[0], file=sys.stderr)
        return 2
    sock, cmd, rest = argv[1], argv[2], argv[3:]
    if cmd in ("serve", "start"):
        listen, keep, renew = None, 8.0, False
        while rest:
            k = rest.pop(0)
            if k == "--listen" and rest:
                listen = rest.pop(0) or None
            elif k == "--keep" and rest:
                keep = float(rest.pop(0))
            elif k == "--renew" and cmd == "start":
                renew = True
            else:
                print(f"logkeep {cmd}: unknown {k}", file=sys.stderr)
                return 2
        return serve(sock, listen, keep) if cmd == "serve" else start(sock, listen, keep, renew)
    if cmd == "send" and len(rest) == 1:
        return send(sock, rest[0])
    if cmd in ("cat", "follow") and len(rest) == 1:
        return read(sock, rest[0], cmd == "follow")
    if cmd == "open" and rest:
        a = ask(sock, "O " + " ".join(rest))
        if a != "ok":
            print(f"logkeep: {sock}: no keeper" if a is None else f"logkeep: {sock}: refused (another user's keeper)", file=sys.stderr)
        return 0 if a == "ok" else 1
    if cmd in ("ping", "stop"):
        a = ask(sock, "P" if cmd == "ping" else "Q")
        if a and cmd == "ping":
            print(a[3:])
        return 0 if a and a.split()[0] in ("ok", "bye") else 1
    print("logkeep: serve | start | send STREAM | open STREAM... | cat STREAM | follow STREAM | ping | stop", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
