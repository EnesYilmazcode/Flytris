"""Render the arcade scene with headless Chromium.

  python capture.py stills --n 1500 --times 0 5 9 14 20 27 31 36
  python capture.py video --n 7172 --out ../media/flytris_arcade_3d.mp4

Serves this folder on localhost, drives window.flytris frame by frame, and pipes
PNG frames into ffmpeg. Rendering is deterministic: frame i is time i / fps.
"""
import argparse
import base64
import functools
import http.server
import subprocess
import sys
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
EXTRA = ""
ARGS = ["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist", "--force_high_performance_gpu"]


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve():
    handler = functools.partial(Quiet, directory=str(HERE))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def open_page(p, port, n, w, h):
    browser = p.chromium.launch(headless=True, args=ARGS)
    page = browser.new_page(viewport={"width": w, "height": h})
    page.on("console", lambda m: m.type in ("error", "warning") and print("[page]", m.text[:400], file=sys.stderr))
    page.goto(f"http://127.0.0.1:{port}/index.html?mode=capture&n={n}&w={w}&h={h}{EXTRA}")
    page.wait_for_function("window.flytris?.ready || window.flytrisError", timeout=180_000)
    err = page.evaluate("window.flytrisError")
    if err:
        raise RuntimeError(err)
    gpu = page.evaluate("""() => { const gl = document.querySelector('canvas').getContext('webgl2');
        const e = gl.getExtension('WEBGL_debug_renderer_info'); return e ? gl.getParameter(e.UNMASKED_RENDERER_WEBGL) : '?'; }""")
    print("GPU:", gpu, "| info:", page.evaluate("window.flytris.info()"))
    return browser, page


def decode(url):
    return base64.b64decode(url.split(",", 1)[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["stills", "video", "views", "timeline"])
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--w", type=int, default=1080)
    ap.add_argument("--h", type=int, default=1350)
    ap.add_argument("--times", type=float, nargs="*", default=[0, 4, 8, 11, 15, 19, 24, 29, 32, 35, 37.5])
    ap.add_argument("--outdir", type=Path, default=HERE / "stills")
    ap.add_argument("--out", type=Path, default=HERE.parent / "media" / "flytris_arcade_3d.mp4")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--end", type=float, default=0)
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    global EXTRA
    EXTRA = a.extra

    server = serve()
    with sync_playwright() as p:
        browser, page = open_page(p, server.server_address[1], a.n, a.w, a.h)
        if a.what == "timeline":
            import json
            (HERE / "timeline.json").write_text(json.dumps(page.evaluate("window.flytris.timeline()")))
            print("wrote timeline.json")
        elif a.what == "stills":
            a.outdir.mkdir(exist_ok=True)
            for t in a.times:
                t0 = time.time()
                page.evaluate(f"window.flytris.renderAt({t})")
                url = page.evaluate("document.querySelector('canvas').toDataURL('image/png')")
                path = a.outdir / f"t{t:05.1f}.png"
                path.write_bytes(decode(url))
                print(f"{path.name} {time.time() - t0:.2f}s")
        elif a.what == "views":
            # Debug orbit around the hero fly at t=0.5.
            a.outdir.mkdir(exist_ok=True)
            hx, hz, _, _ = page.evaluate("window.flytris.heroPos()")
            import math
            for k, (az, el, d) in enumerate([(100, 5, 1.6), (60, 12, 1.7), (25, 25, 1.8), (150, 8, 1.4), (90, 55, 1.8), (120, 2, 1.0)]):
                look = [hx, 1.2, hz + 1.1]
                pos = [look[0] + d * math.sin(math.radians(az)) * math.cos(math.radians(el)), look[1] + d * math.sin(math.radians(el)),
                       look[2] + d * math.cos(math.radians(az)) * math.cos(math.radians(el))]
                page.evaluate(f"window.flytris.renderView(0.5, {pos}, {look})")
                (a.outdir / f"t{k:05.1f}.png").write_bytes(decode(page.evaluate("document.querySelector('canvas').toDataURL('image/png')")))
        else:
            fps = page.evaluate("window.flytris.fps")
            end = a.end or page.evaluate("window.flytris.duration")
            # An explicit --end is exclusive, so chunks rendered back to back never repeat a frame.
            first, last = round(a.start * fps), round(end * fps) - (1 if a.end else 0)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            tmp = a.out.with_suffix(".partial.mp4")
            ff = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(fps), "-c:v", "png",
                 "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", str(tmp)],
                stdin=subprocess.PIPE,
            )
            t0 = time.time()
            for i in range(first, last + 1):
                ff.stdin.write(decode(page.evaluate(f"window.flytris.renderFrame({i})")))
                if i % 30 == 0:
                    rate = (i - first + 1) / (time.time() - t0)
                    print(f"frame {i}/{last}  {rate:.1f} fps", flush=True)
            ff.stdin.close()
            if ff.wait() != 0:
                raise RuntimeError("ffmpeg failed")
            tmp.replace(a.out)
            print("wrote", a.out)
        browser.close()
    server.shutdown()


if __name__ == "__main__":
    main()
