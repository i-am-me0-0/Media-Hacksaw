#!/usr/bin/env python3
"""
Media Hacksaw
Requirements: pip install flask  +  ffmpeg/ffprobe in PATH
Optional:     pip install smartcut
Usage:        python server.py  ->  http://localhost:5000
"""

import os, re, json, subprocess, tempfile, base64, struct, shutil, time, threading, atexit
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, jsonify, request, send_file, Response

app = Flask(__name__, static_folder=".", template_folder=".")
HERE           = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FOLDER = os.path.join(HERE, "videos")
SKIP_DIRS      = {"cuts", "done", "skip"}

# ── Detection tuning ──────────────────────────────────────────────────────────
BLACK_HIGH   = 24            # a frame is "black" if its 95th-percentile luma <= this (16 = pure black)
DARK_HIGH    = 70            # silence + 95th-percentile luma below this => "silence_dark" candidate
WINDOW       = (0.30, 0.70)  # only suggest cuts in this fraction of the runtime (two-episode files)
PAD          = 0.15          # tolerance (s) between a black frame and its silence
MAX_PREVIEWS = int(os.environ.get("MKVCUT_MAX_PREVIEWS", 5))

_cache = {}
_vstats_cache = {}   # per-file video stats: changing the silence threshold only re-runs the fast audio pass
_NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# ─── Utilities ────────────────────────────────────────────────────────────────

def run(cmd, timeout=600, binary=False):
    kw: dict = dict(capture_output=True, timeout=timeout, creationflags=_NOWIN)   # annotated so Pylance does not infer dict[str, int]
    if not binary:
        kw.update(text=True, encoding="utf-8", errors="replace")
    return subprocess.run(cmd, **kw)


def fmt_size(n):
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024: return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.0f} PB"


_probe_cache = {}
def probe(path):
    key = (path, os.path.getmtime(path))
    if key in _probe_cache: return _probe_cache[key]
    r = run(["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", path], 30)
    j = json.loads(r.stdout)
    v = next((s for s in j.get("streams", []) if s.get("codec_type") == "video"), {})
    try:
        n, _, d = (v.get("avg_frame_rate") or "0/1").partition("/")
        fps = float(n) / float(d or 1)
    except Exception:
        fps = 0
    info = dict(duration=float(j["format"]["duration"]),
                vcodec=v.get("codec_name", ""), pix_fmt=v.get("pix_fmt", ""),
                width=int(v.get("width", 0)), height=int(v.get("height", 0)),
                fps=fps or 24.0)
    _probe_cache[key] = info
    return info


def get_duration(path):
    return probe(path)["duration"]


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


# ─── Detection ────────────────────────────────────────────────────────────────

def detect_crop(path, info):
    """
    Find the real picture area (4:3 content in a 16:9 frame has black side bars).
    The old blackdetect counted those bars as black pixels, so dark-ish frames
    passed as 'black'. Returns (w,h,x,y) or None if there is nothing to crop.
    """
    dur, w, h = info["duration"], info["width"], info["height"]
    best = None
    for frac in (0.2, 0.4, 0.6, 0.8):
        r = run(["ffmpeg", "-hide_banner", "-ss", f"{dur*frac:.2f}", "-i", path, "-t", "2",
                 "-an", "-sn", "-vf", "cropdetect=limit=24:round=2:reset=0",
                 "-f", "null", "-"], 60)
        m = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", r.stderr)
        if m:
            cw, ch, x, y = map(int, m[-1])
            if best is None or cw * ch > best[0] * best[1]:
                best = (cw, ch, x, y)   # widest box wins: dark scenes only ever shrink it
    if not best or best[0] * best[1] >= 0.97 * w * h:
        return None
    return best


def video_stats(path, crop):
    """
    One pass, per-frame luma stats (YHIGH = 95th percentile, YAVG) of the picture
    area only. Gives frame-exact black runs and lets us measure 'how dark' any
    point is, instead of relying on blackdetect's all-or-nothing answer.
    Scaled down only because signalstats is much heavier than blackdetect.
    """
    vf = (f"crop={crop[0]}:{crop[1]}:{crop[2]}:{crop[3]}," if crop else "") + \
         "scale=320:-2:flags=fast_bilinear,signalstats," \
         "metadata=mode=print:key=lavfi.signalstats.YHIGH:file=-"   # ONE writer: two filters on stdout interleave mid-line on Windows
    r = run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-an", "-sn",
             "-vf", vf, "-f", "null", "-"], 1800)
    highs, cur = {}, None
    for line in r.stdout.splitlines():
        m = re.search(r"pts_time:(-?\d+(?:\.\d+)?)", line)
        if m:
            cur = float(m.group(1)); continue
        m = re.search(r"YHIGH=(\d+(?:\.\d+)?)", line)
        if m and cur is not None:
            highs[cur] = float(m.group(1))
    return [(t, highs[t], highs[t]) for t in sorted(highs)]   # (time, luma95, luma95)


def frame_runs(frames, pred, fps):
    out, start, last, fd = [], None, None, 1.0 / fps
    for t, hi, av in frames:
        if pred(hi, av):
            if start is None: start = t
            last = t
        elif start is not None:
            out.append((start, last + fd)); start = None
    if start is not None: out.append((start, last + fd))
    return out


def audio_silences(path, noise_db, min_dur, dur):
    """
    Bug fix: ffmpeg prints 'silence_start: 12.3' (with a space). The old regex
    had no \\s*, so it never matched and the result was always 0 silences.
    """
    r = run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vn", "-sn",
             "-af", f"aformat=channel_layouts=mono,silencedetect=noise={noise_db}dB:d={max(min_dur, 0.05)}",
             "-f", "null", "-"], 900)
    s = [max(0.0, float(x)) for x in re.findall(r"silence_start:\s*(-?[\d.]+)", r.stderr)]
    e = [float(x) for x in re.findall(r"silence_end:\s*(-?[\d.]+)", r.stderr)]
    if len(s) > len(e): e.append(dur)
    return list(zip(s, e))


BASE    = {"confirmed": 100, "silence_dark": 70, "black_only": 40, "silence_only": 15}
PAIR    = (6.0, 10.0)   # s: boundary gap -> ~8 s title card -> second gap. The FIRST gap of such a pair is the cut.
CLUSTER = 20.0   # s: a black+silent gap this soon after another one is usually "title card -> content", not the boundary

def score(t, dur):
    """Type first, then closeness to the middle. No bonus for LONGER blacks: the boundary before a
    title card is short, the black after the card is long, and the earlier one is the right cut."""
    pos = t["midpoint"] / dur
    if not (WINDOW[0] <= pos <= WINDOW[1]): return 0
    return round(BASE[t["type"]] + 20 * max(0.0, 1 - abs(pos - 0.5) / 0.2) + (5 if t["dur"] >= 0.2 else 0), 1)


def detect_transitions(path, noise_db: float = -35, min_dur: float = 0.04):
    """
    Candidate types (best first):
      confirmed    fully black picture AND silent audio
      silence_dark silent audio at a dark (not perfectly black) frame
      black_only   fully black but audio is playing
      silence_only silence in the middle of the file, picture not dark
    Cut point (`midpoint`) = middle of the black/silence overlap, so it lands
    where BOTH are true. The best candidate in the middle of the file is flagged
    `suggested`.
    """
    info = probe(path)
    dur, fps = info["duration"], info["fps"]
    vkey = (path, os.path.getmtime(path))
    def _video():
        if vkey not in _vstats_cache:
            c = detect_crop(path, info)
            _vstats_cache[vkey] = (c, video_stats(path, c))
        return _vstats_cache[vkey]

    with ThreadPoolExecutor(2) as ex:
        fv = ex.submit(_video)
        fa = ex.submit(audio_silences, path, noise_db, min_dur, dur)
        (crop, frames), silences = fv.result(), fa.result()

    min_run = min(min_dur, 1.0 / fps - 0.001)
    blacks = [(s, e) for s, e in frame_runs(frames, lambda hi, av: hi <= BLACK_HIGH, fps)
              if e - s >= min_run]

    T = []
    for bs, be in blacks:
        bd = be - bs
        best = max(silences, key=lambda s: overlap(bs - PAD, be + PAD, s[0], s[1]), default=None)
        ov = overlap(bs - PAD, be + PAD, best[0], best[1]) if best else 0
        silent = ov >= 0.5 * bd
        cs, ce = (max(bs, best[0]), min(be, best[1])) if best else (bs, be)
        cut = (cs + ce) / 2 if (silent and ce > cs) else (bs + be) / 2
        T.append(dict(start=round(bs, 3), end=round(be, 3), midpoint=round(cut, 3),
                      dur=round(bd, 3), type="confirmed" if silent else "black_only", rms_db=None))

    # Silence alone is a weak signal (cartoons have hundreds of pauses), so it only adds candidates when it
    # helps: silence on a DARK frame is always listed (capped); plain silence only if no black gap exists at all.
    sil = []
    for ss, se in silences:
        if se - ss < 0.3: continue
        mid = (ss + se) / 2
        if not (WINDOW[0] <= mid / dur <= WINDOW[1]): continue
        if any(overlap(ss - PAD, se + PAD, bs, be) > 0 for bs, be in blacks): continue
        inside = [(t, av) for t, hi, av in frames if ss <= t <= se]
        if not inside: continue
        lo = min(av for _, av in inside)
        cut = min((t for t, av in inside if av <= lo + 3), key=lambda t: abs(t - mid))
        sil.append(dict(start=round(ss, 3), end=round(se, 3), midpoint=round(cut, 3), dur=round(se - ss, 3),
                        type="silence_dark" if lo <= DARK_HIGH else "silence_only", rms_db=None))
    in_win = lambda t: WINDOW[0] <= t["midpoint"] / dur <= WINDOW[1]
    longest = lambda L, n: sorted(L, key=lambda x: -x["dur"])[:n]
    T += longest([x for x in sil if x["type"] == "silence_dark"], 8)
    if not any(in_win(t) for t in T):
        T += longest([x for x in sil if x["type"] == "silence_only"], 8)

    for t in T: t["score"] = score(t, dur); t["suggested"] = False
    confirmed = [t for t in T if t["type"] == "confirmed"]
    for t in confirmed:   # title-card pattern: this gap is followed 6-10 s later by another black+silent gap
        if t["score"] > 0 and any(PAIR[0] <= p["midpoint"] - t["midpoint"] <= PAIR[1] for p in confirmed):
            t["score"] = round(t["score"] + 30, 1)
    strong = [t for t in confirmed if t["dur"] >= 0.1]
    for t in T:   # later gap soon after an earlier black+silent gap -> prefer the earlier one
        if t["score"] > 0 and any(0 < t["midpoint"] - p["midpoint"] <= CLUSTER for p in strong):
            t["score"] = round(t["score"] - 25, 1)
    T.sort(key=lambda t: t["start"])
    top = max(T, key=lambda t: t["score"], default=None)
    if top and top["score"] > 0: top["suggested"] = True

    return {
        "duration": dur, "noise_db": noise_db, "fps": fps, "vcodec": info["vcodec"],
        "crop": list(crop) if crop else None,
        "blacks": [{"s": round(s, 3), "e": round(e, 3)} for s, e in blacks],
        "silences": [{"s": round(s, 3), "e": round(e, 3)} for s, e in silences],
        "transitions": T,
    }


# ─── Playback clips (ONLY for files the browser cannot play directly) ─────────
# Normal case: the original file is streamed as-is (no conversion, no writes).
# Fallback (e.g. x265 in Firefox): a 90 s, 480p H.264 clip around the cut point,
# aligned to a 30 s grid so nearby cut points reuse the same clip. ~3-5 MB each.
# Written to a temp folder (MKVCUT_CLIP_DIR can point to a RAM disk), removed on exit.
CLIP_GRID, CLIP_LEN, MAX_CLIPS = 30, 90, 6
CLIP_DIR = tempfile.mkdtemp(prefix="mkvcut_", dir=os.environ.get("MKVCUT_CLIP_DIR") or None)
atexit.register(shutil.rmtree, CLIP_DIR, True)
_clips, _clip_locks, _clip_guard = {}, {}, threading.Lock()


def get_clip(path, start):
    key = (path, os.path.getmtime(path), start)
    with _clip_guard:
        lock = _clip_locks.setdefault(key, threading.Lock())
    with lock:
        out = _clips.get(key)
        if out and os.path.isfile(out): return out
        out = os.path.join(CLIP_DIR, f"{abs(hash(key))}.mp4")
        tmp = out + ".part.mp4"
        r = run(["ffmpeg", "-y", "-hide_banner", "-hwaccel", "auto", "-ss", str(start), "-i", path,
                 "-t", str(CLIP_LEN), "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn",
                 "-vf", "scale=-2:480:flags=fast_bilinear", "-c:v", "libx264", "-preset", "ultrafast",
                 "-crf", "30", "-g", "12", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
                 "-movflags", "+faststart", tmp], 300, binary=True)
        if r.returncode != 0 or not os.path.isfile(tmp):
            try: os.unlink(tmp)
            except OSError: pass
            return None
        os.replace(tmp, out)
        _clips[key] = out
        while len(_clips) > MAX_CLIPS:
            try: os.unlink(_clips.pop(next(iter(_clips))))
            except OSError: pass
        return out


def safe_move(src, dst):
    """Retry for Windows, where the browser may still hold a read handle."""
    for attempt in range(6):
        try:
            shutil.move(src, dst); return
        except (PermissionError, OSError):
            if attempt < 5: time.sleep(0.3 * (attempt + 1))
            else: raise


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/api/config")
def config():
    return jsonify(default_folder=DEFAULT_FOLDER)


@app.route("/")
def index():
    with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.route("/api/scan", methods=["POST"])
def scan():
    folder = (request.json or {}).get("folder", "").strip()
    if not os.path.isdir(folder):
        return jsonify(error=f"Not a directory: {folder}"), 400
    files = []
    for root, dirs, names in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(names):
            if name.lower().endswith(".mkv"):
                p = os.path.join(root, name)
                files.append({"path": p, "name": name, "rel": os.path.relpath(p, folder),
                              "size": fmt_size(os.path.getsize(p))})
    files.sort(key=lambda x: x["rel"])
    return jsonify(files=files, total=len(files))


@app.route("/api/analyze", methods=["POST"])
def analyze():
    d        = request.json or {}
    path     = d.get("path", "")
    noise_db = float(d.get("noise_db", -35))
    min_dur  = float(d.get("min_dur", 0.04))
    force    = bool(d.get("force", False))
    if not os.path.isfile(path):
        return jsonify(error="File not found"), 400
    key = f"{path}::{noise_db}::{min_dur}"
    if force: _cache.pop(key, None)
    if key not in _cache:
        try:    _cache[key] = detect_transitions(path, noise_db, min_dur)
        except subprocess.TimeoutExpired:
            return jsonify(error="Analysis timed out"), 500
        except Exception as e:
            return jsonify(error=str(e)), 500
    return jsonify(_cache[key])


@app.route("/api/video")
def video():
    path = request.args.get("path", "")
    if not os.path.isfile(path): return "Not found", 404
    return send_file(path, mimetype="video/x-matroska", conditional=True)


@app.route("/api/clip")
def clip():
    path = request.args.get("path", "")
    if not os.path.isfile(path): return "Not found", 404
    start = max(0, int(float(request.args.get("start", 0))))
    start -= start % CLIP_GRID
    f = get_clip(path, start)
    if not f: return "Clip conversion failed - check ffmpeg", 500
    return send_file(f, mimetype="video/mp4", conditional=True)


@app.route("/api/waveform")
def waveform():
    path  = request.args.get("path", "")
    start = float(request.args.get("start", 0))
    dur   = float(request.args.get("dur", 20))
    pts   = int(request.args.get("points", 900))
    if not os.path.isfile(path): return jsonify(error="Not found"), 404
    r = run(["ffmpeg", "-y", "-ss", str(max(0, start)), "-i", path, "-t", str(dur),
             "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "pipe:1"], 120, binary=True)
    raw = r.stdout
    if not raw: return jsonify(samples=[], duration=dur)
    n = len(raw) // 2
    samples = struct.unpack(f"{n}h", raw[:n * 2])
    step = max(1, n // pts)
    result = [round(max(abs(s) for s in samples[i:i + step]) / 32768.0, 4) for i in range(0, n, step)]
    return jsonify(samples=result[:pts], duration=dur)


@app.route("/api/frames")
def frames():
    """Evenly spaced thumbnails (fps cap raised from 12 to 30)."""
    path  = request.args.get("path", "")
    start = float(request.args.get("start", 0))
    dur   = float(request.args.get("dur", 8))
    fps   = min(float(request.args.get("fps", 4)), 30)
    if not os.path.isfile(path): return jsonify(error="Not found"), 404
    with tempfile.TemporaryDirectory() as td:
        run(["ffmpeg", "-y", "-ss", str(max(0, start)), "-i", path, "-t", str(dur), "-an",
             "-vf", f"fps={fps},scale=-1:80", os.path.join(td, "%04d.jpg")], 60, binary=True)
        result = []
        for fn in sorted(f for f in os.listdir(td) if f.endswith(".jpg")):
            with open(os.path.join(td, fn), "rb") as f:
                result.append(base64.b64encode(f.read()).decode())
    return jsonify(frames=result, fps=fps)


@app.route("/api/cutframes")
def cutframes():
    """
    NEW: every real frame around a timestamp, each with its true pts (from showinfo).
    Use this for the preview at the cut point: a 4 fps strip can skip a 42 ms black frame
    entirely, which is why the thumbnails never seemed to match the detected black.
    ?path=&ts=&before=12&after=12&height=120
    """
    path   = request.args.get("path", "")
    ts     = float(request.args.get("ts", 0))
    before = int(request.args.get("before", 12))
    after  = int(request.args.get("after", 12))
    height = int(request.args.get("height", 120))
    if not os.path.isfile(path): return jsonify(error="Not found"), 404
    fps   = probe(path)["fps"]
    start = max(0.0, ts - before / fps)
    with tempfile.TemporaryDirectory() as td:
        r = run(["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{start:.3f}", "-i", path, "-an", "-sn",
                 "-frames:v", str(before + after + 1),
                 "-vf", f"showinfo,scale=-2:{height}", os.path.join(td, "%04d.jpg")], 60)
        # after -ss the output clock restarts at 0, so add the seek offset back
        times = [round(start + float(x), 3) for x in re.findall(r"pts_time:\s*(-?[\d.]+)", r.stderr)]
        out = []
        for i, fn in enumerate(sorted(f for f in os.listdir(td) if f.endswith(".jpg"))):
            with open(os.path.join(td, fn), "rb") as f:
                out.append({"t": times[i] if i < len(times) else None,
                            "img": base64.b64encode(f.read()).decode()})
    return jsonify(frames=out, fps=fps)


@app.route("/api/frame")
def single_frame():
    path = request.args.get("path", "")
    ts   = float(request.args.get("ts", 0))
    if not os.path.isfile(path): return "Not found", 404
    r = run(["ffmpeg", "-y", "-ss", str(max(0, ts)), "-i", path, "-an", "-sn", "-frames:v", "1",
             "-vf", "scale=-1:160", "-f", "image2pipe", "-c:v", "mjpeg", "pipe:1"], 15, binary=True)
    if not r.stdout: return "No frame", 500
    return Response(r.stdout, mimetype="image/jpeg")


@app.route("/api/cut", methods=["POST"])
def cut():
    d     = request.json or {}
    path  = d.get("path", "")
    point = float(d.get("point", 0))
    tool  = d.get("tool", "ffmpeg")
    mode  = d.get("mode", "split")
    if not os.path.isfile(path):
        return jsonify(error="File not found"), 400
    video_dir = os.path.dirname(os.path.abspath(path))
    stem      = os.path.splitext(os.path.basename(path))[0]
    ext       = os.path.splitext(path)[1]
    out_dir   = os.path.join(video_dir, "cuts")
    os.makedirs(out_dir, exist_ok=True)
    use_sc = tool in ("smartcut", "smc")
    exe    = "smart-media-cutter" if tool == "smc" else "smartcut"
    def ff(*args): return subprocess.run(list(args), capture_output=True, text=True, timeout=600)
    out1 = out2 = outs = None
    try:
        if mode == "multi":
            pts = sorted(float(x) for x in d.get("points", []))
            if not pts: return jsonify(error="No cut points marked"), 400
            bounds = [0.0] + pts + [get_duration(path)]
            outs = []
            for n, (a, b) in enumerate(zip(bounds, bounds[1:]), 1):
                o = os.path.join(out_dir, f"{stem}_part{n}{ext}")
                r = ff(exe, path, o, "--keep", f"{a},{b}") if use_sc \
                    else ff("ffmpeg", "-y", "-ss", str(a), "-i", path, "-t", str(b - a), "-c", "copy", o)
                if r.returncode != 0: return jsonify(error=f"Part{n}:\n{r.stderr[-500:]}"), 500
                outs.append(o)
            out1, out2 = outs[0], outs[-1]
        elif mode == "trim_end":
            out1 = os.path.join(out_dir, f"{stem}_trimmed{ext}")
            r1 = ff(exe, path, out1, "--keep", f"0,{point}") if use_sc \
                 else ff("ffmpeg", "-y", "-i", path, "-t", str(point), "-c", "copy", out1)
            if r1.returncode != 0: return jsonify(error=f"Trim:\n{r1.stderr[-500:]}"), 500
        elif mode == "trim_start":
            out1 = os.path.join(out_dir, f"{stem}_trimmed{ext}")
            if use_sc:
                r1 = ff(exe, path, out1, "--keep", f"{point},{get_duration(path)}")
            else:
                r1 = ff("ffmpeg", "-y", "-i", path, "-ss", str(point), "-c", "copy", out1)
            if r1.returncode != 0: return jsonify(error=f"Trim:\n{r1.stderr[-500:]}"), 500
        else:
            out1 = os.path.join(out_dir, f"{stem}_part1{ext}")
            out2 = os.path.join(out_dir, f"{stem}_part2{ext}")
            if use_sc:
                dur = get_duration(path)
                r1  = ff(exe, path, out1, "--keep", f"0,{point}")
                r2  = ff(exe, path, out2, "--keep", f"{point},{dur}")
            else:
                r1 = ff("ffmpeg", "-y", "-i", path, "-t", str(point), "-c", "copy", out1)
                r2 = ff("ffmpeg", "-y", "-i", path, "-ss", str(point), "-c", "copy", out2)
            if r1.returncode != 0: return jsonify(error=f"Part1:\n{r1.stderr[-500:]}"), 500
            if r2.returncode != 0: return jsonify(error=f"Part2:\n{r2.stderr[-500:]}"), 500
        if d.get("keep_original"):   # leave the source where it is so it can be cut again
            return jsonify(success=True, out1=out1, out2=out2, outs=outs, mode=mode,
                           out_dir=out_dir, done_path=None, kept=True)
        done_dir  = os.path.join(video_dir, "done")
        os.makedirs(done_dir, exist_ok=True)
        done_path = os.path.join(done_dir, os.path.basename(path))
        try:
            safe_move(path, done_path)
        except Exception as mv_err:
            return jsonify(success=True, out1=out1, out2=out2, outs=outs, mode=mode,
                           out_dir=out_dir, done_path=None, move_warning=str(mv_err))
        return jsonify(success=True, out1=out1, out2=out2, outs=outs, mode=mode,
                       out_dir=out_dir, done_path=done_path)
    except FileNotFoundError:
        return jsonify(error=f'"{tool}" not found. Install: pip install smartcut'), 500
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/api/move", methods=["POST"])
def move_file():
    d         = request.json or {}
    path      = d.get("path", "")
    subfolder = d.get("subfolder", "skip").strip()
    if not subfolder or "/" in subfolder or "\\" in subfolder or subfolder.startswith("."):
        return jsonify(error="Invalid subfolder name"), 400
    if not os.path.isfile(path): return jsonify(error="File not found"), 400
    dest_dir  = os.path.join(os.path.dirname(os.path.abspath(path)), subfolder)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, os.path.basename(path))
    try:
        safe_move(path, dest_path)
        return jsonify(success=True, dest_path=dest_path)
    except Exception as e:
        return jsonify(error=str(e)), 500



# ─── Final remux of the cut files ─────────────────────────────────────────────
# Cuts made between keyframes (smartcut / ffmpeg -c copy) can leave odd timestamps
# or a broken first GOP. A stream-copy remux rewrites the container cleanly.
# Runs as a background job so the UI can show progress.

REMUX_EXTS = (".mkv", ".mp4")
_remux = {"running": False, "total": 0, "done": 0, "current": "", "failed": [],
          "outdirs": [], "finished": False, "replace": False}
_remux_lock = threading.Lock()


def find_cut_files(root):
    """All media files in any 'cuts' folder under root (or root itself if it is one)."""
    root = os.path.abspath(root)
    cut_dirs = [root] if os.path.basename(root).lower() == "cuts" else []
    for r, dirs, _ in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ("done", "skip", "remuxed")]
        if "cuts" in dirs: cut_dirs.append(os.path.join(r, "cuts"))
    files = []
    for cd in dict.fromkeys(cut_dirs):
        for n in sorted(os.listdir(cd)):
            fp = os.path.join(cd, n)
            if os.path.isfile(fp) and n.lower().endswith(REMUX_EXTS):
                files.append(fp)
    return files


def _remux_one(src, replace):
    outdir = os.path.join(os.path.dirname(src), "remuxed")
    os.makedirs(outdir, exist_ok=True)
    dst = os.path.join(outdir, os.path.basename(src))
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
           "-map", "0:v?", "-map", "0:a?", "-map", "0:s?", "-c", "copy",
           "-avoid_negative_ts", "make_zero"]
    if src.lower().endswith(".mp4"): cmd += ["-movflags", "+faststart"]
    r = run(cmd + [dst], 3600)
    ok = r.returncode == 0 and os.path.isfile(dst) and os.path.getsize(dst) > 0
    with open(os.path.join(outdir, "remux-log.txt"), "a", encoding="utf-8") as lf:
        lf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {'OK  ' if ok else 'FAIL'}  {src}\n")
        if not ok: lf.write(r.stderr[-800:] + "\n")
    if not ok:
        try: os.unlink(dst)
        except OSError: pass
        return False, outdir, r.stderr[-300:]
    if replace:
        safe_move(dst, src)    # overwrite the cut with the clean remux
    return True, outdir, ""


def _remux_worker(files, replace):
    for fp in files:
        with _remux_lock: _remux["current"] = os.path.basename(fp)
        try:    ok, outdir, err = _remux_one(fp, replace)
        except Exception as e: ok, outdir, err = False, os.path.dirname(fp), str(e)
        with _remux_lock:
            _remux["done"] += 1
            if outdir not in _remux["outdirs"]: _remux["outdirs"].append(outdir)
            if not ok: _remux["failed"].append({"file": os.path.basename(fp), "error": err})
    with _remux_lock:
        _remux.update(running=False, finished=True, current="")


@app.route("/api/remux", methods=["POST"])
def remux_start():
    d       = request.json or {}
    folder  = (d.get("folder") or DEFAULT_FOLDER).strip().strip('"')
    replace = bool(d.get("replace", False))
    if not os.path.isdir(folder): return jsonify(error=f"Not a directory: {folder}"), 400
    with _remux_lock:
        if _remux["running"]: return jsonify(error="A remux is already running"), 409
        files = find_cut_files(folder)
        if not files: return jsonify(error="No .mkv/.mp4 files found in any 'cuts' folder"), 404
        _remux.update(running=True, finished=False, total=len(files), done=0,
                      current="", failed=[], outdirs=[], replace=replace)
    threading.Thread(target=_remux_worker, args=(files, replace), daemon=True).start()
    return jsonify(started=True, total=len(files))


@app.route("/api/remux/status")
def remux_status():
    with _remux_lock: return jsonify(dict(_remux))


@app.route("/api/open_folder", methods=["POST"])
def open_folder():
    p = (request.json or {}).get("path", "")
    if not os.path.isdir(p): return jsonify(error="Not a directory"), 400
    if os.name == "nt": os.startfile(p)
    else: subprocess.Popen(["open" if shutil.which("open") else "xdg-open", p])
    return jsonify(success=True)


REMUX_PAGE = """<!doctype html><meta charset=utf-8><title>Remux cuts</title>
<style>body{font:15px system-ui;background:#12141c;color:#dde;max-width:640px;margin:40px auto;padding:0 16px}
input[type=text]{width:100%;padding:8px;background:#1b1e2b;color:#dde;border:1px solid #333;border-radius:6px}
button{padding:9px 16px;margin-top:12px;background:#4b8cff;color:#fff;border:0;border-radius:6px;cursor:pointer}
button:disabled{opacity:.5}#bar{height:10px;background:#1b1e2b;border-radius:5px;margin-top:16px}
#fill{height:100%;width:0;background:#4b8cff;border-radius:5px;transition:width .3s}</style>
<h2>Remux cut files</h2>
<input id=f type=text placeholder="folder containing the 'cuts' folder(s)">
<label><input id=r type=checkbox> overwrite the cut files with the remuxed version</label><br>
<button id=b onclick=go()>Remux all cuts</button> <button id=o onclick=op() disabled>Open folder</button>
<div id=bar><div id=fill></div></div><p id=s></p><pre id=e></pre>
<script>
let dirs=[];fetch('/api/config').then(r=>r.json()).then(c=>f.value=c.default_folder);
async function go(){b.disabled=true;e.textContent='';
 const r=await fetch('/api/remux',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({folder:f.value,replace:r_.checked})});
 const j=await r.json();if(!r.ok){s.textContent=j.error;b.disabled=false;return}poll()}
const r_=document.getElementById('r');
async function poll(){const j=await (await fetch('/api/remux/status')).json();
 fill.style.width=(j.total?100*j.done/j.total:0)+'%';dirs=j.outdirs;
 s.textContent=j.finished?`Done: ${j.done-j.failed.length}/${j.total} remuxed, ${j.failed.length} failed`:`${j.done}/${j.total}  ${j.current}`;
 if(j.failed.length)e.textContent=j.failed.map(x=>x.file+'\\n'+x.error).join('\\n');
 if(j.finished){b.disabled=false;o.disabled=!dirs.length}else setTimeout(poll,800)}
function op(){fetch('/api/open_folder',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:dirs[0]})})}
</script>"""


@app.route("/remux")
def remux_page():
    return REMUX_PAGE


if __name__ == "__main__":
    print("\n  Media Hacksaw  ->  http://localhost:5000\n")
    app.run(port=5000, threaded=True, debug=False)