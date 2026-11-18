"""Motor do JM.

Faz tudo em 5 passos:
1. Acha as batidas da música.
2. Escolhe o trecho mais forte da música (o "drop").
3. Analisa o vídeo: cenas, movimento e barulho.
4. Escolhe as cenas mais fortes, uma para cada corte.
5. Monta o edit em 9:16 com flash e zoom na cor escolhida.

Usa só numpy, scipy (opcional) e ffmpeg.
"""
import os
import re
import subprocess

import numpy as np

SR = 22050          # taxa de amostragem usada para analisar a música
HOP = 512
N_FFT = 1024
W, H = 720, 1280    # tamanho do vídeo final (9:16)
FPS = 30
AF = 4              # quadros por segundo usados para analisar o vídeo


# ---------------------------------------------------------------- utilidades
def _run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode(errors="ignore")[-800:])
    return p.stdout


def probe_duration(path):
    out = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1", path])
    return float(out.decode().strip())


def load_audio(path, sr=SR, max_seconds=None):
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    out = _run(cmd)
    return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0


def _norm01(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.percentile(x, 10), np.percentile(x, 95)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def clean_color(c):
    if isinstance(c, str) and re.fullmatch(r"#?[0-9a-fA-F]{6}", c):
        return c.lstrip("#").lower()
    return "ffffff"


# ------------------------------------------------------------------- música
def onset_strength(y):
    """Quanto 'bate' a música a cada instante (fluxo espectral)."""
    if len(y) < N_FFT * 8:
        raise ValueError("A música é curta demais.")
    y = np.ascontiguousarray(y, dtype=np.float32)
    n_frames = 1 + (len(y) - N_FFT) // HOP
    st = y.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        y, shape=(n_frames, N_FFT), strides=(st * HOP, st))
    spec = np.abs(np.fft.rfft(frames * np.hanning(N_FFT).astype(np.float32), axis=1))
    spec = np.log1p(spec * 20.0)
    flux = np.maximum(0.0, np.diff(spec, axis=0)).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    k = 43
    local = np.convolve(flux, np.ones(k) / k, mode="same")
    return np.maximum(0.0, flux - local)


def find_beats(flux, fps, bpm_min=75, bpm_max=170):
    """Acha o tempo (BPM) e a posição de cada batida."""
    n = len(flux)
    f = flux / (flux.max() + 1e-9)
    size = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(f, size)
    ac = np.fft.irfft(np.abs(spec) ** 2)[:n]

    lo = max(2, int(fps * 60 / bpm_max))
    hi = min(n - 1, int(fps * 60 / bpm_min))
    if hi <= lo:
        raise ValueError("Não consegui achar o ritmo dessa música.")
    lag = lo + int(np.argmax(ac[lo:hi]))

    fm = f.copy()
    fm[1:] = np.maximum(fm[1:], f[:-1])
    fm[:-1] = np.maximum(fm[:-1], f[1:])

    best_score, best_idx, best_period = -1.0, None, None
    for period in np.arange(lag - 1.5, lag + 1.51, 0.1):
        if period < 3:
            continue
        for off in range(int(period) + 1):
            k = np.arange(0, int((n - 1 - off) / period) + 1)
            idx = np.round(off + k * period).astype(int)
            idx = idx[idx < n]
            score = fm[idx].sum()
            if score > best_score:
                best_score, best_idx, best_period = score, idx, period
    return best_idx / fps, best_period / fps


def pick_window(beats, flux, fps, duration, total):
    """Escolhe o começo (numa batida) do trecho mais forte da música."""
    win = int(duration * fps)
    cs = np.concatenate([[0.0], np.cumsum(flux)])
    best_t, best = float(beats[0]), -1.0
    for b in beats:
        if b + duration > total - 0.2:
            break
        i = int(b * fps)
        j = min(len(flux), i + win)
        s = cs[j] - cs[i]
        if s > best:
            best, best_t = s, float(b)
    return best_t


def plan_cuts(beats, flux, fps, start, duration, beats_per_cut):
    sel = [float(b) for b in beats if start - 1e-6 <= b < start + duration - 0.3]
    sel = sel[::beats_per_cut]
    if not sel:
        raise ValueError("Não consegui montar os cortes com essa música.")
    bounds = [b - start for b in sel]
    bounds[0] = 0.0
    strengths = []
    for b in sel:
        i = int(b * fps)
        strengths.append(float(flux[max(0, i - 2):i + 3].max()))
    bounds.append(duration)
    if len(bounds) > 3 and (bounds[-1] - bounds[-2]) < 0.5 * (bounds[1] - bounds[0]):
        del bounds[-2]
        strengths = strengths[:len(bounds) - 1]
    med = float(np.median(strengths)) if strengths else 0.0
    strong = [s >= med for s in strengths]
    return bounds, strong


# -------------------------------------------------------------------- vídeo
def analyze_video(path):
    """Devolve (pontuação por quadro, quadros onde começa uma cena nova)."""
    cmd = ["ffmpeg", "-v", "error", "-threads", "2", "-i", path, "-an",
           "-vf", f"fps={AF},scale=64:36,format=gray", "-f", "rawvideo", "-"]
    raw = _run(cmd)
    fsize = 64 * 36
    n = len(raw) // fsize
    if n < AF * 5:
        raise ValueError("O vídeo é curto demais.")
    frames = np.frombuffer(raw[:n * fsize], dtype=np.uint8).reshape(n, fsize).astype(np.float32)
    d = np.abs(np.diff(frames, axis=0)).mean(axis=1)        # d[i]: quadro i -> i+1

    thr = max(18.0, float(np.median(d) + 4 * d.std()))
    cut_idx = np.where(d > thr)[0]
    scene_starts = [0]
    for i in cut_idx:
        s = int(i) + 1
        if s - scene_starts[-1] >= 2:
            scene_starts.append(s)

    motion = d.copy()
    motion[cut_idx] = 0.0
    motion = np.concatenate([[0.0], motion])
    motion = np.convolve(motion, np.ones(3) / 3, mode="same")

    try:
        a = load_audio(path, sr=8000)
        step = 8000 // AF
        m = len(a) // step
        rms = np.sqrt((a[:m * step].reshape(m, step) ** 2).mean(axis=1)) if m else np.zeros(1)
    except RuntimeError:
        rms = np.zeros(1)
    loud = np.zeros(n)
    k = min(n, len(rms))
    loud[:k] = rms[:k]

    score = 0.6 * _norm01(motion) + 0.4 * _norm01(loud)
    return score, scene_starts


def pick_clips(score, scene_starts, slot_lens):
    """Escolhe um trecho do vídeo para cada corte, sem repetir."""
    n = len(score)
    need = len(slot_lens)
    lmax = max(slot_lens)
    wf = max(2, int(np.ceil(lmax * AF)))
    cs = np.concatenate([[0.0], np.cumsum(score)])

    def candidates(starts):
        out = []
        ends = list(starts[1:]) + [n]
        for sf, ef in zip(starts, ends):
            t = sf + 1
            while t + wf <= ef - 1:
                out.append(((cs[t + wf] - cs[t]) / wf, t))
                t += 2
        return out

    cands = candidates(scene_starts)
    if len(cands) < need:
        cands = candidates([0])
    if not cands:
        raise ValueError("O vídeo é curto demais para esse edit.")
    cands.sort(reverse=True)

    chosen = []
    for gap in (wf + 1, max(1, wf // 2), 0):
        chosen = []
        for sc, t in cands:
            if all(abs(t - c) > gap for c in chosen):
                chosen.append(t)
                if len(chosen) == need:
                    break
        if len(chosen) == need:
            break
    chosen.sort()
    return [t / AF for t in chosen]


# ------------------------------------------------------------------- montagem
def render_segment(video, start, nframes, color, strong, out):
    zoom = 0.12 if strong else 0.06
    z = f"(1+{zoom}*max(0,1-t/0.28))"
    vf = (
        f"scale={W}:{H}:force_original_aspect_ratio=increase,"
        f"scale=w='trunc(iw*{z}/2)*2':h='trunc(ih*{z}/2)*2':eval=frame,"
        f"crop={W}:{H},setsar=1,"
        f"eq=contrast=1.08:saturation=1.25,"
        f"drawbox=x=0:y=0:w=iw:h=ih:color=0x{color}@0.55:t=fill:enable='lt(t,0.09)',"
        f"drawbox=x=0:y=0:w=iw:h=ih:color=0x{color}@0.85:t=12"
    )
    cmd = ["ffmpeg", "-y", "-v", "error", "-threads", "2", "-ss", f"{start:.3f}",
           "-i", video, "-an", "-vf", vf, "-r", str(FPS), "-frames:v", str(nframes),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
           "-pix_fmt", "yuv420p", "-g", str(FPS), out]
    _run(cmd)


def build_edit(music_path, video_path, out_path, duration=30, beats_per_cut=2,
               color="#ffffff", workdir=None, progress=lambda p, m: None):
    workdir = workdir or os.path.dirname(out_path)
    color = clean_color(color)

    # 1) música
    progress(0.03, "Analisando a música…")
    y = load_audio(music_path, max_seconds=480)
    total = len(y) / SR
    if total < 8:
        raise ValueError("A música precisa ter pelo menos 8 segundos.")
    duration = float(min(duration, total - 0.5))
    flux = onset_strength(y)
    fps = SR / HOP
    beats, period = find_beats(flux, fps)
    start = pick_window(beats, flux, fps, duration, total)
    bounds, strong = plan_cuts(beats, flux, fps, start, duration, beats_per_cut)
    slots = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]

    # 2) vídeo
    progress(0.12, "Analisando o vídeo (pode demorar)…")
    score, scene_starts = analyze_video(video_path)
    progress(0.40, "Escolhendo as melhores cenas…")
    starts = pick_clips(score, scene_starts, slots)
    if len(starts) < len(slots):
        slots = slots[:len(starts)]
        strong = strong[:len(starts)]
        bounds = bounds[:len(starts) + 1]
    # segue a ordem do vídeo; se faltou cena, o edit fica um pouco mais curto
    total_len = bounds[-1]

    # 3) montagem
    seg_files = []
    cum = [int(round(b * FPS)) for b in bounds]
    for i, st in enumerate(starts):
        nframes = max(2, cum[i + 1] - cum[i])
        seg = os.path.join(workdir, f"seg_{i:03d}.mp4")
        render_segment(video_path, st, nframes, color, strong[i], seg)
        seg_files.append(seg)
        progress(0.40 + 0.50 * (i + 1) / len(starts),
                 f"Montando os cortes ({i + 1}/{len(starts)})…")

    progress(0.93, "Juntando com a música…")
    lst = os.path.join(workdir, "list.txt")
    with open(lst, "w") as f:
        for s in seg_files:
            f.write(f"file '{s}'\n")
    total_len = sum(cum[i + 1] - cum[i] for i in range(len(seg_files))) / FPS
    fade = min(0.8, total_len / 4)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
           "-ss", f"{start:.3f}", "-t", f"{total_len:.3f}", "-i", music_path,
           "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
           "-af", f"afade=t=in:d=0.05,afade=t=out:st={max(0, total_len - fade):.3f}:d={fade:.3f}",
           "-shortest", "-movflags", "+faststart", out_path]
    _run(cmd)
    for s in seg_files:
        try:
            os.remove(s)
        except OSError:
            pass
    progress(1.0, "Pronto!")
    return {"bpm": round(60.0 / period), "cuts": len(starts), "seconds": round(total_len, 1)}
