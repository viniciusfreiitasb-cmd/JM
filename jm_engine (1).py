"""Motor do JM (versão 2).

1. Acha as batidas da música e o trecho mais forte dela.
2. Analisa o vídeo: cenas, movimento, barulho e onde está a ação.
3. Escolhe cenas INTEIRAS (cada corte vai até o fim da jogada) e encaixa nas batidas.
4. Monta em 9:16 (fundo desfocado ou seguindo a ação) com transições e efeitos.

Usa só numpy e ffmpeg.
"""
import json
import os
import re
import subprocess

import numpy as np

SR = 22050          # taxa usada para analisar a música
HOP = 512
N_FFT = 1024
W, H = 720, 1280    # tamanho do vídeo final (9:16)
FPS = 30
AF = 8              # quadros por segundo usados para analisar o vídeo

# velocidade dos cortes: (mínimo, máximo) de batidas por corte
RHYTHMS = {1: (1, 2), 2: (2, 4), 4: (4, 8)}


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


def probe_size(path):
    """Largura e altura do vídeo já considerando a rotação do celular."""
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=width,height:stream_tags=rotate:stream_side_data=rotation",
                "-of", "json", path])
    d = json.loads(out)["streams"][0]
    w, h = int(d["width"]), int(d["height"])
    rot = 0
    try:
        rot = int(d.get("tags", {}).get("rotate", 0))
    except (TypeError, ValueError):
        pass
    for sd in d.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    if abs(rot) % 180 == 90:
        w, h = h, w
    return w, h


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


_FILTERS = None


def has_filter(name):
    global _FILTERS
    if _FILTERS is None:
        try:
            txt = _run(["ffmpeg", "-hide_banner", "-filters"]).decode(errors="ignore")
            _FILTERS = set(re.findall(r"^\s*[A-Z.]{3}\s+(\w+)\s", txt, re.M))
        except Exception:                                   # noqa: BLE001
            _FILTERS = set()
    return name in _FILTERS


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


def beat_grid(beats, start, duration):
    """Tempo de cada batida dentro do trecho escolhido (0 = primeira batida)."""
    i0 = int(np.argmin(np.abs(beats - start)))
    grid = [float(b) - start for b in beats[i0:] if b - start <= duration + 1e-6]
    if len(grid) < 3:
        raise ValueError("Não consegui montar os cortes com essa música.")
    grid[0] = 0.0
    return i0, grid


# -------------------------------------------------------------------- vídeo
def analyze_video(path):
    """Analisa o vídeo em baixa resolução.

    Devolve cenas, pontuação de cada quadro (movimento + barulho) e
    onde está a ação na horizontal (0 = esquerda, 1 = direita).
    """
    cmd = ["ffmpeg", "-v", "error", "-threads", "2", "-i", path, "-an",
           "-vf", f"fps={AF},scale=64:36,format=gray", "-f", "rawvideo", "-"]
    raw = _run(cmd)
    fsize = 64 * 36
    n = len(raw) // fsize
    if n < AF * 5:
        raise ValueError("O vídeo é curto demais.")
    frames = np.frombuffer(raw[:n * fsize], dtype=np.uint8).reshape(n, 36, 64)
    del raw

    m = n - 1
    dmean = np.empty(m, dtype=np.float32)
    colw = np.empty((m, 64), dtype=np.float32)
    for a in range(0, m, 1000):
        b = min(m, a + 1000)
        diff = np.abs(frames[a + 1:b + 1].astype(np.int16)
                      - frames[a:b].astype(np.int16)).astype(np.float32)
        dmean[a:b] = diff.mean(axis=(1, 2))
        colw[a:b] = diff.sum(axis=1)

    med_d = float(np.median(dmean))
    mad = float(np.median(np.abs(dmean - med_d))) * 1.4826     # robusto: os cortes não inflam
    thr = max(20.0, med_d + 8 * mad)
    cut_idx = np.where(dmean > thr)[0]
    scene_starts = [0]
    for i in cut_idx:
        s = int(i) + 1
        if s - scene_starts[-1] >= 3:
            scene_starts.append(s)

    motion = dmean.copy()
    motion[cut_idx] = 0.0
    motion = np.concatenate([[0.0], motion])
    motion = np.convolve(motion, np.ones(3) / 3, mode="same")

    # onde está a ação (colunas da imagem que mais mudam)
    wgt = np.maximum(0.0, colw - 1.5 * colw.mean(axis=1, keepdims=True))
    tot = wgt.sum(axis=1)
    xs = np.arange(64, dtype=np.float32)
    cx = np.where(tot > 1e-6, (wgt * xs).sum(axis=1) / np.maximum(tot, 1e-6) / 63.0, 0.5)
    tot[cut_idx] = 0.0
    cx[cut_idx] = 0.5
    cx = np.concatenate([[0.5], cx])
    cw = np.concatenate([[0.0], tot])

    # barulho do vídeo (torcida, narração)
    loud = np.zeros(n)
    try:
        a = load_audio(path, sr=4000)
        step = 4000 // AF
        k = len(a) // step
        if k:
            rms = np.empty(k, dtype=np.float32)
            for s0 in range(0, k, 4000):
                s1 = min(k, s0 + 4000)
                blk = a[s0 * step:s1 * step].reshape(s1 - s0, step)
                rms[s0:s1] = np.sqrt((blk * blk).mean(axis=1))
            kk = min(n, k)
            loud[:kk] = rms[:kk]
    except RuntimeError:
        pass

    score = 0.6 * _norm01(motion) + 0.4 * _norm01(loud)
    return {"score": score, "scenes": scene_starts, "cx": cx, "cw": cw, "n": n}


def pick_clips(info, period, kmin, kmax, total_beats):
    """Escolhe as cenas mais fortes e deixa cada corte ir até o fim da jogada.

    Cada cena vira um corte de k batidas, alinhado com o FIM da cena. Se a
    cena é maior que o máximo, fica a parte final dela (onde a jogada termina).
    """
    score, starts, n = info["score"], info["scenes"], info["n"]
    pf = period * AF                                   # quadros por batida
    cs = np.concatenate([[0.0], np.cumsum(score)])

    def mean(a, b):
        return float((cs[b] - cs[a]) / max(1, b - a))

    ends = list(starts[1:]) + [n]
    cands = []                                         # (nota, quadro_inicial, k)
    for sf, ef in zip(starts, ends):
        a, b = sf + 1, ef - 1                          # folga de 1 quadro nas pontas
        usable = (b - a) / pf
        k = min(max(kmin, min(kmax, int(round(usable)))), int(usable + 1e-9))
        if k < 1:
            continue
        wf = int(round(k * pf))
        t0 = max(a, b - wf)
        cands.append((mean(t0, min(b, t0 + wf)) + 0.05, t0, k))
        if usable > 2.5 * kmax:                        # cena longa: mais trechos antes do final
            wfx = int(round(kmax * pf))
            cur, extra = t0 - wfx, 0
            while cur >= a and extra < 4:
                cands.append((mean(cur, cur + wfx), cur, kmax))
                cur -= wfx
                extra += 1
    if not cands:
        raise ValueError("Não achei cenas boas nesse vídeo.")
    cands.sort(reverse=True)

    chosen, got = [], 0
    for sc, t0, k in cands:
        chosen.append((t0, k))
        got += k
        if got >= total_beats:
            break
    chosen.sort()

    clips, used = [], 0
    for t0, k in chosen:
        if used >= total_beats:
            break
        kk = min(k, total_beats - used)
        t0 += int(round((k - kk) * pf))                # se precisar encurtar, mantém o final
        clips.append({"t0f": t0, "k": kk, "wf": int(round(kk * pf))})
        used += kk
    return clips


def pan_centers(info, t0f, wf):
    """Onde está a ação no começo e no fim do corte (0 a 1)."""
    cx, cw = info["cx"], info["cw"]
    a, b = t0f, min(len(cx), t0f + max(2, wf))
    mid = (a + b) // 2

    def avg(x, y):
        w = cw[x:y]
        return float((cx[x:y] * w).sum() / w.sum()) if w.sum() > 1e-6 else None

    c0, c1 = avg(a, mid), avg(mid, b)
    if c0 is None and c1 is None:
        return 0.5, 0.5
    c0 = c1 if c0 is None else c0
    c1 = c0 if c1 is None else c1
    # puxa só um pouquinho pro centro para não ficar nervoso
    return 0.9 * c0 + 0.05, 0.9 * c1 + 0.05


# ------------------------------------------------------------------- montagem
STRONG_KINDS = ["punch", "shake", "glitch", "twist"]
WEAK_KINDS = ["slide_l", "slide_r", "push", "punch"]


def base_graph(framing, dur, c0, c1):
    """Deixa o corte em 720x1280. Saída: [base]."""
    if framing == "follow":
        x = f"min(max(({c0:.4f}+({c1:.4f}-{c0:.4f})*t/{dur:.3f})*iw-ow/2,0),iw-ow)"
        return (f"[0:v]crop=w='trunc(ih*9/16/2)*2':h=ih:x='{x}':y=0,"
                f"scale={W}:{H}:flags=bicubic,setsar=1[base]")
    if framing == "blur":
        return (f"[0:v]split=2[bg][fg];"
                f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H},boxblur=24:3,eq=brightness=-0.10[b];"
                f"[fg]scale={W}:-2[f];"
                f"[b][f]overlay=x=(W-w)/2:y=(H-h)/2,setsar=1[base]")
    return (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H},setsar=1[base]")


def effect_chain(kind, strong, dur):
    s = 1.0 if strong else 0.6
    if kind == "glitch" and not has_filter("rgbashift"):
        kind = "shake"
    if kind == "punch":
        z = f"(1+{0.04 + 0.16 * s:.3f}*max(0,1-t/0.28))"
        return (f"scale=w='trunc(iw*{z}/2)*2':h='trunc(ih*{z}/2)*2':eval=frame,"
                f"crop={W}:{H}")
    if kind == "push":
        z = f"(1+0.12*min(t/{max(dur, 0.2):.3f},1))"
        return (f"scale=w='trunc(iw*{z}/2)*2':h='trunc(ih*{z}/2)*2':eval=frame,"
                f"crop={W}:{H}")
    if kind in ("slide_l", "slide_r"):
        sgn = 1 if kind == "slide_l" else -1
        sw, sh = int(W * 1.25) // 2 * 2, int(H * 1.25) // 2 * 2
        return (f"scale={sw}:{sh},crop={W}:{H}:"
                f"x='(iw-ow)/2+({sgn}*(iw-ow)/2)*pow(max(0,1-t/0.3),2)':y='(ih-oh)/2'")
    if kind == "shake":
        a = int(18 + 18 * s)
        return (f"scale=792:1408,crop={W}:{H}:"
                f"x='36+{a}*sin(t*75)*max(0,1-t/0.4)':"
                f"y='64+{a}*sin(t*93+1)*max(0,1-t/0.4)'")
    if kind == "twist":
        ang = 0.12 + 0.08 * s
        return (f"scale=1008:1792,rotate=a='{ang:.3f}*max(0,1-t/0.32)':ow=iw:oh=ih:c=black,"
                f"crop={W}:{H}")
    if kind == "glitch":
        return (f"rgbashift=rh=-16:bh=16:gv=6:enable='lt(t,0.2)',"
                f"scale=792:1408,crop={W}:{H}:"
                f"x='36+22*sin(t*120)*max(0,1-t/0.25)':y='64'")
    return f"scale={W}:{H}"


def render_segment(video, start, nframes, color, strong, kind, framing, c0, c1, out):
    dur = nframes / FPS
    s = 1.0 if strong else 0.6
    tail = (
        f"eq=contrast=1.08:saturation=1.25:"
        f"brightness='{0.10 + 0.08 * s:.3f}*max(0,1-t/0.18)':eval=frame,"
        f"drawbox=x=0:y=0:w=iw:h=ih:color=0x{color}@0.55:t=fill:enable='lt(t,0.09)',"
        f"drawbox=x=0:y=0:w=iw:h=ih:color=0x{color}@0.85:t=12,"
        f"format=yuv420p"
    )
    graph = f"{base_graph(framing, dur, c0, c1)};[base]{effect_chain(kind, strong, dur)},{tail}[out]"
    cmd = ["ffmpeg", "-y", "-v", "error", "-threads", "2", "-ss", f"{start:.3f}",
           "-i", video, "-an", "-filter_complex", graph, "-map", "[out]",
           "-r", str(FPS), "-frames:v", str(nframes),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
           "-pix_fmt", "yuv420p", "-g", str(FPS), out]
    _run(cmd)


def build_edit(music_path, video_path, out_path, duration=30, beats_per_cut=2,
               color="#ffffff", framing="blur", workdir=None,
               progress=lambda p, m: None, debug=None):
    workdir = workdir or os.path.dirname(out_path)
    color = clean_color(color)
    kmin, kmax = RHYTHMS.get(beats_per_cut, RHYTHMS[2])

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
    i0, grid = beat_grid(beats, start, duration)
    total_beats = len(grid) - 1
    bstrength = []
    for j in range(len(grid)):
        i = int(beats[min(len(beats) - 1, i0 + j)] * fps)
        bstrength.append(float(flux[max(0, i - 2):i + 3].max()))
    med = float(np.median(bstrength))

    # 2) vídeo
    progress(0.12, "Analisando o vídeo (pode demorar)…")
    vw, vh = probe_size(video_path)
    if vw / vh <= 0.62:
        framing = "cover"                    # vídeo já é em pé
    elif framing not in ("blur", "follow"):
        framing = "blur"
    info = analyze_video(video_path)
    progress(0.40, "Escolhendo as melhores cenas…")
    clips = pick_clips(info, period, kmin, kmax, total_beats)
    used = sum(c["k"] for c in clips)
    if used < 2:
        raise ValueError("O vídeo é curto demais para esse edit.")
    bounds = [grid[j] for j in range(used + 1)]

    # 3) montagem
    seg_files, cum, j = [], [int(round(b * FPS)) for b in bounds], 0
    cnt = {"s": 0, "w": 0}
    for i, c in enumerate(clips):
        j0 = j
        j += c["k"]
        nframes = max(2, int(round(bounds[j] * FPS)) - int(round(bounds[j0] * FPS)))
        strong = bstrength[j0] >= med
        pool = STRONG_KINDS if strong else WEAK_KINDS
        key = "s" if strong else "w"
        kind = pool[cnt[key] % len(pool)]
        cnt[key] += 1
        c0, c1 = pan_centers(info, c["t0f"], c["wf"]) if framing == "follow" else (0.5, 0.5)
        src = c["t0f"] / AF
        if debug is not None:
            debug.append({"src": round(src, 2), "len_beats": c["k"], "kind": kind,
                          "strong": bool(strong), "pan": (round(c0, 2), round(c1, 2)),
                          "out_start": round(bounds[j0], 2)})
        seg = os.path.join(workdir, f"seg_{i:03d}.mp4")
        render_segment(video_path, src, nframes, color, strong, kind, framing, c0, c1, seg)
        seg_files.append(seg)
        progress(0.40 + 0.50 * (i + 1) / len(clips),
                 f"Montando os cortes ({i + 1}/{len(clips)})…")

    progress(0.93, "Juntando com a música…")
    lst = os.path.join(workdir, "list.txt")
    with open(lst, "w") as f:
        for sfile in seg_files:
            f.write(f"file '{sfile}'\n")
    total_len = int(round(bounds[used] * FPS)) / FPS
    fade = min(0.8, total_len / 4)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
           "-ss", f"{start:.3f}", "-t", f"{total_len:.3f}", "-i", music_path,
           "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
           "-af", f"afade=t=in:d=0.05,afade=t=out:st={max(0, total_len - fade):.3f}:d={fade:.3f}",
           "-shortest", "-movflags", "+faststart", out_path]
    _run(cmd)
    for sfile in seg_files:
        try:
            os.remove(sfile)
        except OSError:
            pass
    progress(1.0, "Pronto!")
    return {"bpm": round(60.0 / period), "cuts": len(clips), "seconds": round(total_len, 1)}
