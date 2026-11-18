"""JM: site que monta edits de vídeo no ritmo da música (formato 9:16)."""
import os
import re
import shutil
import threading
import time
import uuid

from flask import Flask, jsonify, render_template, request, send_file

import jm_engine

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=BASE)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024   # cada pedaço enviado é pequeno

DATA = os.environ.get("JM_DATA", "/tmp/jm")
os.makedirs(DATA, exist_ok=True)

JOBS = {}
RENDER_LOCK = threading.Lock()          # um edit por vez (o servidor grátis é fraco)
ID_RE = re.compile(r"^[0-9a-f]{32}$")
KINDS = {"music": "musica", "video": "video"}


def job_dir(job_id):
    if not ID_RE.match(job_id):
        raise KeyError
    return os.path.join(DATA, job_id)


def cleanup_old(max_age=3 * 3600):
    now = time.time()
    for name in os.listdir(DATA):
        p = os.path.join(DATA, name)
        try:
            if now - os.path.getmtime(p) > max_age:
                shutil.rmtree(p, ignore_errors=True)
                JOBS.pop(name, None)
        except OSError:
            pass


@app.errorhandler(KeyError)
def bad_id(_):
    return jsonify(error="Edit não encontrado."), 404


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/manifest.webmanifest")
def manifest():
    return jsonify(
        name="JM", short_name="JM", start_url="/", display="standalone",
        background_color="#0c0c10", theme_color="#0c0c10",
        icons=[{"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml",
                "purpose": "any maskable"}])


@app.route("/icon.svg")
def icon():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">'
           '<rect width="512" height="512" fill="#0c0c10"/>'
           '<text x="256" y="330" font-family="Arial Black,Arial,sans-serif" '
           'font-size="230" font-weight="900" text-anchor="middle" fill="#ffffff">JM</text>'
           '<rect x="96" y="372" width="320" height="16" rx="8" fill="#a855f7"/></svg>')
    return app.response_class(svg, mimetype="image/svg+xml")


@app.route("/api/new", methods=["POST"])
def new_job():
    cleanup_old()
    job_id = uuid.uuid4().hex
    os.makedirs(job_dir(job_id))
    JOBS[job_id] = {"state": "idle", "progress": 0.0, "stage": "", "error": "",
                    "chunks": {}, "result": None}
    return jsonify(id=job_id)


@app.route("/api/upload/<job_id>/<kind>", methods=["POST"])
def upload(job_id, kind):
    d = job_dir(job_id)
    job = JOBS.get(job_id)
    if job is None or kind not in KINDS:
        return jsonify(error="Edit não encontrado."), 404
    try:
        i = int(request.form.get("i", "0"))
    except ValueError:
        return jsonify(error="Pedaço inválido."), 400
    f = request.files.get("file")
    if f is None:
        return jsonify(error="Faltou o arquivo."), 400

    path = os.path.join(d, KINDS[kind])
    last = job["chunks"].get(kind, -1)
    if i == last:                       # o celular reenviou: já recebemos esse pedaço
        return jsonify(ok=True)
    if i != last + 1:
        return jsonify(error="Pedaço fora de ordem.", expected=last + 1), 409
    with open(path, "wb" if i == 0 else "ab") as out:
        shutil.copyfileobj(f.stream, out)
    job["chunks"][kind] = i
    return jsonify(ok=True)


def _work(job_id, duration, rhythm, color):
    job = JOBS[job_id]
    d = job_dir(job_id)

    def progress(p, msg):
        job["progress"] = p
        job["stage"] = msg

    job["state"] = "waiting"
    job["stage"] = "Na fila…"
    with RENDER_LOCK:
        job["state"] = "running"
        try:
            out = os.path.join(d, "edit.mp4")
            job["result"] = jm_engine.build_edit(
                os.path.join(d, "musica"), os.path.join(d, "video"), out,
                duration=duration, beats_per_cut=rhythm, color=color,
                workdir=d, progress=progress)
            job["state"] = "done"
            for name in ("musica", "video"):       # libera espaço
                try:
                    os.remove(os.path.join(d, name))
                except OSError:
                    pass
        except ValueError as e:
            job["state"], job["error"] = "error", str(e)
        except Exception as e:                      # noqa: BLE001
            job["state"] = "error"
            job["error"] = "Deu um erro ao montar o edit. Tenta de novo ou com outro vídeo."
            app.logger.exception("falha no edit: %s", e)


@app.route("/api/start/<job_id>", methods=["POST"])
def start(job_id):
    job_dir(job_id)
    job = JOBS.get(job_id)
    if job is None:
        return jsonify(error="Edit não encontrado."), 404
    if "music" not in job["chunks"] or "video" not in job["chunks"]:
        return jsonify(error="Envie a música e o vídeo primeiro."), 400
    if job["state"] in ("waiting", "running"):
        return jsonify(ok=True)
    body = request.get_json(silent=True) or {}
    duration = body.get("duration", 30)
    rhythm = body.get("rhythm", 2)
    duration = duration if duration in (15, 30, 45, 60) else 30
    rhythm = rhythm if rhythm in (1, 2, 4) else 2
    color = body.get("color", "#ffffff")
    job.update(state="waiting", progress=0.0, stage="Na fila…", error="", result=None)
    threading.Thread(target=_work, args=(job_id, duration, rhythm, color),
                     daemon=True).start()
    return jsonify(ok=True)


@app.route("/api/status/<job_id>")
def status(job_id):
    job_dir(job_id)
    job = JOBS.get(job_id)
    if job is None:
        return jsonify(error="Edit não encontrado."), 404
    return jsonify(state=job["state"], progress=round(job["progress"], 3),
                   stage=job["stage"], error=job["error"], result=job["result"])


@app.route("/api/download/<job_id>")
def download(job_id):
    d = job_dir(job_id)
    path = os.path.join(d, "edit.mp4")
    if not os.path.exists(path):
        return jsonify(error="O edit ainda não está pronto."), 404
    inline = request.args.get("inline") == "1"
    return send_file(path, mimetype="video/mp4", conditional=True,
                     as_attachment=not inline, download_name="JM-edit.mp4")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), threaded=True)
