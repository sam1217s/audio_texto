#!/usr/bin/env python3
"""
Transcriptor de Audio con Whisper - Interfaz Web Flask
Ejecutar: python app.py
Luego abrir: http://localhost:5200
"""
import time

import os
import uuid

# Asegurar que FFmpeg esté disponible para Python y Whisper
_FFMPEG_BIN = os.path.expandvars(
    r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.1-full_build\bin"
)
if os.path.isdir(_FFMPEG_BIN) and _FFMPEG_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")
import sys
import threading
import types
import whisper
import whisper.transcribe  # importa el módulo aunque el nombre quede sombreado
import subprocess
import json
from tqdm import tqdm as _OriginalTqdm
from flask import Flask, request, render_template, jsonify, send_file
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Cargar variables del .env (HF_TOKEN para pyannote)
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

# === Límites de seguridad para diarización ===
# Si Whisper genera más segmentos que esto, abortamos la diarización antes de
# que pyannote construya una matriz O(n²) y reviente la RAM. Para 50 min con
# `large` lo normal es ~600-1500 segmentos. 3500 es un techo conservador.
MAX_SEGMENTOS_DIARIZACION = 3500
# Si el audio supera esta duración (en min) sin que el usuario haya indicado
# num_hablantes, fijamos un max_speakers por defecto para acotar el clustering.
DEFAULT_MAX_SPEAKERS_AUDIO_LARGO = 6
DURACION_AUDIO_LARGO_MIN = 30

# Activar TF32 para acelerar inferencia en GPU NVIDIA (Ampere/Ada/Blackwell).
# Pequeña pérdida de precisión teórica, ganancia 2-3x en matmul fp32.
try:
    import torch as _torch_init
    if _torch_init.cuda.is_available():
        _torch_init.backends.cuda.matmul.allow_tf32 = True
        _torch_init.backends.cudnn.allow_tf32 = True
        _torch_init.backends.cudnn.benchmark = True
except Exception:
    pass

# Pipeline de diarización (lazy load)
_diarization_pipeline = None
_diarization_lock = threading.Lock()


def get_diarization_pipeline():
    """Carga el pipeline de pyannote la primera vez que se usa."""
    global _diarization_pipeline
    with _diarization_lock:
        if _diarization_pipeline is None:
            from pyannote.audio import Pipeline
            import torch
            # Intentamos primero el modelo nuevo (community-1) y caemos al
            # 3.1 si no está disponible (más rápido en GPU para audios largos).
            modelos = [
                "pyannote/speaker-diarization-3.1",
                "pyannote/speaker-diarization-community-1",
            ]
            ultimo_error = None
            for nombre in modelos:
                try:
                    _diarization_pipeline = Pipeline.from_pretrained(
                        nombre, token=HF_TOKEN
                    )
                    print(f"[diarización] Usando modelo: {nombre}")
                    break
                except Exception as e:
                    ultimo_error = e
                    print(f"[diarización] No se pudo cargar {nombre}: {e}")
            if _diarization_pipeline is None:
                raise RuntimeError(
                    f"No se pudo cargar ningún modelo de diarización. "
                    f"Último error: {ultimo_error}"
                )
            if torch.cuda.is_available():
                _diarization_pipeline.to(torch.device("cuda"))
                print("[diarización] Pipeline movido a GPU")
    return _diarization_pipeline


def asignar_hablantes(segmentos_whisper, diarizacion):
    """Para cada segmento de Whisper, encuentra el hablante con más solapamiento."""
    # pyannote 4.x devuelve DiarizeOutput; las versiones viejas devuelven Annotation directo
    if hasattr(diarizacion, "speaker_diarization"):
        anotacion = diarizacion.speaker_diarization
    else:
        anotacion = diarizacion
    turnos = [(t.start, t.end, spk) for t, _, spk in anotacion.itertracks(yield_label=True)]
    resultado = []
    for seg in segmentos_whisper:
        ini, fin = seg["start"], seg["end"]
        mejor_spk = None
        mejor_overlap = 0.0
        for t_ini, t_fin, spk in turnos:
            overlap = max(0.0, min(fin, t_fin) - max(ini, t_ini))
            if overlap > mejor_overlap:
                mejor_overlap = overlap
                mejor_spk = spk
        resultado.append({
            "start": ini, "end": fin,
            "speaker": mejor_spk or "DESCONOCIDO",
            "text": seg["text"].strip()
        })
    return resultado


def formatear_diarizacion(segs_con_hablante):
    """Convierte los segmentos a texto con etiquetas de hablante, agrupando consecutivos."""
    lineas = []
    actual_spk = None
    buffer_texto = []
    for s in segs_con_hablante:
        if s["speaker"] != actual_spk:
            if buffer_texto:
                lineas.append(f"[{actual_spk}]: {' '.join(buffer_texto).strip()}")
            actual_spk = s["speaker"]
            buffer_texto = [s["text"]]
        else:
            buffer_texto.append(s["text"])
    if buffer_texto:
        lineas.append(f"[{actual_spk}]: {' '.join(buffer_texto).strip()}")
    return "\n\n".join(lineas)


def extraer_muestras_hablantes(segs_con_hablante, min_segundos=6.0,
                                max_segundos=15.0, max_chars=180):
    """Para cada hablante, devuelve {start, end, snippet} de una muestra
    suficientemente larga (entre min_segundos y max_segundos) para reconocer
    su voz. Acumula segmentos consecutivos del mismo hablante hasta llegar
    al mínimo, y si su primer turno es muy corto busca otro más largo."""
    muestras = {}

    # Agrupar todos los segmentos por hablante en orden
    por_hablante = {}
    for s in segs_con_hablante:
        por_hablante.setdefault(s["speaker"], []).append(s)

    for spk, segs in por_hablante.items():
        # Buscar el primer "bloque" de >= min_segundos acumulando segmentos
        # consecutivos en el tiempo (gap < 1.5 s entre ellos).
        mejor = None
        i = 0
        while i < len(segs):
            inicio = segs[i]
            acumulado_textos = [inicio["text"].strip()]
            fin = float(inicio["end"])
            j = i + 1
            while j < len(segs):
                gap = float(segs[j]["start"]) - fin
                duracion_actual = fin - float(inicio["start"])
                if gap > 1.5 or duracion_actual >= max_segundos:
                    break
                acumulado_textos.append(segs[j]["text"].strip())
                fin = float(segs[j]["end"])
                j += 1

            duracion = fin - float(inicio["start"])
            if duracion >= min_segundos:
                mejor = (inicio, fin, acumulado_textos)
                break
            # Guardamos como fallback el bloque más largo visto hasta ahora
            if mejor is None or duracion > (mejor[1] - float(mejor[0]["start"])):
                mejor = (inicio, fin, acumulado_textos)
            i = j if j > i else i + 1

        if mejor is None:
            continue
        inicio, fin, textos = mejor
        snippet = " ".join(t for t in textos if t).strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."
        muestras[spk] = {
            "start": float(inicio["start"]),
            "end": float(fin),
            "snippet": snippet,
        }
    return muestras

# Almacena el job_id actual por hilo para que el tqdm interceptado sepa cuál actualizar
_current_job = threading.local()


class WhisperProgressTqdm(_OriginalTqdm):
    """tqdm interceptado: actualiza el progreso del job mientras Whisper transcribe."""
    def update(self, n=1):
        super().update(n)
        job_id = getattr(_current_job, "job_id", None)
        if job_id and self.total:
            # Espejamos el % real de Whisper, reservando 95-100 para guardado/diarización
            pct = int((self.n / self.total) * 95)
            if job_id in jobs and jobs[job_id].get("progreso", 0) < pct:
                jobs[job_id]["progreso"] = min(pct, 95)


# Whisper usa `tqdm.tqdm(...)` en su módulo transcribe. Inyectamos un módulo proxy.
_fake_tqdm_module = types.ModuleType("tqdm")
_fake_tqdm_module.tqdm = WhisperProgressTqdm
sys.modules["whisper.transcribe"].tqdm = _fake_tqdm_module

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2 GB máximo

# Configuración
CARPETA_AUDIOS = Path("audios")
CARPETA_TRANSCRIPCIONES = Path("transcripciones")
EXTENSIONES_PERMITIDAS = {
    "mp3", "wav", "m4a", "ogg", "flac",
    "mp4", "avi", "mkv", "webm", "wma"
}

# Crear carpetas si no existen
CARPETA_AUDIOS.mkdir(exist_ok=True)
CARPETA_TRANSCRIPCIONES.mkdir(exist_ok=True)

# Diccionario global de jobs: job_id -> { estado, progreso, texto, idioma, error }
jobs = {}


def extension_permitida(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in EXTENSIONES_PERMITIDAS


def obtener_duracion(ruta_audio: str) -> float:
    """Obtiene la duración del audio en segundos usando ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(ruta_audio)],
            capture_output=True, text=True, timeout=30
        )
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception:
        return 0.0


def procesar_audio(job_id: str, ruta_audio: str, modelo_nombre: str,
                   idioma, nombre_archivo_original: str, diarizar: bool = False,
                   num_hablantes=None):
    """Transcribe el audio en un hilo separado y actualiza el progreso."""
    try:
        tiempo_inicio = time.time()
        jobs[job_id]["estado"] = "cargando_modelo"
        jobs[job_id]["progreso"] = 1

        # Obtener duración del audio (informativa)
        duracion_total = obtener_duracion(ruta_audio)
        jobs[job_id]["duracion"] = duracion_total

        # Cargar modelo
        modelo = whisper.load_model(modelo_nombre)
        jobs[job_id]["estado"] = "transcribiendo"

        # Asociar este hilo con el job para que el tqdm interceptado actualice progreso
        _current_job.job_id = job_id

        opciones = {"verbose": False}
        if idioma:
            opciones["language"] = idioma

        resultado = modelo.transcribe(str(ruta_audio), **opciones)

        texto = resultado["text"].strip()
        idioma_detectado = resultado.get("language", "desconocido")
        texto_final = texto

        if diarizar:
            if not HF_TOKEN:
                raise RuntimeError(
                    "Falta HF_TOKEN en el archivo .env para usar diarización."
                )

            # === Protección 1: validar número de segmentos ===
            num_segs = len(resultado.get("segments", []))
            if num_segs > MAX_SEGMENTOS_DIARIZACION:
                raise RuntimeError(
                    f"El audio generó {num_segs} segmentos, demasiados para diarizar "
                    f"de forma segura (máx {MAX_SEGMENTOS_DIARIZACION}). "
                    f"Usa un modelo más pequeño (medium/small) o divide el audio en partes."
                )

            # === Protección 2: liberar el modelo de Whisper antes de pyannote ===
            # Whisper en `large` ocupa ~3 GB de VRAM/RAM. Liberarlo evita que se
            # acumule con pyannote y dispare la memoria.
            del modelo
            try:
                import torch as _torch_clean
                if _torch_clean.cuda.is_available():
                    _torch_clean.cuda.empty_cache()
                import gc
                gc.collect()
            except Exception:
                pass

            jobs[job_id]["estado"] = "diarizando"
            jobs[job_id]["progreso"] = 96
            pipeline = get_diarization_pipeline()

            # Convertir a WAV mono 16kHz con FFmpeg (formato que soundfile lee
            # sin depender de torchcodec, que falla con Python 3.14 + Torch nuevo)
            import soundfile as sf
            import torch as _torch
            ruta_wav = str(Path(ruta_audio).with_suffix(".diar.wav"))
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(ruta_audio),
                 "-ac", "1", "-ar", "16000", "-vn", ruta_wav],
                capture_output=True, check=True
            )
            waveform_np, sample_rate = sf.read(ruta_wav, always_2d=True)
            waveform = _torch.from_numpy(waveform_np.T).float()  # (channels, time)

            # === Protección 3: kwargs del pipeline con cap de hablantes ===
            kwargs_pipeline = {}
            if num_hablantes:
                kwargs_pipeline["num_speakers"] = int(num_hablantes)
            elif duracion_total and duracion_total / 60 > DURACION_AUDIO_LARGO_MIN:
                # Audio largo sin pista de cuántos hablantes: limitamos para que
                # el clustering no explore configuraciones extremas.
                kwargs_pipeline["max_speakers"] = DEFAULT_MAX_SPEAKERS_AUDIO_LARGO
                print(f"[diarización] Audio >{DURACION_AUDIO_LARGO_MIN}min sin "
                      f"num_hablantes — fijando max_speakers={DEFAULT_MAX_SPEAKERS_AUDIO_LARGO}")

            diarizacion = pipeline(
                {"waveform": waveform, "sample_rate": sample_rate},
                **kwargs_pipeline
            )
            if Path(ruta_wav).exists():
                os.remove(ruta_wav)
            segs_con_hablante = asignar_hablantes(resultado["segments"], diarizacion)
            texto_final = formatear_diarizacion(segs_con_hablante)
            jobs[job_id]["muestras_hablantes"] = extraer_muestras_hablantes(segs_con_hablante)

        jobs[job_id]["progreso"] = 98
        jobs[job_id]["estado"] = "guardando"

        # Guardar transcripción
        nombre_base = Path(nombre_archivo_original).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_txt = f"{nombre_base}_{timestamp}.txt"
        ruta_txt = CARPETA_TRANSCRIPCIONES / nombre_txt

        with open(ruta_txt, "w", encoding="utf-8") as f:
            f.write(f"Archivo: {nombre_archivo_original}\n")
            f.write(f"Modelo: {modelo_nombre}\n")
            f.write(f"Idioma detectado: {idioma_detectado}\n")
            f.write(f"Diarización: {'sí' if diarizar else 'no'}\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 50 + "\n\n")
            f.write(texto_final)

        # Si hubo diarización, conservamos el audio para poder reproducir
        # fragmentos de cada hablante. Si no, lo borramos como antes.
        if not diarizar and Path(ruta_audio).exists():
            os.remove(ruta_audio)

        segundos = round(time.time() - tiempo_inicio)
        jobs[job_id].update({
            "estado": "completado",
            "progreso": 100,
            "texto": texto_final,
            "idioma": idioma_detectado,
            "archivo_guardado": nombre_txt,
            "tiempo_segundos": segundos,
            "ruta_audio": str(ruta_audio) if diarizar else None,
        })

    except Exception as e:
        if Path(ruta_audio).exists():
            os.remove(ruta_audio)
        jobs[job_id].update({
            "estado": "error",
            "progreso": 0,
            "error": str(e)
        })


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/transcribir", methods=["POST"])
def transcribir():
    if "audio" not in request.files:
        return jsonify({"error": "No se encontró ningún archivo de audio."}), 400

    archivo = request.files["audio"]
    if archivo.filename == "":
        return jsonify({"error": "No se seleccionó ningún archivo."}), 400

    if not extension_permitida(archivo.filename):
        return jsonify({"error": f"Formato no soportado. Usa: {', '.join(EXTENSIONES_PERMITIDAS)}"}), 400

    modelo_nombre = request.form.get("modelo", "base")
    idioma = request.form.get("idioma", "") or None
    diarizar = request.form.get("diarizar", "").lower() in ("1", "true", "on", "yes")
    try:
        num_hablantes = int(request.form.get("num_hablantes", "") or 0) or None
    except ValueError:
        num_hablantes = None

    # Guardar archivo
    extension = archivo.filename.rsplit(".", 1)[1].lower()
    nombre_unico = f"{uuid.uuid4().hex}.{extension}"
    ruta_audio = CARPETA_AUDIOS / nombre_unico
    archivo.save(str(ruta_audio))

    # Crear job
    job_id = uuid.uuid4().hex
    jobs[job_id] = {
        "estado": "iniciando",
        "progreso": 0,
        "texto": None,
        "idioma": None,
        "error": None,
        "archivo_guardado": None
    }

    # Lanzar hilo de transcripción
    hilo = threading.Thread(
        target=procesar_audio,
        args=(job_id, str(ruta_audio), modelo_nombre, idioma,
              archivo.filename, diarizar, num_hablantes),
        daemon=True
    )
    hilo.start()

    return jsonify({"job_id": job_id})


@app.route("/progreso/<job_id>")
def progreso(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job no encontrado."}), 404
    return jsonify(job)


@app.route("/audio/<job_id>")
def audio(job_id):
    """Sirve el audio original del job (solo si se hizo diarización)."""
    job = jobs.get(job_id)
    if not job or not job.get("ruta_audio"):
        return jsonify({"error": "Audio no disponible."}), 404
    ruta = Path(job["ruta_audio"])
    if not ruta.exists():
        return jsonify({"error": "Archivo de audio no encontrado."}), 404
    return send_file(str(ruta), conditional=True)


@app.route("/renombrar/<job_id>", methods=["POST"])
def renombrar(job_id):
    """Renombra los SPEAKER_XX en el texto y reescribe el .txt guardado."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job no encontrado."}), 404
    if job.get("estado") != "completado" or not job.get("texto"):
        return jsonify({"error": "El job aún no tiene texto disponible."}), 400

    data = request.get_json(silent=True) or {}
    mapeo = data.get("mapeo") or {}
    if not isinstance(mapeo, dict) or not mapeo:
        return jsonify({"error": "Mapeo inválido."}), 400

    texto = job["texto"]
    # Reemplazar de los nombres más largos a los más cortos para evitar colisiones
    for original in sorted(mapeo.keys(), key=len, reverse=True):
        nuevo = (mapeo[original] or "").strip()
        if not nuevo:
            continue
        texto = texto.replace(f"[{original}]", f"[{nuevo}]")

    job["texto"] = texto

    # Reescribir el archivo guardado conservando la cabecera
    nombre_txt = job.get("archivo_guardado")
    if nombre_txt:
        ruta_txt = CARPETA_TRANSCRIPCIONES / nombre_txt
        if ruta_txt.exists():
            try:
                contenido = ruta_txt.read_text(encoding="utf-8")
                separador = "-" * 50 + "\n\n"
                if separador in contenido:
                    cabecera, _ = contenido.split(separador, 1)
                    ruta_txt.write_text(
                        cabecera + separador + texto, encoding="utf-8"
                    )
                else:
                    ruta_txt.write_text(texto, encoding="utf-8")
            except Exception as e:
                return jsonify({"error": f"No se pudo actualizar el archivo: {e}"}), 500

    # Ya identificaste a los hablantes: el audio temporal ya no se necesita.
    ruta_audio = job.get("ruta_audio")
    if ruta_audio and Path(ruta_audio).exists():
        try:
            os.remove(ruta_audio)
        except Exception:
            pass
    job["ruta_audio"] = None

    return jsonify({"texto": texto, "archivo_guardado": nombre_txt})


if __name__ == "__main__":
    print("=" * 50)
    print("  🎙️  WHISPER TRANSCRIPTOR - INTERFAZ WEB")
    print("=" * 50)
    print("Abre tu navegador en: http://localhost:5200")
    print("Presiona Ctrl+C para detener el servidor.")
    print("=" * 50)
    import threading, webbrowser
    threading.Timer(1.5, lambda: webbrowser.open("http://localhost:5200")).start()
    app.run(debug=False, host="0.0.0.0", port=5200, threaded=True)
