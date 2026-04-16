#!/usr/bin/env python3
"""
Transcriptor de Audio con Whisper - Interfaz Web Flask
Ejecutar: python app.py
Luego abrir: http://localhost:5200
"""
import time

import os
import uuid
import threading
import whisper
import subprocess
import json
from flask import Flask, request, render_template, jsonify
from pathlib import Path
from datetime import datetime

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


# Velocidad aproximada de cada modelo (segundos de audio por segundo de proceso)
VELOCIDAD_MODELO = {
    'tiny': 12, 'base': 7, 'small': 4, 'medium': 2, 'large': 1
}


def simular_progreso(job_id: str, duracion_audio: float, modelo_nombre: str):
    """Hilo que avanza el progreso suavemente de 10% a 90% mientras transcribe."""
    factor = VELOCIDAD_MODELO.get(modelo_nombre, 5)
    tiempo_estimado = max(duracion_audio / factor, 8)  # mínimo 8 segundos
    inicio = time.time()

    while True:
        estado = jobs.get(job_id, {}).get('estado', 'error')
        if estado in ('completado', 'error', 'guardando'):
            break
        elapsed = time.time() - inicio
        # Avanza de 10% a 90% durante el tiempo estimado (curva suave)
        ratio = min(elapsed / tiempo_estimado, 1.0)
        pct = int(10 + ratio * 80)
        if jobs[job_id]['progreso'] < pct:
            jobs[job_id]['progreso'] = pct
        time.sleep(0.4)


def procesar_audio(job_id: str, ruta_audio: str, modelo_nombre: str,
                   idioma, nombre_archivo_original: str):
    """Transcribe el audio en un hilo separado y actualiza el progreso."""
    try:
        jobs[job_id]["estado"] = "cargando_modelo"
        jobs[job_id]["progreso"] = 5

        # Obtener duración del audio para estimar progreso
        duracion_total = obtener_duracion(ruta_audio)
        jobs[job_id]["duracion"] = duracion_total

        # Cargar modelo
        modelo = whisper.load_model(modelo_nombre)
        jobs[job_id]["progreso"] = 10
        jobs[job_id]["estado"] = "transcribiendo"

        # Lanzar hilo de simulación de progreso
        hilo_prog = threading.Thread(
            target=simular_progreso,
            args=(job_id, duracion_total, modelo_nombre),
            daemon=True
        )
        hilo_prog.start()

        opciones = {"verbose": False}
        if idioma:
            opciones["language"] = idioma

        resultado = modelo.transcribe(str(ruta_audio), **opciones)

        jobs[job_id]["progreso"] = 95
        jobs[job_id]["estado"] = "guardando"

        texto = resultado["text"].strip()
        idioma_detectado = resultado.get("language", "desconocido")

        # Guardar transcripción
        nombre_base = Path(nombre_archivo_original).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_txt = f"{nombre_base}_{timestamp}.txt"
        ruta_txt = CARPETA_TRANSCRIPCIONES / nombre_txt

        with open(ruta_txt, "w", encoding="utf-8") as f:
            f.write(f"Archivo: {nombre_archivo_original}\n")
            f.write(f"Modelo: {modelo_nombre}\n")
            f.write(f"Idioma detectado: {idioma_detectado}\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 50 + "\n\n")
            f.write(texto)

        # Eliminar audio temporal
        if Path(ruta_audio).exists():
            os.remove(ruta_audio)

        jobs[job_id].update({
            "estado": "completado",
            "progreso": 100,
            "texto": texto,
            "idioma": idioma_detectado,
            "archivo_guardado": nombre_txt
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
        args=(job_id, str(ruta_audio), modelo_nombre, idioma, archivo.filename),
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


if __name__ == "__main__":
    print("=" * 50)
    print("  🎙️  WHISPER TRANSCRIPTOR - INTERFAZ WEB")
    print("=" * 50)
    print("Abre tu navegador en: http://localhost:5200")
    print("Presiona Ctrl+C para detener el servidor.")
    print("=" * 50)
    app.run(debug=False, host="0.0.0.0", port=5200, threaded=True)
