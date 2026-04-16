#!/usr/bin/env python3
"""
Transcriptor de Audio con Whisper - Script CLI
Uso: python transcriber.py <archivo_audio> [--modelo large] [--idioma es]
"""

import argparse
import os
import sys
import whisper
from pathlib import Path
from datetime import datetime


def transcribir(ruta_audio: str, modelo_nombre: str = "large", idioma: str = None) -> str:
    """
    Transcribe un archivo de audio usando Whisper.

    Args:
        ruta_audio: Ruta al archivo de audio
        modelo_nombre: Nombre del modelo Whisper (tiny, large, small, medium, large)
        idioma: Código de idioma opcional (ej: 'es', 'en'). None = detección automática

    Returns:
        Texto transcrito
    """
    # Verificar que el archivo existe
    if not os.path.exists(ruta_audio):
        print(f"❌ Error: No se encontró el archivo '{ruta_audio}'")
        sys.exit(1)

    print(f"🔄 Cargando modelo '{modelo_nombre}'...")
    modelo = whisper.load_model(modelo_nombre)

    print(f"🎙️  Transcribiendo: {ruta_audio}")
    if idioma:
        print(f"🌍 Idioma: {idioma}")
        resultado = modelo.transcribe(ruta_audio, language=idioma)
    else:
        print("🌍 Idioma: detección automática")
        resultado = modelo.transcribe(ruta_audio)

    texto = resultado["text"].strip()
    idioma_detectado = resultado.get("language", "desconocido")
    print(f"✅ Idioma detectado: {idioma_detectado}")

    return texto


def guardar_transcripcion(texto: str, ruta_audio: str) -> str:
    """
    Guarda la transcripción en la carpeta 'transcripciones/'.

    Returns:
        Ruta del archivo guardado
    """
    # Crear carpeta si no existe
    carpeta = Path("transcripciones")
    carpeta.mkdir(exist_ok=True)

    # Nombre del archivo de salida
    nombre_large = Path(ruta_audio).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_salida = carpeta / f"{nombre_large}_{timestamp}.txt"

    with open(nombre_salida, "w", encoding="utf-8") as f:
        f.write(texto)

    return str(nombre_salida)


def main():
    parser = argparse.ArgumentParser(
        description="🎙️  Transcriptor de Audio con Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python transcriber.py audio.mp3
  python transcriber.py audio.wav --modelo small
  python transcriber.py audio.m4a --modelo medium --idioma es
        """
    )

    parser.add_argument(
        "audio",
        help="Ruta al archivo de audio a transcribir"
    )
    parser.add_argument(
        "--modelo", "-m",
        default="large",
        choices=["tiny", "large", "small", "medium", "large"],
        help="Modelo Whisper a usar (default: large)"
    )
    parser.add_argument(
        "--idioma", "-i",
        default=None,
        help="Código de idioma (ej: es, en, fr). Por defecto detecta automáticamente."
    )
    parser.add_argument(
        "--guardar", "-g",
        action="store_true",
        default=True,
        help="Guardar transcripción en archivo .txt (activado por defecto)"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("  🎙️  TRANSCRIPTOR DE AUDIO - WHISPER")
    print("=" * 50)

    # Transcribir
    texto = transcribir(args.audio, args.modelo, args.idioma)

    # Mostrar resultado
    print("\n" + "=" * 50)
    print("📄 TRANSCRIPCIÓN:")
    print("=" * 50)
    print(texto)
    print("=" * 50)

    # Guardar en archivo
    if args.guardar:
        ruta_guardada = guardar_transcripcion(texto, args.audio)
        print(f"\n💾 Transcripción guardada en: {ruta_guardada}")

    return texto


if __name__ == "__main__":
    main()
