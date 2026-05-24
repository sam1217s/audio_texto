# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Comandos de desarrollo

```bash
# Aplicación básica (sin reconocimiento de voz)
python app.py          # http://localhost:5200

# Aplicación con biblioteca de voces persistente
python appmodel.py     # http://localhost:5202

# Ambas pueden correr en paralelo sin conflicto (puertos distintos)

# Graphify — análisis de arquitectura
python graph.py mapa       # árbol de archivos con sus paquetes importados
python graph.py cerebro    # vista por capas + hubs + módulos de dominio (inyectado vía hook)
python graph.py detective  # endpoints, paquetes exclusivos, variables de entorno
```

## Arquitectura general

Proyecto monorepo plano (sin subcarpetas de módulos). Un solo nivel de código Python.

```
audio_texto/
├── app.py                  # Flask app básica — puerto 5200
├── appmodel.py             # Flask app extendida con biblioteca de voces — puerto 5202
├── transcriber.py          # Script CLI standalone (sin Flask)
├── templates/
│   ├── index.html          # UI de app.py
│   └── index_modelvoz.html # UI de appmodel.py
├── audios/                 # Archivos de audio temporales (se limpian por job)
├── transcripciones/        # .txt y .pdf de salida (persistentes)
├── voces/
│   └── voces.json          # Biblioteca de embeddings de voz (solo appmodel.py)
└── .env                    # HF_TOKEN para pyannote (requerido para diarización)
```

## Backend

### Flujo de request (transcripción)

```
POST /transcribir
  → valida extensión (mp3/wav/m4a/ogg/flac/mp4/avi/mkv/webm/wma)
  → guarda audio en audios/{uuid}.{ext}
  → crea jobs[job_id] = {estado, progreso, texto, ...}
  → lanza threading.Thread(target=procesar_audio, ...)
  → devuelve {job_id}

GET /progreso/{job_id}
  → devuelve jobs[job_id] (polling desde el frontend)

procesar_audio() [hilo]:
  → estado: "cargando_modelo"
  → whisper.load_model(modelo)
  → estado: "transcribiendo"
  → modelo.transcribe() — progreso 0→95% vía WhisperProgressTqdm
  → [si diarizar]:
      → del modelo + cuda.empty_cache()  # libera VRAM antes de pyannote
      → estado: "diarizando" (96%)
      → ffmpeg convierte a WAV mono 16kHz (soundfile lee mejor que torchcodec)
      → pipeline(waveform, sample_rate)
      → asignar_hablantes() + formatear_diarizacion()
      → [solo appmodel.py, si nombres_esperados]:
          → estado: "identificando_voces" (97%)
          → asignar_nombres_por_voz() → matching con biblioteca
  → estado: "guardando" (98%)
  → escribe transcripciones/{stem}_{timestamp}.txt
  → estado: "completado" (100%)
```

### Protecciones críticas de memoria/RAM

- `MAX_SEGMENTOS_DIARIZACION = 3500`: si Whisper genera más segmentos que este límite, la diarización se aborta. pyannote construye una matriz O(n²) que revienta la RAM con audios muy largos.
- El modelo Whisper se elimina explícitamente (`del modelo` + `cuda.empty_cache()`) antes de cargar pyannote. Whisper large ocupa ~3 GB de VRAM; acumularlo con pyannote causa OOM.
- `DEFAULT_MAX_SPEAKERS_AUDIO_LARGO = 6`: para audios >30 min sin `num_hablantes`, se fija `max_speakers=6` para acotar el clustering.

### Progreso de Whisper (hack de tqdm)

Whisper usa `tqdm.tqdm` internamente en su módulo `transcribe`. Se inyecta un módulo proxy antes de que Whisper lo importe:

```python
_fake_tqdm_module = types.ModuleType("tqdm")
_fake_tqdm_module.tqdm = WhisperProgressTqdm
sys.modules["whisper.transcribe"].tqdm = _fake_tqdm_module
```

`WhisperProgressTqdm` mapea el progreso real de Whisper a `jobs[job_id]["progreso"]` en el rango 0–95%. El 96–100% está reservado para diarización/guardado.

### Estado de un job

```python
{
  "estado": "iniciando|cargando_modelo|transcribiendo|diarizando|identificando_voces|guardando|completado|error",
  "progreso": 0-100,
  "texto": str | None,
  "idioma": str | None,
  "error": str | None,
  "archivo_guardado": "stem_YYYYMMDD_HHMMSS.txt" | None,
  "duracion": float,           # segundos del audio
  "segmentos": [...],          # lista de {start, end, speaker, text}
  "muestras_hablantes": {...}, # {SPEAKER_XX: {start, end, snippet}} — para UI
  "tiempo_segundos": int,
  "ruta_audio": str,           # conservado para reproductor karaoke
  # solo appmodel.py:
  "mapeo_auto": {"SPEAKER_00": "Juan", ...}
}
```

El audio se conserva en `audios/` hasta que el usuario pulsa "Nueva transcripción" (`POST /eliminar/{job_id}`). No hay limpieza automática al completar.

### Conversión de audio (decisión de diseño)

pyannote 4.x con torchaudio puede fallar con Python 3.14+ (torchcodec roto). Por eso **siempre** se convierte el audio a WAV mono 16kHz con FFmpeg antes de pasarlo a soundfile, en lugar de dejarlo leer a torchaudio directamente.

### Endpoints de app.py (puerto 5200)

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Sirve index.html |
| POST | `/transcribir` | Inicia job de transcripción |
| GET | `/progreso/<job_id>` | Polling del estado del job |
| GET | `/audio/<job_id>` | Sirve el audio para el reproductor |
| POST | `/renombrar/<job_id>` | Renombra SPEAKER_XX → nombre real en texto y .txt |
| POST | `/eliminar/<job_id>` | Limpia el audio del job |
| GET | `/descargar/<job_id>/<formato>` | Descarga TXT o PDF |

### Endpoints adicionales de appmodel.py (puerto 5202)

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/voces` | Lista nombres registrados (sin embeddings) |
| POST | `/voces/registrar` | Registra nueva voz (nombre + audio) |
| DELETE | `/voces/<nombre>` | Elimina voz de la biblioteca |
| POST | `/voces/registrar_desde_job/<job_id>` | Registra voz desde un SPEAKER_XX de un job ya completado |

### Sistema de reconocimiento de voz (solo appmodel.py)

**Biblioteca**: `voces/voces.json` → `{nombre: [embedding_float, ...]}`

**Flujo de registro**: audio → WAV mono 16kHz (ffmpeg) → `pyannote/embedding` model → vector 192D → guardado en JSON.

**Flujo de matching**:
1. Por cada SPEAKER_XX detectado, extrae un bloque contiguo de hasta 30s de audio.
2. Calcula embedding con `pyannote/embedding`.
3. Construye matriz de similitud coseno (hablantes × voces candidatas).
4. Asignación óptima con `scipy.optimize.linear_sum_assignment` (algoritmo húngaro). Fallback greedy si scipy falla.
5. Solo asigna si similitud ≥ `UMBRAL_SIMILITUD_VOZ = 0.55`.

El usuario indica en el formulario qué voces espera en el audio (`nombres_esperados`). Solo esas voces se comparan, no toda la biblioteca.

### Modelos disponibles en UI

Whisper: `tiny`, `base`, `small`, `medium`, `large`, `turbo`
pyannote diarización: `pyannote/speaker-diarization-community-1` (preferido) o `3.1` (fallback público).

### Variable de entorno requerida

```
HF_TOKEN=hf_xxx   # Token de Hugging Face — requerido para cargar pyannote
```

Sin este token, la diarización falla con RuntimeError explícito.

## Frontend (templates)

### index.html (app.py)
- Subida de archivo + selección de modelo/idioma/diarización
- Polling de `/progreso/{job_id}` con barra de progreso
- Reproductor de audio con seguimiento de segmentos (karaoke)
- Panel de muestras de hablantes para renombrado manual
- Descarga TXT/PDF

### index_modelvoz.html (appmodel.py)
- Todo lo de index.html más:
- Selector de voces esperadas (carga `/voces` al iniciar)
- Reconocimiento automático de hablantes (muestra `mapeo_auto`)
- Botón para registrar nueva voz desde la UI
- Registro de voz desde un SPEAKER_XX de un job existente

## Convenciones importantes

**FFmpeg en PATH**: app.py y appmodel.py agregan el path de FFmpeg instalado vía WinGet al inicio. Si FFmpeg no está en esa ruta, falla silenciosamente y luego subprocess falla al convertir audio. El path está hardcodeado en `_FFMPEG_BIN`.

**TF32 activado en startup**: Para GPUs NVIDIA Ampere/Ada/Blackwell, se habilita TF32 (`matmul.allow_tf32`, `cudnn.allow_tf32`, `cudnn.benchmark`). Ganancia 2-3x en matmul fp32. No se puede desactivar sin editar el código.

**Jobs en memoria, sin persistencia**: `jobs = {}` es un dict global en RAM. Al reiniciar el servidor, todos los jobs se pierden. Los archivos `.txt` en `transcripciones/` son la única persistencia real.

**Diarización lazy load**: El pipeline de pyannote se carga la primera vez que se usa (no al iniciar el servidor), protegido con `threading.Lock()`. Igual para el modelo de embeddings. Esto evita consumir VRAM/RAM si nunca se usa diarización.

**Renombrado seguro**: Al renombrar hablantes, se reemplaza de los nombres más largos a los más cortos para evitar colisiones (ej. `SPEAKER_100` antes que `SPEAKER_1`).

**Límite de upload**: `MAX_CONTENT_LENGTH = 2 GB`. Flask no stream-procesa; el archivo se guarda completo en disco antes de transcribir.

**Archivos temporales de matching**: Durante el matching de voces en appmodel.py se crean archivos `audios/_match_SPEAKER_XX_*.wav` y `audios/_voz_*.wav`. Se borran en el bloque `finally` incluso si falla el embedding.

## Sistema graphify — arquitectura en tiempo real

CLAUDE.md (este archivo) y graphify son complementarios:

| | CLAUDE.md | graph.py cerebro |
|---|---|---|
| **Naturaleza** | Estático — escrito a mano, versionado en git | Dinámico — generado desde el código en tiempo real |
| **Contenido** | Convenciones, reglas de negocio, flujos, decisiones | Paquetes por capa, hubs, módulos de dominio, endpoints |
| **Actualización** | Manual | Automática tras cada edición vía hook PostToolUse |
| **Cuándo se lee** | Al inicio de cada sesión (system prompt de Claude Code) | Al inicio (SessionStart) y tras cada edición (PostToolUse) |

Al abrir Claude Code, recibe ambos automáticamente:
1. Este CLAUDE.md — entiende las reglas y convenciones del proyecto
2. El output de `graph.py cerebro` vía hook — entiende el estado actual de la arquitectura
