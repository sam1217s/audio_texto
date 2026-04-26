# Whisper Audio Transcriptor

Herramienta para transcribir archivos de audio y video a texto usando [OpenAI Whisper](https://github.com/openai/whisper). Disponible como interfaz web (Flask) y como script de línea de comandos. Incluye **detección de hablantes** opcional con [pyannote.audio](https://github.com/pyannote/pyannote-audio).

## Características

- **Interfaz web** con barra de progreso en tiempo real
- **CLI** para transcribir desde la terminal
- Procesamiento asíncrono: la UI no se bloquea mientras transcribe
- **Detección de hablantes (diarización)** opcional con etiquetas `[SPEAKER_XX]`
- **Renombrado de hablantes** desde la interfaz (asigna nombres reales a cada voz detectada)
- Detección automática de idioma
- Modal con tiempo total al finalizar
- Apertura automática del navegador al iniciar el servidor
- Transcripciones guardadas automáticamente en `transcripciones/`
- Soporta archivos de hasta **2 GB**
- Compatible con múltiples formatos de audio y video
- **Funciona 100% offline** una vez descargados los modelos (la diarización requiere descarga inicial autenticada en HuggingFace)

## Requisitos previos

- Python 3.8+
- [FFmpeg](https://ffmpeg.org/download.html) instalado y en el PATH del sistema
- Cuenta gratuita en [HuggingFace](https://huggingface.co/) (solo si quieres usar la diarización)

## Instalación

```bash
python -m pip install -r requirements.txt
```

## Configuración de GPU (recomendado)

Por defecto Whisper puede correr en CPU, pero es significativamente más lento. Si tienes una GPU NVIDIA, configurarla acelera la transcripción varias veces.

### 1. Verificar si tienes GPU NVIDIA

```powershell
powershell "Get-WmiObject Win32_VideoController | Select-Object Name"
```

Si aparece una GPU NVIDIA (GeForce, RTX, GTX), continúa con el siguiente paso.

### 2. Verificar drivers NVIDIA y versión de CUDA

```bash
nvidia-smi
```

Busca la línea `CUDA Version: X.X` en la esquina superior derecha. Eso indica qué versión de CUDA tienes instalada.

### 3. Verificar si PyTorch detecta la GPU

```bash
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No detectada')"
```

- Si dice `CUDA disponible: True` — todo listo, Whisper usará la GPU automáticamente.
- Si dice `CUDA disponible: False` — PyTorch está instalado sin soporte CUDA. Ver siguiente paso.

### 4. Instalar PyTorch con soporte CUDA

Si CUDA no está disponible, reinstala PyTorch forzando la versión con CUDA. Usa el comando según tu versión de CUDA (vista en `nvidia-smi`):

**CUDA 12.8:**
```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

**CUDA 12.4:**
```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

**CUDA 12.1:**
```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

> El flag `--force-reinstall` es necesario porque pip puede detectar torch ya instalado y saltarse la reinstalación aunque sea la versión CPU.

Luego vuelve a verificar con el comando del paso 3.

### ¿Por qué importa la GPU?

| Modo     | Velocidad aproximada (modelo large) |
|----------|-------------------------------------|
| CPU      | Muy lento (puede tardar horas)      |
| GPU NVIDIA | Rápido (minutos)                  |

Whisper también mostrará el aviso `FP16 is not supported on CPU; using FP32 instead` si está corriendo en CPU — esto es normal y no es un error, solo indica que no hay GPU disponible.

## Detección de hablantes (diarización)

Es **opcional**. Si la activas, después de transcribir el sistema identifica cuántas personas distintas hablan y etiqueta cada turno como `[SPEAKER_00]`, `[SPEAKER_01]`, etc. Luego puedes renombrar cada `SPEAKER_XX` a su nombre real desde la interfaz.

### 1. Crear cuenta en HuggingFace y aceptar términos de los modelos

1. Crea una cuenta gratuita en https://huggingface.co/
2. Acepta los términos de uso de **los dos modelos** que necesita pyannote (basta con pulsar "Agree and access repository" en cada uno):
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-community-1
3. Genera un token de acceso en https://huggingface.co/settings/tokens (tipo "Read").

### 2. Crear archivo `.env`

En la raíz del proyecto, crea un archivo llamado `.env` con tu token:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> ⚠️ El archivo `.env` está en `.gitignore` y no se sube al repositorio. **Nunca compartas tu token públicamente** — si se filtra, revócalo en https://huggingface.co/settings/tokens y genera uno nuevo.

### 3. Usar la diarización

En la interfaz web, activa el toggle **"👥 Detectar hablantes"** antes de pulsar "Transcribir Audio". Al terminar verás:

```
[SPEAKER_00]: Buenos días equipo, ¿ya enviaron el reporte?

[SPEAKER_01]: Sí, lo mandé anoche.
```

Y un panel debajo del resultado con un input por cada hablante para asignarle un nombre real. Tras pulsar "Aplicar nombres", el texto en pantalla **y el archivo `.txt` guardado** quedan actualizados:

```
[Juan]: Buenos días equipo, ¿ya enviaron el reporte?

[María]: Sí, lo mandé anoche.
```

> La primera vez que actives la diarización se descargan ~500 MB de modelos. Después queda en caché y se usa offline.

Para más detalles sobre cómo funciona y otras opciones (reconocimiento automático por voz), ver [hablantes.md](hablantes.md).

## Uso

### Interfaz web

```bash
python app.py
```

El navegador se abre automáticamente en **http://localhost:5200**. Sube tu archivo, opcionalmente activa "Detectar hablantes", y obtén la transcripción.

### Línea de comandos (CLI)

```bash
# Transcripción básica
python transcriber.py mi_audio.mp3

# Especificar modelo
python transcriber.py mi_audio.mp3 --modelo large

# Especificar idioma (evita la detección automática)
python transcriber.py mi_audio.mp3 --idioma es
```

> El CLI no soporta diarización — solo la interfaz web.

El texto transcrito se guarda en `transcripciones/` como archivo `.txt` con timestamp.

## Modelos disponibles

| Modelo | Velocidad   | Precisión | VRAM aproximada |
|--------|-------------|-----------|-----------------|
| tiny   | Muy rápido  | Básica    | ~1 GB           |
| base   | Rápido      | Buena     | ~1 GB           |
| small  | Moderado    | Muy buena | ~2 GB           |
| medium | Lento       | Excelente | ~5 GB           |
| large  | Muy lento   | Máxima    | ~10 GB          |

> Para la mayoría de los casos se recomienda `base` o `small`.

Los modelos se descargan automáticamente la primera vez que se usan. La descarga puede tardar varios minutos dependiendo del tamaño del modelo y la velocidad de conexión. Una vez descargados, el sistema funciona **sin internet**.

## Formatos soportados

`mp3` `wav` `m4a` `ogg` `flac` `mp4` `avi` `mkv` `webm` `wma`

## Estructura del proyecto

```
audio_texto/
├── app.py               # Servidor web Flask + diarización
├── transcriber.py       # Script CLI
├── requirements.txt     # Dependencias
├── hablantes.md         # Documentación de diarización
├── .env                 # HF_TOKEN para pyannote (no versionado)
├── templates/
│   └── index.html       # Interfaz web
├── audios/              # Audios subidos temporalmente (auto-generado)
└── transcripciones/     # Transcripciones generadas (auto-generado)
```

## Solución de problemas

**FFmpeg no encontrado / [WinError 2]** — FFmpeg no está instalado o no está en el PATH. Instálalo con:

```bash
winget install ffmpeg
```

Luego cierra y vuelve a abrir la terminal para que el PATH se actualice. Verifica con `ffmpeg -version`.

**Error de VRAM** — Usa un modelo más pequeño (`tiny` o `base`) si tu GPU tiene poca memoria.

**Archivo demasiado grande** — El límite es 2 GB. Para archivos mayores, divídelos antes de subir.

**CUDA no disponible a pesar de tener GPU NVIDIA** — PyTorch puede estar instalado en su versión CPU. Sigue los pasos de la sección [Configuración de GPU](#configuración-de-gpu-recomendado).

**El frontend se queda en "Cargando modelo de IA... 5%"** — Es normal. El modelo se está cargando (o descargando por primera vez). Una vez completado, el progreso avanza automáticamente. Los modelos grandes como `large` (~2.88 GB) pueden tardar varios minutos en descargarse.

**`Pipeline.from_pretrained() got an unexpected keyword argument 'use_auth_token'`** — La nueva versión de `pyannote.audio` cambió ese parámetro a `token`. Ya está corregido en `app.py`. Si el error persiste, actualiza el repositorio.

**`403 Client Error... Cannot access gated repo`** — No has aceptado los términos de uso de los modelos de pyannote en HuggingFace. Ve a https://huggingface.co/pyannote/segmentation-3.0 y https://huggingface.co/pyannote/speaker-diarization-community-1 y pulsa "Agree and access repository" en ambos.

**`name 'AudioDecoder' is not defined`** — Es un problema de `torchcodec` con Python 3.14 + PyTorch reciente. Ya está sorteado en `app.py` cargando el audio con `soundfile` después de convertirlo a WAV con FFmpeg. Si aparece, asegúrate de tener `soundfile` instalado (`python -m pip install soundfile`).

**El panel "👥 Detectar hablantes" no aparece en la interfaz** — Caché del navegador. Pulsa `Ctrl + Shift + R` para forzar recarga sin caché.
