# Whisper Audio Transcriptor

Herramienta para transcribir archivos de audio y video a texto usando [OpenAI Whisper](https://github.com/openai/whisper). Disponible como interfaz web (Flask) y como script de línea de comandos.

## Características

- **Interfaz web** con barra de progreso en tiempo real
- **CLI** para transcribir desde la terminal
- Procesamiento asíncrono: la UI no se bloquea mientras transcribe
- Detección automática de idioma
- Transcripciones guardadas automáticamente en `transcripciones/`
- Soporta archivos de hasta **2 GB**
- Compatible con múltiples formatos de audio y video
- **Funciona 100% offline** — no envía audio a ningún servidor externo

## Requisitos previos

- Python 3.8+
- [FFmpeg](https://ffmpeg.org/download.html) instalado y en el PATH del sistema

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

## Uso

### Interfaz web

```bash
python app.py
```

Abre el navegador en **http://localhost:5200**, sube tu archivo y obtén la transcripción directamente en el navegador.

### Línea de comandos (CLI)

```bash
# Transcripción básica
python transcriber.py mi_audio.mp3

# Especificar modelo
python transcriber.py mi_audio.mp3 --modelo large

# Especificar idioma (evita la detección automática)
python transcriber.py mi_audio.mp3 --idioma es
```

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
whisper/
├── app.py               # Servidor web Flask
├── transcriber.py       # Script CLI
├── requirements.txt     # Dependencias
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
