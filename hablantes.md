# 👥 Asignación de nombres a hablantes

Cuando activas la diarización, pyannote etiqueta cada turno con `SPEAKER_00`, `SPEAKER_01`, etc. Estos son identificadores anónimos. Este documento explica las dos formas de asignarles **nombres reales** (Juan, María, etc.).

---

## Resumen rápido

| Opción | Cuándo usarla | Esfuerzo | Automático |
|---|---|---|---|
| **A. Renombrado manual** | Audios variados, hablantes distintos cada vez | Bajo | No (lo haces tú al final) |
| **B. Reconocimiento por voz** | Reuniones recurrentes con las mismas personas | Alto | Sí (una vez registradas las voces) |

**Estado actual del proyecto:** ✅ Opción A implementada · ❌ Opción B pendiente.

---

## Opción A — Renombrado manual post-transcripción

### ¿Cómo funciona?

Después de que la transcripción termina y el audio queda diarizado con `[SPEAKER_00]`, `[SPEAKER_01]`, etc., la interfaz muestra **un input por cada hablante detectado**. Escribes el nombre real (ej. "Juan"), pulsas **"Aplicar nombres"** y:

1. El frontend reemplaza todas las apariciones de `[SPEAKER_XX]` por `[Juan]` en el texto mostrado.
2. Llama a un endpoint del backend que reescribe el archivo `.txt` en `transcripciones/` con los nombres definitivos.
3. El nombre se mantiene también si copias el texto.

### Ventajas
- 100% local, sin dependencias adicionales.
- Cero entrenamiento previo: funciona desde el primer audio.
- Tú validas el resultado, así que la precisión es perfecta.
- No le importa si el audio es ruidoso, los hablantes están resfriados, etc.

### Desventajas
- Requiere intervención humana en cada transcripción.
- Necesitas escuchar fragmentos para saber qué `SPEAKER_XX` corresponde a quién.

### Flujo típico

```
1. Subes audio + activas "Detectar hablantes" + "Transcribir"
2. Aparece el resultado:
     [SPEAKER_00]: Buenos días equipo, ¿ya enviaron el reporte?
     [SPEAKER_01]: Sí, lo mandé anoche.
     [SPEAKER_00]: Perfecto, gracias.
3. Aparecen 2 inputs:
     SPEAKER_00 → [ Juan         ]
     SPEAKER_01 → [ María        ]
4. Pulsas "Aplicar nombres" y queda:
     [Juan]:  Buenos días equipo, ¿ya enviaron el reporte?
     [María]: Sí, lo mandé anoche.
     [Juan]:  Perfecto, gracias.
```

### Implementación (resumen técnico)

**Backend (`app.py`):**
- Nuevo endpoint `POST /renombrar/<job_id>` que recibe `{ "mapeo": {"SPEAKER_00": "Juan", ...} }`.
- Lee el `.txt` guardado, hace `replace` por cada par y lo reescribe.
- Devuelve el texto actualizado.

**Frontend (`index.html`):**
- Tras `data.estado === 'completado'`, parsea el texto con un regex `/\[SPEAKER_\d+\]/g` para extraer los hablantes únicos.
- Si hay diarización, renderiza un panel con un input por hablante.
- Botón **"Aplicar nombres"** envía el mapeo, recibe el texto actualizado y lo pinta.

---

## Opción B — Reconocimiento automático por voz (no implementada)

### ¿Cómo funciona?

Funciona en dos fases: **enrollment** (registrar voces una vez) y **matching** (identificar al transcribir).

#### Fase 1 – Enrollment

1. Grabas o subes un fragmento limpio de cada persona, idealmente **10–30 segundos** de habla continua, sin ruido ni música. Ejemplo: `juan.wav`, `maria.wav`.
2. Un modelo de embeddings (ej. `pyannote/embedding` o `speechbrain/spkrec-ecapa-voxceleb`) recibe el audio y devuelve un **vector de 256–512 números** — la "huella vocal" de esa persona. Dos grabaciones de la misma persona dan vectores muy parecidos; de personas distintas, vectores muy diferentes.
3. Guardas en disco (por ejemplo `voces/voces.json`) un diccionario:
   ```json
   {
     "Juan":  [0.12, -0.45, 0.88, ...],
     "María": [-0.33, 0.19, -0.07, ...]
   }
   ```

#### Fase 2 – Matching

1. Whisper + pyannote generan los segmentos por hablante: `SPEAKER_00`, `SPEAKER_01`, etc.
2. Para **cada `SPEAKER_XX` detectado**, juntas un par de minutos de su audio (los recortes donde habla) y le sacas el embedding con el mismo modelo de la fase 1.
3. Comparas ese embedding contra cada uno de la biblioteca usando **similitud coseno** (un número entre -1 y 1, donde 1 = idénticos):
   - Si la similitud con "Juan" es 0.85 y con "María" 0.31 → este `SPEAKER_00` es **Juan**.
   - Si **ninguna** supera un umbral (típicamente 0.5–0.7) → lo dejas como `SPEAKER_00` (desconocido).
4. Reemplazas las etiquetas en el texto final.

### Ventajas
- Automático: una vez registrada una voz, aparece sola en cada nueva reunión.
- Ideal para reuniones recurrentes con el mismo equipo.

### Desventajas
- Necesitas muestras limpias — si la persona habla bajito o hay eco, el embedding sale ruidoso.
- Voces parecidas (mismo género, edad, tono) pueden confundirse.
- Cambios físicos (resfriado, ronquera) bajan la precisión.
- Más infra: UI para registrar voces, almacenamiento de embeddings, manejo de "desconocidos".

### Implementación (esbozo técnico)

**Backend:**
1. Función `extraer_embedding(ruta_audio: str) -> list[float]` que usa `pyannote/embedding` o `SpeechBrain`.
2. Endpoint `POST /voces/registrar` con `{ nombre, audio }` → guarda embedding en `voces/voces.json`.
3. Endpoint `GET /voces` → lista las voces registradas.
4. Endpoint `DELETE /voces/<nombre>` → eliminar una voz.
5. En `procesar_audio()`, después de la diarización:
   - Por cada `SPEAKER_XX`, concatenar sus segmentos en un audio temporal.
   - Sacar embedding y comparar (cosine similarity) contra `voces.json`.
   - Si la mejor similitud supera el umbral, mapear `SPEAKER_XX → nombre`.

**Frontend:**
- Nueva pestaña/sección "Voces registradas" con:
  - Lista de voces guardadas + botón eliminar.
  - Form para registrar una nueva voz (nombre + audio).
- Indicador en el resultado: hablantes auto-identificados vs. desconocidos.

### Dependencias adicionales

```
pyannote.audio  (ya instalado — incluye pyannote/embedding)
numpy           (para cosine similarity)
scipy           (opcional, para distancias)
```

### Cuándo escalar a opción B

- Si haces **3+ transcripciones por semana** con un grupo recurrente, los minutos ahorrados compensan la inversión inicial.
- Si haces transcripciones esporádicas de gente distinta, **opción A es siempre mejor**.

---

## Notas

- Las dos opciones son **compatibles**: podrías combinarlas — usar B para auto-identificar a los conocidos, y A para renombrar manualmente a los desconocidos.
- El umbral de similitud en B es delicado: muy bajo identifica falsamente, muy alto deja todo como desconocido. Empieza en 0.6 y ajusta.
- Para opción B necesitas **el mismo modelo de embedding** en enrollment y matching. No mezcles `pyannote/embedding` con `speechbrain/spkrec-ecapa-voxceleb`.
