#!/usr/bin/env python3
"""Hook SessionStart: inyecta el cerebro arquitectural al inicio de cada sesión."""
import subprocess
import sys
import json
from pathlib import Path

HERE = Path(__file__).parent

try:
    result = subprocess.run(
        [sys.executable, str(HERE / "graph.py"), "cerebro"],
        capture_output=True, text=True, encoding="utf-8", timeout=20, cwd=str(HERE)
    )
    output = result.stdout or ""

    # Limpiar códigos ANSI
    import re
    output = re.sub(r"\x1b\[[0-9;]*m", "", output)

    if len(output) > 8000:
        output = output[:8000] + "\n... [truncado]"

    payload = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": f"=== ARQUITECTURA DEL PROYECTO (graph.py cerebro) ===\n\n{output}"
        }
    }
except Exception as e:
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": f"graph.py no disponible: {e}"
        }
    }

sys.stdout.write(json.dumps(payload))
