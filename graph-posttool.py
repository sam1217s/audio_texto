#!/usr/bin/env python3
"""Hook PostToolUse: actualiza el grafo tras cada edición de .py o .html."""
import subprocess
import sys
import json
import re
from pathlib import Path

HERE = Path(__file__).parent

input_data = sys.stdin.read()
try:
    data = json.loads(input_data)
except Exception:
    sys.exit(0)

file_path = (
    data.get("tool_input", {}).get("file_path", "")
    or data.get("tool_response", {}).get("filePath", "")
)

graph_files = {"graph.py", "graph-hook.py", "graph-posttool.py"}
is_graph = any(file_path.endswith(f) for f in graph_files)
is_project = (file_path.endswith(".py") or file_path.endswith(".html")) and "node_modules" not in file_path

if not is_project or is_graph:
    sys.exit(0)

file_name = Path(file_path).name

try:
    result = subprocess.run(
        [sys.executable, str(HERE / "graph.py"), "cerebro"],
        capture_output=True, text=True, encoding="utf-8", timeout=20, cwd=str(HERE)
    )
    output = result.stdout or ""
    output = re.sub(r"\x1b\[[0-9;]*m", "", output)
    if len(output) > 8000:
        output = output[:8000] + "\n... [truncado]"

    payload = {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": f"=== GRAFO ACTUALIZADO (después de editar {file_name}) ===\n\n{output}"
        }
    }
    sys.stdout.write(json.dumps(payload))
except Exception:
    sys.exit(0)
