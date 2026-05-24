#!/usr/bin/env python3
"""
graph.py — grafo de dependencias del proyecto audio_texto
Uso: python graph.py [mapa|cerebro|detective]
"""
import os
import re
import sys
from pathlib import Path

# Forzar UTF-8 en la salida estándar (necesario en consolas Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Colores ANSI ─────────────────────────────────────────────────────────────
R = "\033[0m"
B = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"
BLUE = "\033[34m"
WHITE = "\033[37m"

ROOT = Path(__file__).parent

# Capas del proyecto
LAYERS = {
    "apps":      ["app.py", "appmodel.py"],
    "scripts":   ["transcriber.py", "graph.py", "graph-hook.py", "graph-posttool.py"],
    "templates": ["templates"],
}

STDLIB = {
    "os", "sys", "re", "io", "json", "time", "uuid", "math", "gc",
    "threading", "subprocess", "pathlib", "datetime", "types", "typing",
    "collections", "functools", "itertools", "contextlib", "abc",
    "traceback", "logging", "copy", "shutil", "tempfile", "socket",
    "http", "urllib", "email", "html", "xml", "csv", "struct",
    "hashlib", "hmac", "secrets", "base64", "binascii", "codecs",
    "enum", "dataclasses", "inspect", "importlib", "pkgutil",
    "argparse", "textwrap", "string", "random", "statistics",
    "asyncio", "concurrent", "multiprocessing", "queue",
    "sqlite3", "unittest", "doctest", "pdb",
}

KNOWN_PACKAGES = {
    "flask": "Flask",
    "whisper": "OpenAI-Whisper",
    "torch": "PyTorch",
    "pyannote": "pyannote.audio",
    "tqdm": "tqdm",
    "dotenv": "python-dotenv",
    "soundfile": "soundfile",
    "fpdf": "fpdf2",
    "numpy": "numpy",
    "scipy": "scipy",
    "ffmpeg": "ffmpeg-python",
}


def collect_files():
    """Recolecta todos los .py del proyecto (excluye venv, __pycache__)."""
    skip = {"__pycache__", ".git", "venv", "env", ".venv", "site-packages",
            "audios", "transcripciones", "voces"}
    files = []
    for p in ROOT.rglob("*.py"):
        parts = set(p.parts)
        if parts & skip:
            continue
        files.append(p)
    # Templates HTML
    for p in ROOT.rglob("*.html"):
        if "__pycache__" not in str(p):
            files.append(p)
    return sorted(files)


def file_id(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def extract_imports(path: Path) -> dict:
    """Extrae imports de un .py. Devuelve {stdlib: [...], packages: [...], local: [...]}."""
    result = {"stdlib": [], "packages": [], "local": []}
    if path.suffix != ".py":
        return result
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return result

    # import X [as Y], import X, Y [as Z] — captura solo hasta fin de línea
    for m in re.finditer(r"^[ \t]*import[ \t]+([\w., \t]+)", src, re.MULTILINE):
        raw = m.group(1).split("#")[0].strip()  # quitar comentarios inline
        # cada token separado por coma es un módulo, posiblemente con "as alias"
        for token in raw.split(","):
            parts = token.strip().split()
            if not parts:
                continue
            name = parts[0]  # "whisper.transcribe as wt" → "whisper.transcribe"
            if not re.match(r"^[\w.]+$", name):
                continue
            root_name = name.split(".")[0]
            _classify(root_name, result)

    # from X import Y — solo nos interesa X
    for m in re.finditer(r"^[ \t]*from[ \t]+([\w.]+)[ \t]+import", src, re.MULTILINE):
        name = m.group(1).strip()
        if name.startswith("."):
            if name not in result["local"]:
                result["local"].append(name)
        else:
            root_name = name.split(".")[0]
            _classify(root_name, result)

    # deduplicate
    for k in result:
        result[k] = sorted(set(result[k]))
    return result


def _classify(name: str, result: dict):
    if name in STDLIB:
        if name not in result["stdlib"]:
            result["stdlib"].append(name)
    else:
        if name not in result["packages"]:
            result["packages"].append(name)


def get_layer(path: Path) -> str:
    rel = file_id(path)
    if rel.startswith("templates/"):
        return "templates"
    name = path.name
    if name in LAYERS["apps"]:
        return "apps"
    if name in LAYERS["scripts"]:
        return "scripts"
    return "otros"


def build_graph(files):
    graph = {}
    for f in files:
        fid = file_id(f)
        imports = extract_imports(f)
        graph[fid] = {
            "path": f,
            "layer": get_layer(f),
            "imports": imports,
        }
    return graph


# ── Modos de presentación ────────────────────────────────────────────────────

def mode_mapa(graph):
    print(f"\n{B}{CYAN}═══ MAPA DE ARCHIVOS Y DEPENDENCIAS ═══{R}\n")
    by_layer = {}
    for fid, info in graph.items():
        layer = info["layer"]
        by_layer.setdefault(layer, []).append((fid, info))

    order = ["apps", "templates", "scripts", "otros"]
    for layer in order:
        items = by_layer.get(layer, [])
        if not items:
            continue
        print(f"{B}{YELLOW}[ {layer.upper()} ]{R}")
        for fid, info in items:
            pkgs = info["imports"]["packages"]
            std = info["imports"]["stdlib"]
            print(f"  {GREEN}{fid}{R}")
            if pkgs:
                print(f"    {DIM}packages:{R} {', '.join(pkgs)}")
            if std:
                print(f"    {DIM}stdlib:  {R} {', '.join(std)}")
        print()


def mode_cerebro(graph):
    print(f"\n{B}{CYAN}═══ ARQUITECTURA DEL PROYECTO (cerebro) ═══{R}\n")

    # Resumen por capa
    by_layer = {}
    for fid, info in graph.items():
        layer = info["layer"]
        by_layer.setdefault(layer, []).append((fid, info))

    layer_labels = {
        "apps":      "APLICACIONES FLASK",
        "templates": "TEMPLATES HTML",
        "scripts":   "SCRIPTS / HERRAMIENTAS",
        "otros":     "OTROS",
    }

    print(f"{B}Capas del proyecto:{R}\n")
    for layer, label in layer_labels.items():
        items = by_layer.get(layer, [])
        if not items:
            continue
        total_pkgs = set()
        for _, info in items:
            total_pkgs.update(info["imports"]["packages"])
        bar = "█" * len(items)
        print(f"  {YELLOW}{label:<30}{R}  {bar}  ({len(items)} archivo{'s' if len(items)!=1 else ''})")
        for fid, info in items:
            pkgs = info["imports"]["packages"]
            print(f"    {GREEN}▸ {fid}{R}")
            if pkgs:
                for p in pkgs:
                    label_pkg = KNOWN_PACKAGES.get(p, p)
                    print(f"        {DIM}→ {label_pkg}{R}")
        print()

    # Paquetes más usados (hubs)
    pkg_count = {}
    for fid, info in graph.items():
        for p in info["imports"]["packages"]:
            pkg_count[p] = pkg_count.get(p, 0) + 1
    if pkg_count:
        top = sorted(pkg_count.items(), key=lambda x: -x[1])[:10]
        print(f"{B}Top paquetes externos (hubs):{R}\n")
        for p, c in top:
            bar = "█" * c
            label_pkg = KNOWN_PACKAGES.get(p, p)
            print(f"  {MAGENTA}{label_pkg:<30}{R}  {bar}  ({c} archivo{'s' if c!=1 else ''})")
        print()

    # Módulos de dominio (keywords en nombres de archivo)
    domain_keywords = {
        "transcripción": ["transcrib", "whisper"],
        "diarización":   ["diariz", "hablante", "speaker"],
        "voz/embedding": ["voz", "embed", "matching"],
        "audio":         ["audio", "ffmpeg", "ffprobe"],
        "web/api":       ["flask", "route", "jsonify", "render_template"],
        "almacenamiento":["transcripciones", "audios", "voces"],
    }
    print(f"{B}Módulos de dominio detectados:{R}\n")
    for domain, keywords in domain_keywords.items():
        matched_files = []
        for fid, info in graph.items():
            try:
                src = info["path"].read_text(encoding="utf-8", errors="ignore").lower()
            except Exception:
                continue
            if any(kw in src for kw in keywords):
                matched_files.append(fid)
        if matched_files:
            print(f"  {CYAN}{domain}:{R}")
            for f in matched_files:
                print(f"    {DIM}• {f}{R}")
    print()


def mode_detective(graph):
    print(f"\n{B}{CYAN}═══ ANÁLISIS DETECTIVE ═══{R}\n")

    py_files = [(fid, info) for fid, info in graph.items() if fid.endswith(".py")]
    html_files = [(fid, info) for fid, info in graph.items() if fid.endswith(".html")]

    print(f"{B}Estadísticas globales:{R}")
    print(f"  Archivos Python:   {len(py_files)}")
    print(f"  Templates HTML:    {len(html_files)}")
    total_pkgs = set()
    total_std = set()
    for _, info in py_files:
        total_pkgs.update(info["imports"]["packages"])
        total_std.update(info["imports"]["stdlib"])
    print(f"  Paquetes externos: {len(total_pkgs)}")
    print(f"  Módulos stdlib:    {len(total_std)}")
    print()

    # Archivos más complejos (más imports)
    print(f"{B}Top archivos por cantidad de imports:{R}")
    ranked = sorted(py_files, key=lambda x: -(len(x[1]["imports"]["packages"]) + len(x[1]["imports"]["stdlib"])))
    for fid, info in ranked[:10]:
        total = len(info["imports"]["packages"]) + len(info["imports"]["stdlib"])
        pkgs = len(info["imports"]["packages"])
        print(f"  {GREEN}{fid:<40}{R}  total={total}  (externos={pkgs})")
    print()

    # Paquetes exclusivos de cada app
    print(f"{B}Paquetes exclusivos por aplicación:{R}")
    all_app_pkgs = {}
    for fid, info in py_files:
        if info["layer"] == "apps":
            all_app_pkgs[fid] = set(info["imports"]["packages"])
    for fid, pkgs in all_app_pkgs.items():
        otros = set()
        for fid2, pkgs2 in all_app_pkgs.items():
            if fid2 != fid:
                otros.update(pkgs2)
        exclusivos = pkgs - otros
        if exclusivos:
            print(f"  {GREEN}{fid}{R}: {', '.join(sorted(exclusivos))}")
        else:
            print(f"  {GREEN}{fid}{R}: (sin paquetes exclusivos)")
    print()

    # Paquetes compartidos entre todas las apps
    if all_app_pkgs:
        shared = set.intersection(*all_app_pkgs.values()) if all_app_pkgs else set()
        print(f"{B}Paquetes compartidos entre apps:{R}")
        for p in sorted(shared):
            label_pkg = KNOWN_PACKAGES.get(p, p)
            print(f"  {MAGENTA}• {label_pkg}{R}")
        print()

    # Archivos sin paquetes externos (huérfanos de deps)
    orphans = [fid for fid, info in py_files if not info["imports"]["packages"]]
    if orphans:
        print(f"{B}Archivos sin paquetes externos:{R}")
        for f in orphans:
            print(f"  {DIM}• {f}{R}")
        print()

    # Variables de entorno usadas
    env_vars = set()
    for fid, info in py_files:
        try:
            src = info["path"].read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in re.finditer(r'os\.environ(?:\.get)?\(["\'](\w+)["\']', src):
            env_vars.add(m.group(1))
    if env_vars:
        print(f"{B}Variables de entorno usadas:{R}")
        for v in sorted(env_vars):
            print(f"  {YELLOW}• {v}{R}")
        print()

    # Endpoints Flask detectados
    print(f"{B}Endpoints Flask detectados:{R}")
    for fid, info in py_files:
        if info["layer"] != "apps":
            continue
        try:
            src = info["path"].read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        routes = re.findall(r'@app\.route\(["\']([^"\']+)["\'](?:.*?methods=\[([^\]]+)\])?', src)
        if routes:
            print(f"  {GREEN}{fid}{R}:")
            for path_r, methods in routes:
                m_str = methods.replace('"', '').replace("'", "").strip() if methods else "GET"
                print(f"    {DIM}{m_str:<20}{R} {path_r}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "cerebro"
    files = collect_files()
    graph = build_graph(files)

    if mode == "mapa":
        mode_mapa(graph)
    elif mode == "detective":
        mode_detective(graph)
    else:
        mode_cerebro(graph)
