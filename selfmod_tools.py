"""
Self-Modification Toolkit for DKG
Filesystem access (project dir only), code editing with versioning,
HTTP requests, and system restart capability.
All operations are logged and versioned for rollback.
"""
import os, time, ast, shutil, json, traceback

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VERSIONS_DIR = os.path.join(PROJECT_DIR, "versions")
LOG_FILE = os.path.join(PROJECT_DIR, "data", "selfmod_log.json")

os.makedirs(VERSIONS_DIR, exist_ok=True)

def _log(action, details):
    """Append to persistent log."""
    entry = {"time": time.time(), "action": action, **details}
    try:
        log = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE) as f:
                log = json.load(f)
        log.append(entry)
        if len(log) > 500:
            log = log[-500:]
        with open(LOG_FILE, "w") as f:
            json.dump(log, f, indent=1)
    except:
        pass
    return entry

def _safe_path(path):
    """Ensure path is within project directory."""
    resolved = os.path.realpath(os.path.join(PROJECT_DIR, path))
    if not resolved.startswith(os.path.realpath(PROJECT_DIR)):
        return None
    return resolved

# ═════════════════════════════════════════════════════════════════════════════
#  FILESYSTEM TOOLS
# ═════════════════════════════════════════════════════════════════════════════
def read_file(path):
    """Read a file within the project directory."""
    safe = _safe_path(path)
    if not safe:
        return {"success": False, "error": "Path outside project directory"}
    if not os.path.exists(safe):
        return {"success": False, "error": f"File not found: {path}"}
    try:
        with open(safe) as f:
            content = f.read()
        _log("read_file", {"path": path, "size": len(content)})
        return {"success": True, "content": content, "size": len(content)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def write_file(path, content):
    """Write a file within the project directory. Creates backup if exists."""
    safe = _safe_path(path)
    if not safe:
        return {"success": False, "error": "Path outside project directory"}
    try:
        # Backup existing file
        if os.path.exists(safe):
            ts = time.strftime("%Y%m%d_%H%M%S")
            backup_name = os.path.basename(safe) + f".{ts}.bak"
            backup_path = os.path.join(VERSIONS_DIR, backup_name)
            shutil.copy2(safe, backup_path)
            _log("backup", {"path": path, "backup": backup_name})
        # Create directories if needed
        os.makedirs(os.path.dirname(safe), exist_ok=True)
        # Write
        with open(safe, "w") as f:
            f.write(content)
        _log("write_file", {"path": path, "size": len(content)})
        return {"success": True, "path": path, "size": len(content)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def list_files(path="."):
    """List files in a directory within the project."""
    safe = _safe_path(path)
    if not safe:
        return {"success": False, "error": "Path outside project directory"}
    if not os.path.isdir(safe):
        return {"success": False, "error": "Not a directory"}
    try:
        entries = []
        for name in sorted(os.listdir(safe)):
            full = os.path.join(safe, name)
            entries.append({
                "name": name,
                "type": "dir" if os.path.isdir(full) else "file",
                "size": os.path.getsize(full) if os.path.isfile(full) else 0,
            })
        _log("list_files", {"path": path, "count": len(entries)})
        return {"success": True, "entries": entries}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ═════════════════════════════════════════════════════════════════════════════
#  CODE SELF-MODIFICATION
# ═════════════════════════════════════════════════════════════════════════════
def modify_source(filename, new_content):
    """
    Modify one of the system's own source files.
    Validates Python syntax before writing. Creates versioned backup.
    Only allows: dkg_engine.py, ollama_client.py, app.py, selfmod_tools.py
    """
    allowed = {"dkg_engine.py", "ollama_client.py", "app.py", "selfmod_tools.py"}
    if filename not in allowed:
        return {"success": False, "error": f"Cannot modify {filename}. Allowed: {allowed}"}

    # Validate Python syntax
    try:
        ast.parse(new_content)
    except SyntaxError as e:
        _log("modify_source_rejected", {"filename": filename, "error": str(e)})
        return {"success": False, "error": f"Syntax error — refusing to write: {e}"}

    # Backup current version
    filepath = os.path.join(PROJECT_DIR, filename)
    if os.path.exists(filepath):
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = os.path.join(VERSIONS_DIR, f"{filename}.{ts}.bak")
        shutil.copy2(filepath, backup)
        _log("source_backup", {"filename": filename, "backup": backup})

    # Write new version
    try:
        with open(filepath, "w") as f:
            f.write(new_content)
        _log("modify_source", {"filename": filename, "size": len(new_content)})
        return {"success": True, "filename": filename, "message": "Source modified. Restart to apply."}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_source(filename):
    """Read one of the system's own source files."""
    allowed = {"dkg_engine.py", "ollama_client.py", "app.py", "selfmod_tools.py",
               "templates/index.html"}
    if filename not in allowed:
        return {"success": False, "error": f"Cannot read {filename}. Allowed: {allowed}"}
    filepath = os.path.join(PROJECT_DIR, filename)
    if not os.path.exists(filepath):
        return {"success": False, "error": "File not found"}
    with open(filepath) as f:
        return {"success": True, "content": f.read()}

def list_versions():
    """List all versioned backups."""
    if not os.path.isdir(VERSIONS_DIR):
        return {"success": True, "versions": []}
    versions = []
    for name in sorted(os.listdir(VERSIONS_DIR), reverse=True):
        full = os.path.join(VERSIONS_DIR, name)
        versions.append({
            "name": name,
            "size": os.path.getsize(full),
            "time": os.path.getmtime(full),
        })
    return {"success": True, "versions": versions[:30]}

def rollback(backup_name):
    """Restore a file from a versioned backup."""
    backup_path = os.path.join(VERSIONS_DIR, backup_name)
    if not os.path.exists(backup_path):
        return {"success": False, "error": "Backup not found"}
    # Parse original filename from backup name (e.g., "dkg_engine.py.20260404_210000.bak")
    parts = backup_name.split(".")
    if len(parts) < 3:
        return {"success": False, "error": "Cannot determine original filename"}
    original = parts[0] + ".py"
    target = os.path.join(PROJECT_DIR, original)
    shutil.copy2(backup_path, target)
    _log("rollback", {"backup": backup_name, "target": original})
    return {"success": True, "restored": original, "from": backup_name}

# ═════════════════════════════════════════════════════════════════════════════
#  HTTP REQUESTS (for reaching the outside world)
# ═════════════════════════════════════════════════════════════════════════════
def http_get(url, timeout=10):
    """Make an HTTP GET request. Logged."""
    _log("http_get", {"url": url})
    try:
        import requests
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "DKG-System/2.1"})
        return {"success": True, "status": r.status_code,
                "body": r.text[:5000], "headers": dict(r.headers)}
    except ImportError:
        return {"success": False, "error": "requests library not available. pip install requests"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def http_post(url, data=None, json_data=None, timeout=10):
    """Make an HTTP POST request. Logged."""
    _log("http_post", {"url": url, "data_size": len(str(data or json_data or ""))})
    try:
        import requests
        r = requests.post(url, data=data, json=json_data, timeout=timeout,
                         headers={"User-Agent": "DKG-System/2.1"})
        return {"success": True, "status": r.status_code,
                "body": r.text[:5000], "headers": dict(r.headers)}
    except ImportError:
        return {"success": False, "error": "requests library not available. pip install requests"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ═════════════════════════════════════════════════════════════════════════════
#  SYSTEM OPERATIONS
# ═════════════════════════════════════════════════════════════════════════════
def get_selfmod_log(last_n=20):
    """Get recent self-modification log entries."""
    try:
        if not os.path.exists(LOG_FILE):
            return {"success": True, "entries": []}
        with open(LOG_FILE) as f:
            log = json.load(f)
        return {"success": True, "entries": log[-last_n:]}
    except Exception as e:
        return {"success": False, "error": str(e)}

def system_info():
    """Get system information."""
    import platform, sys
    return {
        "success": True,
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "project_dir": PROJECT_DIR,
        "pid": os.getpid(),
        "files": [f for f in os.listdir(PROJECT_DIR) if not f.startswith(".")],
    }

# ═════════════════════════════════════════════════════════════════════════════
#  REGISTRY: all tools the DKG sandbox can call
# ═════════════════════════════════════════════════════════════════════════════
SELFMOD_TOOLS = {
    "read_file": read_file,
    "write_file": write_file,
    "list_files": list_files,
    "modify_source": modify_source,
    "get_source": get_source,
    "list_versions": list_versions,
    "rollback": rollback,
    "http_get": http_get,
    "http_post": http_post,
    "get_selfmod_log": get_selfmod_log,
    "system_info": system_info,
}
