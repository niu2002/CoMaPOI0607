"""Utilities for UTF-8-safe run logging."""

from __future__ import annotations

import atexit
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class TeeStream:
    """Mirror writes to both the original stream and a UTF-8 log file."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
        self.encoding = getattr(stream, "encoding", "utf-8")
        self.errors = getattr(stream, "errors", "replace")

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return self.stream.isatty()


def configure_utf8_console() -> None:
    """Force UTF-8 for stdout/stderr when supported."""
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _json_default(value: Any) -> str:
    return str(value)


def setup_run_logging(project_root: Path, run_type: str, dataset: str, args: Any) -> Path:
    """Create a timestamped log file and mirror stdout/stderr into it."""
    configure_utf8_console()

    logs_dir = project_root / "artifacts" / "logs" / run_type
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{run_type}_{dataset}_{timestamp}.log"
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stdout_tee = TeeStream(original_stdout, log_file)
    stderr_tee = TeeStream(original_stderr, log_file)
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee

    def _cleanup() -> None:
        try:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.flush()
        finally:
            log_file.close()

    atexit.register(_cleanup)

    args_dict = vars(args) if hasattr(args, "__dict__") else args
    print(f"Run log: {log_path}")
    print(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Working directory: {Path.cwd()}")
    print(f"Command line: {' '.join(sys.argv)}")
    print("Startup arguments:")
    print(json.dumps(args_dict, ensure_ascii=False, indent=2, default=_json_default))
    print("-" * 80)
    return log_path
