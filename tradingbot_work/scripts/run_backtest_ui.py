#!/usr/bin/env python3
"""Launch the doc pivot + PE backtest web UI."""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn

from src.safety.live_gate import assert_paper_trading_only


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _server_ready(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/guide-filters", timeout=3) as resp:
            if resp.status != 200:
                return False
        with urllib.request.urlopen(f"http://{host}:{port}/api/atr-buffer-options", timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _pid_listening_on_port(port: int) -> int | None:
    try:
        out = subprocess.check_output(["netstat", "-ano"], text=True, errors="replace")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    needle = f":{port}"
    for line in out.splitlines():
        if needle in line and "LISTENING" in line:
            parts = line.split()
            if parts:
                try:
                    return int(parts[-1])
                except ValueError:
                    continue
    return None


def _kill_port(port: int) -> bool:
    pid = _pid_listening_on_port(port)
    if pid is None:
        return False
    subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False, capture_output=True)
    time.sleep(0.8)
    return True


def main() -> int:
    assert_paper_trading_only()
    parser = argparse.ArgumentParser(description="Doc pivot + PE backtest web UI")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Stop any process on the port and start a fresh server",
    )
    args = parser.parse_args()
    host = "127.0.0.1"

    if _port_open(host, args.port):
        if args.force:
            print(f"Stopping old process on port {args.port}…")
            _kill_port(args.port)
        elif _server_ready(host, args.port):
            print(f"UI already running with latest endpoints.")
            print(f"Open http://{host}:{args.port} in your browser.")
            print(f"To restart: python scripts\\run_backtest_ui.py --force")
            return 0
        else:
            print(f"Stale UI server on port {args.port} (old code). Restarting…")
            _kill_port(args.port)
            if _port_open(host, args.port):
                print("Could not free the port. Try: python scripts\\run_backtest_ui.py --force --port 8766")
                return 1

    print("Doc Pivot + PE Backtest UI")
    print(f"Open http://{host}:{args.port} in your browser")
    print("Press Ctrl+C to stop.")
    uvicorn.run("backtest_ui.server:app", host=host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
