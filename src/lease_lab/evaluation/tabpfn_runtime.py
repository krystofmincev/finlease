"""Bounded local checkpoint setup and process-local offline inference guard."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import sys
import urllib.request

from lease_lab.config import ROOT


def configuration() -> dict:
    return json.loads((ROOT / "configs/tabpfn-v2.json").read_text())


def checkpoint(config: dict) -> Path:
    folder = ROOT / ".local/tabpfn-models"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / config["checkpoint_filename"]
    if (
        path.exists()
        and hashlib.sha256(path.read_bytes()).hexdigest() == config["checkpoint_sha256"]
    ):
        return path
    url = f"https://huggingface.co/{config['checkpoint_repo']}/resolve/{config['checkpoint_revision']}/{config['checkpoint_filename']}"
    temporary = path.with_suffix(".download")
    digest = hashlib.sha256()
    with (
        urllib.request.urlopen(url, timeout=30) as response,
        temporary.open("wb") as output,
    ):
        while data := response.read(1024 * 1024):
            digest.update(data)
            output.write(data)
    if digest.hexdigest() != config["checkpoint_sha256"]:
        raise ValueError("official checkpoint digest mismatch")
    temporary.replace(path)
    return path


def prepare_environment():
    for key, sub in {
        "HF_HOME": "hf",
        "TORCH_HOME": "torch",
        "MPLCONFIGDIR": "mpl",
        "XDG_CACHE_HOME": "cache",
        "TABPFN_MODEL_CACHE_DIR": "tabpfn-models",
        "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    }.items():
        directory = ROOT / ".local" / sub
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    os.environ["TABPFN_MODEL_VERSION"] = "v2"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


@contextmanager
def offline_inference(seconds: int):
    previous_connect = socket.socket.connect
    previous_alarm = signal.getsignal(signal.SIGALRM)

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError("offline challenger attempted network access")
        return previous_connect(sock, address)

    def expired(*args):
        raise TimeoutError("bounded challenger inference time exceeded")

    socket.socket.connect = connect
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        socket.socket.connect = previous_connect


def peak_memory_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)
