"""One project-owned, versioned configuration; no environment or external inputs."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/lab-v1.json"
CONFIG = json.loads(CONFIG_PATH.read_text())
SEGMENTS = ("new", "growing", "established", "restart")
HISTORICAL_OFFERS = ("r800", "r900", "r1000")
PRIOR_BETA = {"default": (1.0, 1.0), "weak": (0.5, 0.5), "strong": (2.0, 2.0)}
PRIOR_SIGMA = {"default": 3.0, "weak": 6.0, "strong": 1.5}
