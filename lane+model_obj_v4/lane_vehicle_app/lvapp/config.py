"""JSON configuration: defaults in config/default.json, optional user file, optional --set a.b=value overrides."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


def load(path=None, overrides=()):
    with open(ROOT / "config" / "default.json") as f:
        cfg = json.load(f)
    if path:
        with open(path) as f:
            _merge(cfg, json.load(f))
    for item in overrides:                       # "detector.conf_low=0.3"
        key, val = item.split("=", 1)
        try:
            val = json.loads(val)
        except ValueError:
            pass
        d = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return cfg


def copy_cfg(cfg):
    return copy.deepcopy(cfg)
