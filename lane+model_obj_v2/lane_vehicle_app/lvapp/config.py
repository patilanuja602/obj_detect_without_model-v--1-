"""Configuration: JSON preset + optional `--set section.key=value` overrides.  Attribute-style access (cfg.vehicle.conf)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent            # project folder (all relative paths resolve against it)


class Cfg(dict):
    """dict with attribute access.  Nested dicts are converted ONCE (stored as Cfg), so `cfg.vehicle.threads = 4` really modifies the config."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        for key, v in list(self.items()):
            if isinstance(v, dict) and not isinstance(v, Cfg):
                dict.__setitem__(self, key, Cfg(v))

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v


def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


def load(path=None, overrides=()):
    base = json.load(open(ROOT / "config" / "default.json"))
    if path and Path(path).name != "default.json":
        p = Path(path)
        if not p.exists():
            p = ROOT / "config" / path
        _merge(base, json.load(open(p)))
    for o in overrides:
        key, val = o.split("=", 1)
        try:
            val = json.loads(val)
        except ValueError:
            pass
        d = base
        parts = key.split(".")
        for q in parts[:-1]:
            d = d.setdefault(q, {})
        d[parts[-1]] = val
    return Cfg(base)


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
