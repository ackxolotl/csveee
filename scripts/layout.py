"""
Where the metadata lives and where the CSV bytes live.

Mirrors `tests/common/data.rs`; the two must stay in step.
"""

from __future__ import annotations

import os
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITES_DIR = os.path.join(ROOT, "suites")
CONFIG = os.path.join(ROOT, "data.local.toml")

#: Suites whose CSV bytes are committed next to their metadata
COMMITTED = {"fixtures"}

_config_cache: dict | None = None


def _config() -> dict:
    global _config_cache
    if _config_cache is None:
        try:
            with open(CONFIG, "rb") as f:
                _config_cache = tomllib.load(f)
        except FileNotFoundError:
            _config_cache = {}
        except tomllib.TOMLDecodeError as e:
            # Left unparsed, a typo would degrade to "every suite is missing".
            raise SystemExit(f"failed to parse {CONFIG}: {e}") from None
    return _config_cache


def suite_dir(name: str) -> str:
    """Committed metadata for `name`: manifest.toml, overrides.toml, …"""
    return os.path.join(SUITES_DIR, name)


def data_dir(name: str) -> str:
    """The directory holding `name`'s CSV bytes, per the order above."""
    if name in COMMITTED:
        return suite_dir(name)

    env = os.environ.get(f"CSVEEE_DATA_{name.replace('-', '_').upper()}", "").strip()
    if env:
        return env

    env = os.environ.get("CSVEEE_DATA", "").strip()
    if env:
        return os.path.join(env, name)

    cfg = _config()
    per_suite = cfg.get("suites", {}).get(name)
    if per_suite:
        return per_suite
    root = cfg.get("root")
    if root:
        return os.path.join(root, name)

    return os.path.join(ROOT, "data", name)


def describe(name: str) -> str:
    """`data_dir` relative to the repo when inside it, absolute otherwise."""
    path = os.path.abspath(data_dir(name))
    rel = os.path.relpath(path, ROOT)
    return rel if not rel.startswith(os.pardir) else path
