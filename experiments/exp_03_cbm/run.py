from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "_base" in cfg:
        base_path = ROOT / cfg.pop("_base")
        with open(base_path, encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}
        _deep_merge(base, cfg)
        cfg = base

    return cfg


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def main(config_path: Path) -> None:
    cfg = load_config(config_path)
    experiment_name = cfg.get("experiment", {}).get("name", "exp_03_cbm")
    results_root = cfg.get("logging", {}).get("results_dir", "experiments/results")
    results_dir = ROOT / results_root / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": experiment_name,
        "status": "not_implemented",
        "message": "CBM experiment runner allineato al nuovo approccio, logica training da completare.",
        "config": cfg,
    }
    (results_dir / "todo_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.warning("Runner placeholder salvato in %s", results_dir / "todo_status.json")


def cli_main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    cli_main()
