"""CLI for the prequential AQO evaluation experiment."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import (  # noqa: E402
    get_friedman_datasets,
    get_hyperplane_datasets,
    get_rbf_datasets,
    get_real_datasets,
)
from src.evaluation import run_prequential_eval_parallel  # noqa: E402
from src.logging_setup import configure_logging  # noqa: E402
from src.models import get_models  # noqa: E402
from src.paths import REPO_ROOT as SRC_REPO_ROOT  # noqa: E402
from src.paths import resolve_repo_path  # noqa: E402

logger = logging.getLogger(__name__)
DEFAULT_CONFIG = SRC_REPO_ROOT / "configs" / "experiment.yaml"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Adaptive QO prequential evaluation suite.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML config (defaults match the original experiment).",
    )
    parser.add_argument("--instances", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Pickle path (repo-relative or absolute).")
    parser.add_argument("--n-jobs", type=int, default=None)
    return parser.parse_args(argv)


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.instances is not None:
        config["instances"] = args.instances
    if args.seed is not None:
        config["seed"] = args.seed
    if args.n_jobs is not None:
        config["n_jobs"] = args.n_jobs
    if args.output is not None:
        config.setdefault("output", {})["pickle"] = str(args.output)
    return config


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def build_datasets(config: dict, rng: np.random.Generator) -> dict:
    instances = config["instances"]
    friedman_cfg = config["friedman"]
    datasets = {}
    datasets.update(
        get_friedman_datasets(
            friedman_cfg["drift_type"],
            n_datasets=friedman_cfg["n_datasets"],
            n_instances=instances,
            rng=rng,
        )
    )
    datasets.update(get_hyperplane_datasets(config["hyperplane"]["n_datasets"], rng=rng))
    datasets.update(get_rbf_datasets(config["rbf"]["n_datasets"], rng=rng))
    if config.get("real", {}).get("enabled", True):
        datasets.update(get_real_datasets())
    return datasets


def write_manifest(pickle_path: Path, config: dict, models: dict, datasets: dict) -> Path:
    manifest_path = pickle_path.with_suffix(".json")
    payload = {
        "seed": config["seed"],
        "instances": config["instances"],
        "n_jobs": config["n_jobs"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": list(models),
        "n_datasets": len(datasets),
        "pickle": str(pickle_path),
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)
    config_path = args.config if args.config.is_absolute() else resolve_repo_path(args.config)
    config = apply_overrides(load_config(config_path), args)

    rng = seed_everything(int(config["seed"]))
    instances = int(config["instances"])
    n_jobs = int(config["n_jobs"])
    output_path = resolve_repo_path(config["output"]["pickle"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting experiment seed=%s instances=%s n_jobs=%s output=%s",
        config["seed"],
        instances,
        n_jobs,
        output_path,
    )

    datasets = build_datasets(config, rng)
    models = get_models()
    logs = run_prequential_eval_parallel(models, datasets, instances, n_jobs=n_jobs)

    with output_path.open("wb") as handle:
        pickle.dump(logs, handle)
    manifest_path = write_manifest(output_path, config, models, datasets)
    logger.info("Wrote %s logs to %s (manifest %s)", len(logs), output_path, manifest_path)


if __name__ == "__main__":
    main()
