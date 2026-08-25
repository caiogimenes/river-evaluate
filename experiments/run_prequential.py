"""CLI for the prequential AQO evaluation experiment."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow `python experiments/run_prequential.py` without installing the package.
_BOOTSTRAP_ROOT = Path(__file__).resolve().parent.parent
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from src.evaluation import run_prequential_eval_parallel  # noqa: E402
from src.experiment.artifacts import write_logs, write_manifest  # noqa: E402
from src.experiment.config import (  # noqa: E402
    apply_overrides,
    load_config,
    parse_args,
    resolve_config_path,
    seed_everything,
)
from src.experiment.datasets import build_datasets  # noqa: E402
from src.logging_setup import configure_logging  # noqa: E402
from src.models import get_models  # noqa: E402
from src.paths import resolve_repo_path  # noqa: E402

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)
    config_path = resolve_config_path(args.config)
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

    write_logs(output_path, logs)
    manifest_path = write_manifest(output_path, config, models, datasets)
    logger.info("Wrote %s logs to %s (manifest %s)", len(logs), output_path, manifest_path)


if __name__ == "__main__":
    main()
