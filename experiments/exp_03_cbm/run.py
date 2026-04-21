from pathlib import Path

from experiments.utils import load_config, save_json
from models.train import run_cbm_pipeline


def main() -> None:
    config_path = Path(__file__).with_name("config.yaml")
    config = load_config(config_path)
    metrics = run_cbm_pipeline(config)

    output_dir = Path(config.get("output", "experiments/results/exp_03_cbm"))
    save_json(
        {
            "experiment": "exp_03_cbm",
            "config": config,
            "metrics": metrics,
        },
        output_dir / "metrics.json",
    )
    print(f"Saved CBM metrics in {output_dir}")


if __name__ == "__main__":
    main()
