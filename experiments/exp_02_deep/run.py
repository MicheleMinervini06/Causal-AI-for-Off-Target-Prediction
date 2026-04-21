from pathlib import Path

from experiments.utils import load_config, save_json
from models.train import run_deep_pipeline


def main() -> None:
    config_path = Path(__file__).with_name("config.yaml")
    config = load_config(config_path)
    metrics = run_deep_pipeline(config)

    output_dir = Path(config.get("output", "experiments/results/exp_02_deep"))
    save_json(
        {
            "experiment": "exp_02_deep",
            "config": config,
            "metrics": metrics,
        },
        output_dir / "metrics.json",
    )
    print(f"Saved deep metrics in {output_dir}")


if __name__ == "__main__":
    main()
