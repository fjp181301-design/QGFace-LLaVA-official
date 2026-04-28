import argparse
import json
import os
from datetime import datetime

import yaml


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def build_output_dir(config, dataset):
    output_cfg = config.get("output", {})
    save_dir = output_cfg.get("save_dir", f"outputs/{dataset}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def run_training(args, config):
    """
    Lightweight training entry for QGFace-LLaVA.

    This function is a minimal placeholder for the official training pipeline.
    The full implementation should include:
    1. dataset loading;
    2. metadata encoding;
    3. task-aware quality estimation;
    4. quality-aware controlled fusion;
    5. LLM-based task prediction;
    6. evaluation and result saving.
    """

    save_dir = build_output_dir(config, args.dataset)

    run_info = {
        "project": "QGFace-LLaVA",
        "dataset": args.dataset,
        "method": args.method,
        "metadata_setting": args.metadata_setting,
        "config": args.config,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "placeholder_training_completed",
        "note": (
            "This is a lightweight training entry. "
            "Please replace this placeholder with the full QGFace-LLaVA training pipeline."
        ),
    }

    summary_path = os.path.join(save_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    print("=" * 60)
    print("QGFace-LLaVA Training Entry")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Method: {args.method}")
    print(f"Metadata setting: {args.metadata_setting}")
    print(f"Config file: {args.config}")
    print(f"Output directory: {save_dir}")
    print(f"Run summary saved to: {summary_path}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training entry for QGFace-LLaVA."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["fer2013", "celeba40", "utkface"],
        help="Dataset name."
    )

    parser.add_argument(
        "--method",
        type=str,
        default="qgface_llava",
        choices=["base", "naive_fusion", "gate_only", "qgface_llava"],
        help="Metadata usage strategy."
    )

    parser.add_argument(
        "--metadata_setting",
        type=str,
        default="clean",
        choices=["clean", "noisy60", "missing", "shuffled"],
        help="Metadata condition."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    run_training(args, config)


if __name__ == "__main__":
    main()
