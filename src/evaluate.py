import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error


def evaluate_fer2013(prediction_file):
    """
    Expected CSV format:
    y_true,y_pred
    happy,happy
    angry,sad
    """
    df = pd.read_csv(prediction_file)

    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError("FER2013 prediction file must contain 'y_true' and 'y_pred' columns.")

    acc = accuracy_score(df["y_true"], df["y_pred"])

    return {
        "dataset": "fer2013",
        "metric": "accuracy",
        "accuracy": float(acc)
    }


def evaluate_utkface(prediction_file):
    """
    Expected CSV format:
    y_true,y_pred
    25,26.3
    42,39.8
    """
    df = pd.read_csv(prediction_file)

    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError("UTKFace prediction file must contain 'y_true' and 'y_pred' columns.")

    mae = mean_absolute_error(df["y_true"], df["y_pred"])

    return {
        "dataset": "utkface",
        "metric": "MAE",
        "mae": float(mae)
    }


def evaluate_celeba40(prediction_file):
    """
    Expected CSV format:
    y_true_0,y_true_1,...,y_true_39,y_pred_0,y_pred_1,...,y_pred_39

    Each label should be encoded as 0 or 1.
    """
    df = pd.read_csv(prediction_file)

    true_cols = [c for c in df.columns if c.startswith("y_true_")]
    pred_cols = [c for c in df.columns if c.startswith("y_pred_")]

    true_cols = sorted(true_cols, key=lambda x: int(x.split("_")[-1]))
    pred_cols = sorted(pred_cols, key=lambda x: int(x.split("_")[-1]))

    if len(true_cols) == 0 or len(pred_cols) == 0:
        raise ValueError(
            "CelebA-40 prediction file must contain columns like "
            "'y_true_0 ... y_true_39' and 'y_pred_0 ... y_pred_39'."
        )

    if len(true_cols) != len(pred_cols):
        raise ValueError("The number of ground-truth columns and prediction columns must be the same.")

    y_true = df[true_cols].values
    y_pred = df[pred_cols].values

    attr_acc = (y_true == y_pred).mean(axis=0)
    macc = np.mean(attr_acc)

    return {
        "dataset": "celeba40",
        "metric": "mAcc",
        "mAcc": float(macc),
        "num_attributes": int(len(true_cols))
    }


def save_results(results, output_file):
    output_dir = os.path.dirname(output_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    for key, value in results.items():
        print(f"{key}: {value}")

    print("=" * 60)
    print(f"Results saved to: {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluation entry for QGFace-LLaVA."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["fer2013", "celeba40", "utkface"],
        help="Dataset name."
    )

    parser.add_argument(
        "--prediction_file",
        type=str,
        required=True,
        help="Path to the prediction CSV file."
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save the evaluation result JSON file."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.prediction_file):
        raise FileNotFoundError(f"Prediction file not found: {args.prediction_file}")

    if args.dataset == "fer2013":
        results = evaluate_fer2013(args.prediction_file)
    elif args.dataset == "celeba40":
        results = evaluate_celeba40(args.prediction_file)
    elif args.dataset == "utkface":
        results = evaluate_utkface(args.prediction_file)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    if args.output_file is None:
        output_file = os.path.join("outputs", args.dataset, "evaluation_results.json")
    else:
        output_file = args.output_file

    save_results(results, output_file)


if __name__ == "__main__":
    main()
