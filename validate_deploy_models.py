#!/usr/bin/env python3
"""
Evaluate all saved models under deploy_models/ on the hidden test set and write
metrics plus plots under validation_outputs/ (by default).

Does not train. Use after ``python train_and_validate.py`` or notebook Section 2.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import train_and_validate as tv


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate deploy_models on test_data_hidden.csv")
    parser.add_argument(
        "--deploy-dir",
        type=Path,
        default=None,
        help="Directory with sentiment_model_*.pkl (default: deploy_models/)",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=None,
        help="Labeled test CSV (default: Ecommerce_dataset/test_data_hidden.csv)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Plots and metrics_summary.csv (default: validation_outputs/)",
    )
    args = parser.parse_args()

    os.chdir(tv.ROOT)
    tv.validate_deploy_models_only(
        test_csv=args.test_csv,
        deploy_dir=args.deploy_dir,
        plot_output_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
