"""
Labelfree CLI - Command line interface for fluorescence prediction.

Usage:
    labelfree train -c config.yaml
    labelfree predict -c config.yaml --checkpoint model.ckpt
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="labelfree",
        description="Neural network for predicting fluorescence from phase contrast",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run prediction on images")
    predict_parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    predict_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    predict_parser.add_argument(
        "--output",
        type=str,
        default="predictions",
        help="Output directory for predictions",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "train":
        from labelfree.commands.train import run_train
        return run_train(args.config, args.resume)

    elif args.command == "predict":
        from labelfree.commands.predict import run_predict
        return run_predict(args.config, args.checkpoint, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
