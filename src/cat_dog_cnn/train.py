"""Command-line training entry point."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from cat_dog_cnn.data import (
    load_metadata,
    make_dataset,
    resolve_image_dir,
    split_metadata,
    validate_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CNN to classify cat and dog images."
    )
    parser.add_argument("--csv", type=Path, default=Path("cat_dog.csv"))
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=resolve_image_dir(),
        help="Directory holding the training images; defaults to the first valid standard dataset location.",
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Use a balanced subset for a quick run (must be an even number).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/cat_dog_cnn.keras"),
        help="Output path for the best model.",
    )
    parser.add_argument(
        "--skip-image-integrity-check",
        action="store_true",
        help="Check that images exist but skip opening each file before training.",
    )
    args = parser.parse_args()
    if args.image_size <= 0 or args.batch_size <= 0 or args.epochs <= 0:
        parser.error("--image-size, --batch-size, and --epochs must be positive.")
    if args.sample_size is not None and (
        args.sample_size < 4 or args.sample_size % 2 != 0
    ):
        parser.error("--sample-size must be an even number of at least 4.")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    metadata = load_metadata(args.csv, args.image_dir)
    validate_images(
        metadata, check_integrity=not args.skip_image_integrity_check
    )

    if args.sample_size is not None:
        class_counts = metadata["labels"].value_counts()
        per_class = args.sample_size // 2
        if any(class_counts.get(label, 0) < per_class for label in (0, 1)):
            raise ValueError(
                f"--sample-size {args.sample_size} requires at least "
                f"{per_class:,} images from each class."
            )
        metadata = (
            metadata.groupby("labels", group_keys=False)
            .sample(n=per_class, random_state=args.seed)
            .reset_index(drop=True)
        )

    import tensorflow as tf

    from cat_dog_cnn.model import build_model

    tf.random.set_seed(args.seed)
    train_frame, validation_frame, test_frame = split_metadata(metadata, args.seed)
    print(
        f"Images: {len(metadata):,} total | "
        f"{len(train_frame):,} train | {len(validation_frame):,} validation | "
        f"{len(test_frame):,} test"
    )

    train_dataset = make_dataset(
        train_frame, args.image_size, args.batch_size, training=True, seed=args.seed
    )
    validation_dataset = make_dataset(
        validation_frame, args.image_size, args.batch_size
    )
    test_dataset = make_dataset(test_frame, args.image_size, args.batch_size)
    model = build_model(args.image_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(args.output), monitor="val_auc", mode="max", save_best_only=True
        ),
    ]
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    probabilities = model.predict(test_dataset, verbose=0).ravel()
    actual = test_frame["labels"].to_numpy()
    predicted = (probabilities >= 0.5).astype("int32")
    metrics = {
        "test_accuracy": float(accuracy_score(actual, predicted)),
        "test_roc_auc": float(roc_auc_score(actual, probabilities)),
    }
    report = classification_report(
        actual, predicted, target_names=["cat", "dog"], output_dict=True, zero_division=0
    )
    metrics["classification_report"] = report
    metrics_path = args.output.with_name(f"{args.output.stem}_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"Test accuracy: {metrics['test_accuracy']:.4f} | "
        f"ROC-AUC: {metrics['test_roc_auc']:.4f}"
    )
    print(f"Saved model: {args.output}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
