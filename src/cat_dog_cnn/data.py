"""Dataset validation, splitting, and TensorFlow input pipelines."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split


def resolve_image_dir(
    image_dir: str | Path | None = None,
    csv_path: str | Path | None = None,
) -> Path:
    """Return the best available dataset folder, falling back to common project layouts."""
    candidates: list[Path] = []

    if image_dir is not None:
        candidates.append(Path(image_dir).expanduser())

    project_root = Path.cwd()
    if csv_path is not None:
        project_root = Path(csv_path).resolve().parent

    candidates.extend(
        [
            project_root / "data" / "train",
            project_root / "archive (1)" / "cat_dog",
            project_root / "train",
            project_root / "data",
            project_root,
            Path("data") / "train",
            Path("train"),
            Path("data"),
            Path("."),
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.expanduser().resolve(strict=False)
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized.is_dir():
            return normalized

    if image_dir is not None:
        return Path(image_dir).expanduser().resolve(strict=False)
    return (project_root / "data" / "train").resolve(strict=False)


def load_metadata(csv_path: str | Path, image_dir: str | Path) -> pd.DataFrame:
    """Load labels, validate their schema, and resolve image paths."""
    csv_path = Path(csv_path)
    image_dir = resolve_image_dir(image_dir, csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Labels CSV not found: {csv_path.resolve()}")

    frame = pd.read_csv(csv_path)
    required_columns = {"image", "labels"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{csv_path} is missing required column(s): {', '.join(sorted(missing_columns))}"
        )
    if frame.empty:
        raise ValueError(f"No labeled images found in {csv_path}.")
    if frame[["image", "labels"]].isna().any().any():
        raise ValueError("The image and labels columns cannot contain empty values.")
    if frame["image"].duplicated().any():
        raise ValueError("The labels CSV contains duplicate image filenames.")

    try:
        labels = pd.to_numeric(frame["labels"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Labels must be numeric: 0 for cat or 1 for dog.") from exc
    if not labels.isin([0, 1]).all():
        raise ValueError("Labels must be 0 (cat) or 1 (dog).")

    frame = frame.copy()
    frame["labels"] = labels.astype("int32")
    frame["path"] = frame["image"].map(lambda name: str(image_dir / str(name)))
    return frame


def validate_images(frame: pd.DataFrame, check_integrity: bool = True) -> pd.DataFrame:
    """Require every referenced image and optionally verify it can be decoded."""
    missing = [path for path in frame["path"] if not Path(path).is_file()]
    if missing:
        examples = "\n".join(f"  - {path}" for path in missing[:5])
        remaining = len(missing) - min(len(missing), 5)
        suffix = f"\n  ... and {remaining} more" if remaining else ""
        raise FileNotFoundError(
            f"{len(missing):,} image file(s) referenced by the CSV were not found. "
            f"Check --image-dir.\n{examples}{suffix}"
        )

    if not check_integrity:
        return frame

    invalid = []
    for path in frame["path"]:
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError):
            invalid.append(path)
    if invalid:
        examples = "\n".join(f"  - {path}" for path in invalid[:5])
        raise ValueError(
            f"{len(invalid):,} image file(s) are unreadable or corrupt:\n{examples}"
        )
    return frame


def split_metadata(
    frame: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create deterministic, stratified 70/15/15 train/validation/test splits."""
    train_frame, remainder = train_test_split(
        frame, test_size=0.30, stratify=frame["labels"], random_state=seed
    )
    validation_frame, test_frame = train_test_split(
        remainder,
        test_size=0.50,
        stratify=remainder["labels"],
        random_state=seed,
    )
    return (
        train_frame.reset_index(drop=True),
        validation_frame.reset_index(drop=True),
        test_frame.reset_index(drop=True),
    )


def make_dataset(
    frame: pd.DataFrame,
    image_size: int = 128,
    batch_size: int = 32,
    training: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """Build a batched, prefetched image dataset from metadata."""
    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices(
        (frame["path"].to_numpy(), frame["labels"].to_numpy())
    )
    if training:
        dataset = dataset.shuffle(
            len(frame), seed=seed, reshuffle_each_iteration=True
        )

    def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, (image_size, image_size))
        return image, tf.cast(label, tf.float32)

    return (
        dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
