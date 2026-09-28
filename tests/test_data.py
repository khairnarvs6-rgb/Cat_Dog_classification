import pandas as pd
import pytest
from PIL import Image

from cat_dog_cnn.data import (
    load_metadata,
    resolve_image_dir,
    split_metadata,
    validate_images,
)


def test_load_metadata_validates_and_resolves_paths(tmp_path):
    csv_path = tmp_path / "labels.csv"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (8, 8)).save(image_dir / "cat.jpg")
    pd.DataFrame({"image": ["cat.jpg"], "labels": [0]}).to_csv(csv_path, index=False)

    frame = load_metadata(csv_path, image_dir)
    validated = validate_images(frame)

    assert validated.loc[0, "labels"] == 0
    assert validated.loc[0, "path"] == str(image_dir / "cat.jpg")


def test_load_metadata_rejects_non_binary_labels(tmp_path):
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({"image": ["cat.jpg"], "labels": [2]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="0 \\(cat\\) or 1 \\(dog\\)"):
        load_metadata(csv_path, tmp_path)


def test_validate_images_reports_missing_files(tmp_path):
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({"image": ["missing.jpg"], "labels": [0]}).to_csv(
        csv_path, index=False
    )

    with pytest.raises(FileNotFoundError, match="Check --image-dir"):
        validate_images(load_metadata(csv_path, tmp_path))


def test_resolve_image_dir_finds_standard_dataset_locations(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "data" / "train").mkdir(parents=True)
    (project / "data" / "train" / "cat.0.jpg").write_bytes(b"fake")
    monkeypatch.chdir(project)

    resolved = resolve_image_dir()

    assert resolved == project / "data" / "train"


def test_resolve_image_dir_finds_supplied_archive_layout(tmp_path, monkeypatch):
    project = tmp_path / "project"
    archive_images = project / "archive (1)" / "cat_dog"
    archive_images.mkdir(parents=True)
    (archive_images / "cat.0.jpg").write_bytes(b"fake")
    monkeypatch.chdir(project)

    resolved = resolve_image_dir()

    assert resolved == archive_images


def test_split_metadata_is_stratified_and_preserves_rows():
    frame = pd.DataFrame(
        {
            "image": [f"{label}-{index}.jpg" for label in (0, 1) for index in range(20)],
            "labels": [label for label in (0, 1) for _ in range(20)],
            "path": [f"images/{label}-{index}.jpg" for label in (0, 1) for index in range(20)],
        }
    )

    train, validation, test = split_metadata(frame)

    assert len(train) + len(validation) + len(test) == len(frame)
    assert all(part["labels"].mean() == 0.5 for part in (train, validation, test))
