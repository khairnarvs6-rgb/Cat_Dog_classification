"""Command-line prediction entry point."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def predict_image(
    image_path: str | Path, model_path: str | Path
) -> tuple[str, float]:
    """Return the predicted class and confidence for one image."""
    image_path = Path(image_path)
    model_path = Path(model_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path.resolve()}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path.resolve()}")

    from tensorflow import keras

    model = keras.models.load_model(model_path)
    image_size = model.input_shape[1]
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((image_size, image_size))
        pixels = np.asarray(image, dtype=np.float32)
    probability_dog = float(model.predict(pixels[None, ...], verbose=0)[0][0])
    label = "dog" if probability_dog >= 0.5 else "cat"
    confidence = probability_dog if label == "dog" else 1.0 - probability_dog
    return label, confidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one cat/dog image.")
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--model", type=Path, default=Path("models/cat_dog_cnn.keras")
    )
    args = parser.parse_args()
    label, confidence = predict_image(args.image, args.model)
    print(f"{label} ({confidence:.1%} confidence)")


if __name__ == "__main__":
    main()
