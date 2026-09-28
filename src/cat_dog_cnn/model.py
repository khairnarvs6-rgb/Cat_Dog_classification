"""CNN architecture for cat-versus-dog classification."""

from tensorflow import keras
from tensorflow.keras import layers


def build_model(image_size: int = 128) -> keras.Model:
    """Build and compile a transfer-learning CNN for binary classification."""
    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)

    base_model = keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="cat_dog_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model
