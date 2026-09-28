# Cat_Dog_classification

A reproducible TensorFlow project for binary cat/dog image classification. It includes a validated data pipeline, an ImageNet-pretrained MobileNetV2 CNN, command-line training and prediction, and tests. The first training run downloads the pretrained weights if they are not already cached.

## Requirements

- Windows 10/11 (64-bit) or Linux
- Python 3.10 or 3.11
- The image files referenced in `cat_dog.csv` (the CSV contains filenames and labels, not image data)

TensorFlow 2.15.1 is pinned because it supports Python 3.11 on native Windows CPU. For GPU acceleration, use a supported Linux/WSL2 TensorFlow environment.

## Set up on Windows

From the project root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name cat-dog-cnn --display-name "Python (cat-dog-cnn)"
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again. In VS Code, select the `.venv` interpreter and the `Python (cat-dog-cnn)` notebook kernel.

## Get the images

`cat_dog.csv` has 25,000 balanced records (`0` = cat, `1` = dog). The matching images in this workspace are in `archive (1)/cat_dog/`, which training discovers automatically. If you store images elsewhere, pass their directory with `--image-dir`; keep filenames exactly as listed in the CSV, for example `cat.0.jpg` and `dog.11289.jpg`.

The expected layout is:

```text
mnist-cat-dog-cnn/
├── cat_dog.csv
├── archive (1)/
│   └── cat_dog/
│       ├── cat.0.jpg
│       └── dog.11289.jpg
└── ...
```

Image data and trained models are excluded from Git by `.gitignore`. If the archive is not present in your copy, obtain the Dogs vs. Cats training images separately.

## Train

Start with a small balanced subset to verify the data and runtime:

```powershell
python -m cat_dog_cnn.train --sample-size 400 --epochs 1
```

Train on the full dataset with the defaults (30 epochs maximum, early stopping enabled):

```powershell
python -m cat_dog_cnn.train
```

The trainer validates CSV columns and labels, checks that all image files exist and open, and creates stratified 70/15/15 train/validation/test splits. The best model is saved to `models/cat_dog_cnn.keras`, with test metrics in `models/cat_dog_cnn_metrics.json`. Adjust `--batch-size`, `--image-size`, `--epochs`, and `--output` to fit your machine. Use `--skip-image-integrity-check` to skip the initial decode check for a known-good dataset.

## Predict

After training:

```powershell
python -m cat_dog_cnn.predict path\to\an-image.jpg
```

Specify a model saved to another location with `--model path\to\model.keras`.

## Tests

```powershell
python -m pytest
```

The tests use temporary images and do not require downloading the full dataset.

## Notebook

Open `cat_dog_cnn.ipynb` in Jupyter or VS Code and select the `Python (cat-dog-cnn)` kernel. It walks through dataset inspection, CNN training, evaluation, and a single-image prediction. It starts with a balanced 2,000-image sample and trains for up to 6 epochs; set `SAMPLE_SIZE = None` in the configuration cell to train on all images for best accuracy. With seed 42, this configuration achieved 96% accuracy and 0.996 ROC-AUC on its held-out 300-image test split; results may vary with different data or training settings.

## GitHub

The repository includes the source, tests, notebook, environment/dependency configuration, and labels CSV. Keep `.venv/`, image data, and generated model files out of commits; these are excluded by `.gitignore`.
