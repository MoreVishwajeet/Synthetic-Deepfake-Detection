# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a deepfake image detection system using multiple CNN/ViT-family backbones: **EfficientNet-B3**, **FasterViT-2-224**, and **EfficientFormerV2-S1**. The project uses PyTorch and `timm` for model implementations, with a unified orchestration system for training and inference across all models.

## Development Commands

### Setup
```powershell
pip install -r requirements.txt
```
Requires Python 3.12+

### Linting
```powershell
ruff check .
```

Ruff configuration is in `ruff.toml` with target Python 3.12. The project uses:
- Line length: ignored (E501)
- Quote style: double quotes
- Indent: spaces
- Selected rules: E, F, I, UP, B, W

### Testing
```powershell
# Run all tests
pytest

# Run only smoke tests
pytest -q -k "smoke"
```

Tests are minimal; only `tests/test_repo_smoke.py` exists for basic structure validation.

### Training Models
```powershell
# Train all selected models from config/train.yaml
python train.py

# Train with custom config
python train.py --config path/to/custom.yaml
```

Training runs through `orchestrator.py`, which:
1. Reads `config/train.yaml` (or specified config)
2. For each model in `selection:`, creates timestamped run directories under `runs/{model_name}/{timestamp}/`
3. Invokes the model-specific trainer from `trainers/` directory
4. Saves checkpoints, logs, and plots to the run directory

### Running Inference
```powershell
# Evaluate all selected models from config/inference.yaml
python inference.py

# Inference with custom config
python inference.py --config path/to/custom.yaml
```

Inference generates:
- Accuracy metrics in `logs/metrics.jsonl`
- Confusion matrix in `plots/confusion_matrix.png`
- ROC curve in `plots/roc_curve.png` (binary classification only)

### Gradio Web UI
```powershell
python main.py
```

Launches a web interface for real-time deepfake detection with Grad-CAM visualizations. Expects model weights in `weights/` directory:
- `weights/EfficientNetModel.pth`
- `weights/FasterVitModel.pth`
- `weights/EfficientFormerV2_S1.pth`

## Architecture

### Orchestration System

The project uses a **config-driven orchestration pattern** that separates concerns between infrastructure (orchestrator) and model-specific training logic (trainers).

**Key Components:**

1. **orchestrator.py** - Central dispatcher that:
   - Parses YAML configs (`config/train.yaml`, `config/inference.yaml`)
   - Sets up run directories and checkpointing
   - Injects environment variables to configure trainers
   - Routes execution to model-specific trainers via `importlib`
   - Handles inference evaluation with metrics and visualization

2. **model_registry.py** - Model metadata registry:
   - Maps model names to their trainer modules (e.g., `efficientnet_b3` → `trainers.efficientnet`)
   - Provides builder functions for instantiating models during inference
   - Specifies default image sizes and weights keys per model

3. **train_env.py** - Shared training utilities:
   - Resolves environment variables (DD_OUTPUT_DIR, DD_SEED, DD_BATCH_SIZE, etc.)
   - Manages checkpoint saving/loading (best.ckpt, latest.ckpt)
   - Handles transform toggles via DD_TRANSFORMS JSON override
   - Provides seeding, logging, and console helpers for trainers

4. **trainers/** - Model-specific training scripts:
   - `trainers/efficientnet.py` - EfficientNet-B3 with two-phase training (head warmup + full fine-tune)
   - `trainers/fastervit.py` - FasterViT-2-224 training
   - `trainers/efficientformer_v2.py` - EfficientFormerV2-S1 training
   - Each trainer exposes a `main()` function and reads config from environment variables

### Configuration Structure

YAML configs follow this pattern:
```yaml
seed: 1                    # Global random seed
device: cuda               # Device (cuda/cpu)

data:
  root: data/mini_deepfake_dataset
  train_split: train       # ImageFolder subdirectory name
  val_split: val
  test_split: test
  num_classes: 2
  img_size: 224

models:
  efficientnet_b3:
    output_dir: runs/efficientnet_b3
    transforms:              # Per-model transform toggles
      train:
        ensure_rgb: true
        train_random_resized_crop: true
        train_random_horizontal_flip: true
      eval:
        ensure_rgb: true
        val_resize: true
        val_center_crop: true
    training:                # Training-specific settings
      epochs: 5
      batch_size: 64
      num_workers: 4
      resume: auto           # Auto-resume from latest.ckpt
    inference:               # Inference-specific settings
      weights: weights/EfficientNet_10_21_2025.pth
      split: test
      batch_size: 256
      num_workers: 2

selection:                   # Models to run
  - efficientnet_b3
```

### Data Pipeline

- **Training:** Uses ImageFolder datasets with configurable per-model transforms
- **Expected structure:** `{DATA_ROOT}/{split}/{class_name}/image.jpg`
- **Classes:** Typically `Real` and `Fake` for binary classification (configurable via num_classes)
- **Transform toggles:** Every transform can be enabled/disabled per model in YAML, allowing fine-grained control (e.g., disable normalization for EfficientFormerV2-S1)

### Inference Flow

1. Load model architecture via `model_registry.py` builder
2. Load weights from specified path (auto-downloads from GitHub Releases if missing)
3. Apply eval transforms (configured per model)
4. Run forward pass on test split
5. Compute metrics (accuracy, ROC-AUC for binary classification)
6. Generate plots (confusion matrix, ROC curve)
7. Save results to `logs/metrics.jsonl`

### Checkpointing Strategy

- **best.ckpt** - Full checkpoint (model + optimizer + scheduler + epoch) for best validation accuracy
- **latest.ckpt** - Most recent epoch checkpoint for auto-resume
- **{ModelName}.pth** - Best model weights only (no optimizer state) in run root directory
- Auto-resume enabled via `resume: auto` in config (reads DD_RESUME_AUTO=1 env var)

## Adding New Models

To add a new model:

1. Create `trainers/new_model.py` with a `main()` function that reads env vars from `train_env`
2. Add model spec to `model_registry.py`:
   - Exact match in `_EXACT_SPECS` or prefix match in `_PREFIX_SPECS`
   - Provide `train_module`, `builder`, and `default_image_size`
3. Add model config to YAML files under `models:` key
4. Add model name to `selection:` list to enable it

## Environment Variables Used by Trainers

Set by orchestrator, read by trainer scripts:
- **DD_OUTPUT_DIR** - Run directory for checkpoints/logs
- **DD_SEED** - Random seed
- **DD_DEVICE** - Device override (cuda/cpu)
- **DD_DATA_ROOT** - Dataset root path
- **DD_TRAIN_SPLIT** / **DD_VAL_SPLIT** / **DD_TEST_SPLIT** - Split names
- **DD_IMG_SIZE** - Image size
- **DD_NUM_CLASSES** - Number of classes
- **DD_BATCH_SIZE** - Batch size
- **DD_EPOCHS** - Number of epochs
- **DD_NUM_WORKERS** - DataLoader workers
- **DD_RESUME_AUTO** - 1 to auto-resume from latest.ckpt
- **DD_TRANSFORMS** - JSON string of transform toggles
- **DD_LOG_PATH** - Path for console output mirroring

## Windows Considerations

This codebase is developed on Windows (PowerShell). Commands in this guide use PowerShell syntax. For cross-platform compatibility:
- File paths use `pathlib.Path` throughout
- Line endings configured as LF in ruff.toml but Git handles conversion
- requirements.txt may use UTF-16LE encoding (use appropriate encoding when reading)
