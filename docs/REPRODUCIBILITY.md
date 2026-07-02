# Reproducibility Notes

## Scope

This repository contains the PaddlePaddle/PaddleSeg implementation for the MRSN/MBFM experiments. The C2Seg-BW dataset is not redistributed here.

## Environment

Install repository dependencies:

```bash
pip install -r PaddleCD/requirements.txt
```

Install the PaddlePaddle GPU package that matches your CUDA toolkit. The experiments used an RTX 6000 GPU with CUDA 12.6.

## Dataset Configuration

Create a local `.env` file in the repository root before running:

```bash
C2SEG_BW_ROOT=/path/to/C2Seg_BW/train
```

`PaddleCD/c2seg_config/C2Seg_BW.yml` reads this variable:

```yaml
train_dataset:
  dataset_root: ${C2SEG_BW_ROOT}
  train_path: ${C2SEG_BW_ROOT}\train.txt
val_dataset:
  dataset_root: ${C2SEG_BW_ROOT}
  val_path: ${C2SEG_BW_ROOT}\val.txt
```

The experiments use 7,140 labeled 256 x 256 patches from C2Seg-BW, with 6,426 patches for training and 714 for validation. Keep the exact split files with the release if dataset rules allow it.

## Training Protocol

The shared config uses:

- batch size: 16
- iterations: 40,000
- seed: 1,919,810
- optimizer: AdamW
- learning rate: 0.0002
- scheduler: StepDecay, step size 5,000, gamma 0.5
- save interval: 800 iterations

Example:

```bash
cd PaddleCD
python train.py --config c2seg_config/cxup_4b_BW_PMRG_v2_lossV2.yml --save_dir ../output/cxup_4b_BW_PMRG_v2_lossV2 --do_eval
```

## Evaluation

```bash
cd PaddleCD
python val.py --config c2seg_config/cxup_4b_BW_PMRG_v2_lossV2.yml --model_path ../output/cxup_4b_BW_PMRG_v2_lossV2/best_model/model.pdparams --batch_size 1
```

## Reported Validation Results

| Configuration | mIoU | F1 | ACC | Kappa | Params (M) | FLOPs (G) | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1-branch (UPerNet ref.) | 0.8025 | 0.8878 | 0.9328 | 0.9094 | 30.01 | 31.70 | 143.93 |
| 2-branch | 0.8342 | 0.9079 | 0.9470 | 0.9286 | 58.50 | 51.71 | 104.47 |
| 3-branch | 0.8496 | 0.9173 | 0.9537 | 0.9377 | 87.00 | 71.73 | 85.27 |
| 4-branch | 0.8659 | 0.9269 | 0.9598 | 0.9459 | 115.49 | 91.76 | 49.96 |
| 4-branch + PMRG | 0.8671 | 0.9277 | 0.9602 | 0.9465 | 116.51 | 94.18 | 41.42 |
| 4-branch + ML | 0.8684 | 0.9281 | 0.9656 | 0.9537 | 115.49 | 91.76 | 51.61 |
| MBFM (4-branch + PMRG + ML) | 0.8694 | 0.9287 | 0.9658 | 0.9539 | 116.51 | 94.18 | 42.86 |

## Release Notes

- Dataset paths are read from a local `.env` file. Do not commit private dataset paths.
- The official C2Seg test labels are not public, so reported metrics are internal-validation metrics.
- For exact reproducibility, release the train/validation split files and model checkpoints if licensing and storage constraints permit.
