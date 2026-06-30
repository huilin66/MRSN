# MRSN

Code and manuscript release package for **MRSN / MBFM**, a multimodal remote sensing semantic segmentation project built on PaddlePaddle and PaddleSeg.

The current manuscript title is **A Multi-Branch Fusion Model for Multimodal Remote Sensing Image Segmentation**. The proposed MBFM combines a multi-branch ConvNeXt-Tiny/UPerNet backbone, a Pixel-wise Modality Reliability Gate (PMRG), and an OHEM-enhanced mixed loss for C2Seg-BW semantic segmentation.

## Repository Layout

```text
PaddleCD/                 PaddleSeg-based training, validation, prediction code
PaddleCD/c2seg_config/    Experiment configs for MRSN/MBFM variants and baselines
tools/                    Analysis, visualization, split, and post-processing scripts
pic/                      Repository-level architecture image
manuscript/               Clean LaTeX manuscript package copied from source_version
docs/                     Publication and reproducibility notes
```

## Main Results

All metrics below are reported on the internal C2Seg-BW validation split described in the manuscript.

| Method | mIoU | F1 | ACC | Kappa | Params (M) | FLOPs (G) | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| UPerNet / 1-branch reference | 0.8025 | 0.8878 | 0.9328 | 0.9094 | 30.01 | 31.70 | 143.93 |
| Prior MRSN (4B2H) | 0.8659 | 0.9269 | 0.9595 | 0.9455 | 116.82 | 102.67 | 49.46 |
| MBFM (4-branch + PMRG + ML) | 0.8694 | 0.9287 | 0.9658 | 0.9539 | 116.51 | 94.18 | 42.86 |

## Environment

The code is based on PaddlePaddle/PaddleSeg. A minimal dependency list is available in:

```bash
pip install -r PaddleCD/requirements.txt
```

Install a PaddlePaddle GPU build that matches your CUDA environment. The manuscript experiments used an RTX 6000 GPU with CUDA 12.6.

## Data

The experiments use C2Seg-BW from the 2023 IEEE WHISPERS Cross-City Semantic Segmentation Challenge. The dataset is not redistributed in this repository. Update the paths in `PaddleCD/c2seg_config/C2Seg_BW.yml` to point to your local copy:

```yaml
train_dataset:
  dataset_root: /path/to/C2Seg_BW/train/
  train_path: /path/to/C2Seg_BW/train.txt
val_dataset:
  dataset_root: /path/to/C2Seg_BW/train/
  val_path: /path/to/C2Seg_BW/val.txt
```

## Training

Example command for the full MBFM-style PMRG + mixed-loss variant:

```bash
cd PaddleCD
python train.py --config c2seg_config/cxup_4b_BW_PMRG_v2_loss.yml --save_dir ../output/mbfm --do_eval
```

Other important configs include:

| Config | Purpose |
|---|---|
| `c2seg_config/cxup_1b_BW.yml` | 1-branch stacked-input reference |
| `c2seg_config/cxup_2b_BW.yml` | 2-branch partition |
| `c2seg_config/cxup_3b_BW.yml` | 3-branch partition |
| `c2seg_config/cxup_4b_BW.yml` | 4-branch reference |
| `c2seg_config/cxup_4b_BW_PMRG_v2_loss.yml` | PMRG + OHEM-enhanced mixed loss |
| `c2seg_config/MRSN.yml` | Prior 4-branch/two-head MRSN prototype |

## Evaluation

```bash
cd PaddleCD
python val.py --config c2seg_config/cxup_4b_BW_PMRG_v2_loss.yml --model_path ../output/mbfm/best_model/model.pdparams --batch_size 1
```

Additional analysis scripts for CAM, t-SNE, per-image mIoU, and summary tables are in `tools/`. See `docs/REPRODUCIBILITY.md` for a more complete release checklist.

## Manuscript

The cleaned LaTeX submission package is in `manuscript/`:

- `main.tex`
- `ref.bib`
- `figures/`
- `main.pdf`
- `cover_letter.md`

Before journal submission, review `docs/PUBLICATION_CHECKLIST.md` for unresolved publication items such as target-journal template confirmation, final citation verification, license choice, and dataset/code availability details.

## Citation

If you use this repository, please cite the project metadata in `CITATION.cff`. After the manuscript is published, update `CITATION.cff` with the final venue and DOI.

## License

This repository has not yet declared a project license. Add a `LICENSE` file before public release. Because `PaddleCD` adapts PaddleSeg code, verify compatibility with the Apache License 2.0 notices retained in the source files.
