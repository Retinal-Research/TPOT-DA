# TPOT-DA: Topology-Preserving with Domain-Adaptive Enhancement of Real-World Non-mydriatic Color Fundus Photographs

In real screening data, fundus images are often degraded (blur, glare, uneven illumination, sensor noise). Standard enhancement can improve appearance but may break vessel continuity or introduce artifacts. TPOT-DA is designed to preserve anatomical structure during enhancement and remain robust across domains.

TPOT-DA is a retinal fundus image enhancement framework designed for real-world, non-mydriatic clinical data where domain shift and image degradation are common. The method combines topology-preserving optimal transport (TPOT) with parameter-efficient domain adaptation: a pretrained enhancement backbone is structurally constrained to preserve vessel connectivity, while lightweight residual adapters are fine-tuned for target-domain noise and color characteristics. This design improves image gradability without sacrificing anatomical fidelity required by downstream vascular analysis. 
TPOT-DA reports strong open-domain gains, including conversion-rate improvements on real clinic cohorts, indicating practical value for heterogeneous screening environments.


## Overview

TPOT-DA is a PyTorch framework for retinal image enhancement under domain shift.
It combines:

- `TPOT`: topology-preserving optimal transport enhancement
- `DA adapters`: parameter-efficient residual adapters for target-domain adaptation

The goal is to improve image gradability while preserving vessel topology required by downstream vascular biomarkers.

## Key Contributions

- Topology-preserving enhancement via persistent-homology-based structural regularization
- Parameter-efficient domain adaptation by freezing the backbone and tuning lightweight adapters
- Clinical-oriented evaluation on heterogeneous datasets (EyeQ, UK Biobank, MobileLab)

## Project Structure

```text
TPOT-DA/
├─ dataloader/            # Dataset loaders
├─ dataset/               # User-provided datasets
├─ model/                 # Generator + adapter modules
├─ segmentation/          # Mask/segmentation modules
├─ Helper/                # Utility metrics (e.g., SSIM)
├─ train_full.py          # Full model training
├─ train_adapter.py       # Adapter fine-tuning
├─ train_topo.py          # Topology-aware training
├─ test.py                # Inference + PSNR/SSIM
├─ retrieve_model.py      # Checkpoint selection/evaluation helper
├─ csv_create.py          # CSV helper
└─ requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Data Format

For adapter training/evaluation, prepare paired filename sets under:

```text
<root>/
├─ de_image/
└─ good_quality/
```

CSV files should contain one column named `image_name`.

- `--csv_bad`: degraded image list
- `--csv_good`: high-quality image list
- `--csv_val`: validation image list
- `--csv_test`: test image list

## Quick Start

### 1) Adapter Fine-tuning

```bash
python train_adapter.py \
  --checkpoints <pretrained_generator.pth> \
  --root <dataset_root> \
  --csv_bad <bad.csv> \
  --csv_good <good.csv> \
  --csv_val <val.csv> \
  --save_dir <output_dir>
```

### 2) Full Training

```bash
python train_full.py \
  --batchSize 12 \
  --nEpochs 200 \
  --lr 1e-4 \
  --root dataset/degratation/pre \
  --file_dir dataset/train.csv
```

### 3) Inference + Metrics

```bash
python test.py \
  --checkpoints <model.pth> \
  --root <dataset_root> \
  --csv_test <test.csv> \
  --save <result_dir> \
  --save_dir <metric_dir> \
  --metrics_name metrics.txt
```

## Notes

- Some scripts include legacy hardcoded paths (for example `SottGan/...`); adjust before running.
- Checkpoint formats differ by script (`state_dict` vs `checkpoint["model"]`).
- CUDA is strongly recommended for training.


## Acknowledgements
This work was supported by grants from the National Institutes of Health (RF1AG073424, R01EY032125) and the State of Arizona via the Arizona Alzheimer Consortium.

## Citation
If you find this project helpful to your research, please consider citing [BibTeX]:
```bibtex
@inproceedings{dong2025tpot,
  title={Tpot: Topology preserving optimal transport in retinal fundus image enhancement},
  author={Dong, Xuanzhao and Zhu, Wenhui and Li, Xin and Sun, Guoxin and Su, Yi and Dumitrascu, Oana M and Wang, Yalin},
  booktitle={2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```

## License
[Onhold]

