# Frequency-Spatial Integrated Learning for Generalizable Deepfake Detection Across Unseen Generative Models

**[The Visual Computer](https://www.springer.com/journal/371)** | Hengyang Normal University

> **Note:** This code is directly related to the manuscript submitted to *The Visual Computer*. If you use this code, please cite our paper (citation details will be updated upon acceptance).

```bibtex
@article{he2025fsinet,
  title={Frequency-Spatial Integrated Learning for Generalizable Deepfake Detection Across Unseen Generative Models},
  author={He, Jia-Xin and Zhang, Yue-Tao and Zhao, Hui-Huang},
  journal={The Visual Computer},
  year={2025},
  note={Under review}
}
```

---

## Overview

FSI-Net is a unified frequency-spatial integrated network for generalizable deepfake detection. It combines:

- **Frequency Enhancement Module (FEM):** Multi-branch dilated convolutions to separate high- and low-frequency traces.
- **Feature Mapping Stage (FMS):** Haar wavelet decomposition with global and local multi-scale attention.

Experiments on three diffusion benchmarks with 28 generative models show **96.9% mean accuracy** and **99.6% mean AP**.

---

## Key Algorithms Implementation

| Module | File | Class | Description |
|--------|------|-------|-------------|
| FEM | `networks/common.py` | `FEM` | Multi-branch dilated convolutions with different receptive fields |
| FMS | `networks/common.py` | `FMS` | Haar wavelet + global attention + local multi-scale attention |
| ResNet Backbone | `networks/resnet.py` | `ResNet` | ResNet with NPR preprocessing + FEM + FMS integration |

---

## Getting Started

```sh
pip install -r requirements.txt
```

Following [CNNDetection](https://github.com/PeterWang512/CNNDetection), the training set uses ForenSynths with 4 categories (car, cat, chair, horse) from ProGAN. Testing is performed on DiffusionForensics, Ojha dataset, and Self-Synthesis dataset. Please refer to our paper for detailed dataset descriptions.

### Training

```sh
CUDA_VISIBLE_DEVICES=0 python train.py --name FSI-Net --dataroot ./datasets/ForenSynths --classes car,cat,chair,horse --batch_size 32 --delr_freq 10 --lr 0.0002 --niter 50
```

### Testing

```sh
CUDA_VISIBLE_DEVICES=0 python test.py --model_path ./checkpoints/FSI-Net/model_epoch_best.pth --batch_size {BS}
```

---

## Data Availability

Source code: [GitHub](https://github.com/hejiaxin-2002/FSI-Net-Generalizable-Deepfake-Detection). A DOI-tagged version is archived on Zenodo: [DOI will be added upon release].

**Pre-trained models will be released upon acceptance.**

---

## Acknowledgements

Built upon [NPR-DeepfakeDetection](https://github.com/chuangchuangtan/NPR-DeepfakeDetection) (CVPR 2024) and [CNNDetection](https://github.com/peterwang512/CNNDetection) (CVPR 2020).

## License

MIT License — see [LICENSE](LICENSE).
