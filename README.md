Frequency-Spatial Integrated Learning for Generalizable Deepfake Detection Across Unseen Generative Models

The Visual Computer | Hengyang Normal University

Note: This code is directly related to the manuscript submitted to The Visual Computer. If you use this code, please cite our paper (citation details will be updated upon acceptance).

bibtex
@article{he2025fsinet,
  title={Frequency-Spatial Integrated Learning for Generalizable Deepfake Detection Across Unseen Generative Models},
  author={He, Jia-Xin and Zhang, Yue-Tao and Zhao, Hui-Huang},
  journal={The Visual Computer},
  year={2025},
  note={Under review}
}


Overview

FSI-Net is a unified frequency-spatial integrated network for generalizable deepfake detection. It combines:

Frequency Enhancement Module (FEM): Average-pooling-based high-low frequency decomposition (HLFD) with DenseNet for high-frequency traces and U-Net for low-frequency contours, followed by channel attention fusion and multi-order gated aggregation (MOGA).
Multi-Scale Module (MSM): Multi-scale dilated depth-wise convolutions with channel attention and pixel attention for hierarchical spatial feature extraction.

Experiments on three diffusion benchmarks with 28 generative models show 96.9% mean accuracy and 99.6% mean AP.

Key Algorithms Implementation

表格
Module	File	Class	Description
HLFD (FEM)	networks/common.py	HLFD	AvgPool frequency decomposition + DenseNet + U-Net + Channel Attention
MOGA (FEM)	networks/common.py	MultiOrderGatedAggregation	Multi-order gated feature aggregation with dilated DWConv
MSM	networks/common.py	MixStructureBlock	Multi-scale dilated convolutions + Channel/Pixel Attention
ResNet Backbone	networks/resnet.py	ResNet	ResNet with NPR preprocessing + FEM + MSM integration

Getting Started

sh
pip install -r requirements.txt


Following CNNDetection, the training set uses ForenSynths with 4 categories (car, cat, chair, horse) from ProGAN. Testing is performed on DiffusionForensics, Ojha dataset, and Self-Synthesis dataset. Please refer to our paper for detailed dataset descriptions.

Training

sh
CUDA_VISIBLE_DEVICES=0 python train.py --name FSI-Net --dataroot ./datasets/ForenSynths --classes car,cat,chair,horse --batch_size 32 --delr_freq 10 --lr 0.0002 --niter 50


Testing

sh
CUDA_VISIBLE_DEVICES=0 python test.py --model_path ./checkpoints/FSI-Net/model_epoch_best.pth --batch_size {BS}


Data Availability

Source code: GitHub.

Acknowledgements

Built upon NPR-DeepfakeDetection (CVPR 2024) and CNNDetection (CVPR 2020).

License

MIT License — see LICENSE.
