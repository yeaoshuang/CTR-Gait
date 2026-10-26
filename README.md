# CTR-Gait: Semantic-Aware Contrastive Learning for Fine-Grained Gait Emotion Recognition

<!-- `https://img.shields.io/badge/License-MIT-yellow.svg` ](https://opensource.org/licenses/MIT) -->
<!-- `https://img.shields.io/badge/PyTorch-1.8%2B-orange` ]( `https://pytorch.org/)`  -->

This repository contains the official PyTorch implementation of the paper: **"CTR-Gait: Semantic-Aware Contrastive Learning for Fine-Grained Gait Emotion Recognition"**. The paper is available as an IEEE Transactions on Affective Computing preprint, DOI: [10.1109/TAFFC.2026.3723033](https://doi.org/10.1109/TAFFC.2026.3723033).

## 🚀 Introduction

CTR-Gait is a lightweight yet robust framework designed to capture subtle affective cues from skeletal gait sequences. It addresses the challenge of distinguishing fine-grained emotions (e.g., Sadness vs. Neutral) through semantic-aware contrastive learning and topological refinement.

**Key Achievements:**
- **State-of-the-Art Accuracy:** Achieves **92.66%** on the Emotion-Gait benchmark.
- **Efficiency:** Ultra-lightweight design with **0.39 GFLOPs** and **3.76 M parameters**.
- **Robustness:** Validated on real-world UAV-captured datasets with high stability.

**Core Components:**
- **CTR-Graph**: Channel-wise Topology Refined Graph Convolutions to capture dynamic spatial correlations.
- **ESCL**: Emotion-Semantic Contrastive Learning with learnable Valence-Arousal centers.
- **SadExpert**: A specialized expert module to refine predictions for the minority "Sad" class.

![CTR-Gait Architecture](tools/fig_2.png)
*(Figure: Overview of the CTR-Gait Framework)*

## 📂 Project Structure

```text
CTR-Gait/
├── configs/                 # Configuration files for experiments
│   ├── bone_escl.yaml       # Main config for Bone modality + ESCL
│   └── ...
├── data/                    # Data processing scripts
│   └── generate_bpm_complete_data.py
├── src/                     # Source code
│   ├── dataset/             # Data loaders
│   ├── model/               # Model definitions (CTR-GCN, Transformer, etc.)
│   └── processor.py         # Training and testing loops
├── weights/                 # Directory to save/load pre-trained models
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
└── README.md
```

## 🛠️ Installation

Clone the repository:

```bash
git clone [https://github.com/yeaoshuang/CTR-Gait]
cd CTR-Gait
```

Install dependencies:

Python >= 3.8

PyTorch >= 1.8

```bash
pip install -r requirements.txt
```

## 💾 Data Preparation

### 1. Download Dataset

Please download the Emotion-Gait dataset from [\[INSERT DATASET LINK HERE\]](https://github.com/exped1230/BPM-GCN).

### 2. Organize Directory

Place the raw data in the data/ folder. The structure should look like this:

```
data/
└── emotion_gait/
    ├── raw_data/            # Put original .npy or .txt files here
    └── ...
```

### 3. Preprocessing

Run the generation script to create bone vectors and motion data (BPM):

```bash
python data/generate_bpm_complete_data.py
```

This will generate the processed .npz files required for training in the data/ directory.

## 🏃 Usage

### Training

To train the CTR-Gait model using Bone modality with ESCL enabled:

```bash
python main.py --config bone_escl
```

`--config` takes the config name without the `.yaml` suffix. Check points will be saved in the work_dir specified in the config file.

### Evaluation

To evaluate a pre-trained model on the test set, download the checkpoint from the [v1.0 release](https://github.com/yeaoshuang/CTR-Gait/releases/download/v1.0/bone_escl_CTRGCN_bone.pth.tar) and place it at `weights/bone_escl_CTRGCN_bone.pth.tar`. The checkpoint is distributed as a release asset and is not included in the auto-generated source archive.

```bash
python main.py --config bone_escl --evaluate --pretrained_path weights
```

`--pretrained_path` takes a directory rather than a file, and the checkpoint name must stay `bone_escl_CTRGCN_bone.pth.tar`.

> Expected Result: Accuracy ~92.66%

### Customization

You can modify parameters in configs/bone_escl.yaml:

- batch_size: Adjust according to your GPU memory.
- step: Learning rate decay steps.
- base_lr: Initial learning rate.

## 📊 Model Zoo

We provide pre-trained weights to reproduce the reported results.

| Model | Modality | Accuracy | GFLOPs | Params | Download |
|-------|----------|----------|--------|--------|----------|
| CTR-Gait | Bone | 92.66% | 0.39G | 3.76M | [bone_escl_CTRGCN_bone.pth.tar](https://github.com/yeaoshuang/CTR-Gait/releases/download/v1.0/bone_escl_CTRGCN_bone.pth.tar) |


## ⚖️ License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📝 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{ye2026ctrgait,
  title   = {CTR-Gait: Semantic-Aware Contrastive Learning for Fine-Grained Gait Emotion Recognition},
  author  = {Ye, Aoshuang and Chen, Chunjie and Wang, Zhuo and Chen, Zhujin and Zhang, Zhi and Zhang, Rong and Cao, Wujing and Wu, Xinyu},
  journal = {IEEE Transactions on Affective Computing},
  year    = {2026},
  pages   = {1--14},
  doi     = {10.1109/TAFFC.2026.3723033},
  note    = {Preprint}
}
```

A. Ye, C. Chen, Z. Wang, Z. Chen, Z. Zhang, R. Zhang, W. Cao, and X. Wu, "CTR-Gait: Semantic-Aware Contrastive Learning for Fine-Grained Gait Emotion Recognition," *IEEE Transactions on Affective Computing*, PrePrints, pp. 1-14, 2026, doi: 10.1109/TAFFC.2026.3723033.

