# Clinically-Aware and Fair GAN (CAF-GAN) for Medical Image Generation

This repository contains the official PyTorch implementation for the paper "CAF-GAN: A Clinically-Aware and Fair Generative Adversarial Network." The project focuses on generating realistic, high-fidelity medical images (specifically chest X-rays) that are also fair across different demographic groups and clinically plausible.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.10-red.svg)](https://pytorch.org/)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📜 Table of Contents
- [Introduction](#-introduction)
- [Project Objectives](#-project-objectives)
- [Architecture Overview](#-architecture-overview)
- [Project Structure](#-project-structure)  
- [Quick Start](#-quick-start)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
  - [Dataset Preparation](#dataset-preparation)  
  - [Training](#training)  
  - [Evaluation](#evaluation)  
- [Results (Example)](#results-example)  
- [Citation](#citation)  
- [License](#license)  
- [Acknowledgements & Contact](#acknowledgements--contact)

---

## 📖 Introduction

Generative Adversarial Networks (GANs) have shown great promise in synthesizing realistic medical images, which can be used for data augmentation, anonymization, and education. However, standard GANs can inherit and even amplify biases present in the training data, leading to models that perform inequitably across different patient populations (e.g., race, sex).

This project implements **CAF-GAN**, a novel framework that introduces two specialized "critic" models to guide the generator. These critics ensure that the generated images are not only realistic but also fair from a diagnostic standpoint and anatomically correct.

---

## 🎯 Project Objectives

1.  **High Fidelity:** Generate realistic chest X-ray images that are indistinguishable from real ones.
2.  **Fairness:** Mitigate algorithmic bias by ensuring a downstream diagnostic model performs equitably across different protected demographic groups.
3.  **Clinical Plausibility:** Enforce anatomical correctness in the generated images, such as realistic lung shapes and sizes.
4.  **Downstream Utility:** Produce synthetic data that can be used to train high-performing, fair, and robust downstream classifiers.

---

## 🏗️ Architecture Overview

The CAF-GAN builds upon a standard WGAN-GP framework and introduces two key components that add specialized loss signals to the generator's training objective:

1.  **Fairness Critic (`Cdiag`):** A pre-trained, frozen diagnostic classifier (e.g., ResNet-50) that predicts a specific pathology (e.g., Pneumonia). The generator is penalized if the `Cdiag` model's performance (e.g., True Positive Rate) differs significantly across demographic groups for the generated images.
2.  **Clinical Critic (`Cseg`):** A pre-trained, frozen segmentation model (e.g., U-Net) that identifies key anatomical structures (e.g., lungs). The generator is penalized if the segmented anatomy from its generated images is implausible (e.g., incorrect size, shape, or contiguity).

The total generator loss is a weighted sum of the adversarial loss, the fairness loss, and the clinical plausibility loss.

$$ L_G^{\text{total}} = L_{\text{WGAN}} + \lambda_{\text{fair}} L_{\text{fairness}} + \lambda_{\text{clinic}} L_{\text{clinical}} $$

---


## 📂 Project Structure
```

caf-gan-mimic-cxr/
│
├── data/ # Raw and processed data (ignored by git)
├── configs/ # YAML config files for training and experiments
├── notebooks/ # Notebooks for exploration & tests
├── outputs/ # Saved checkpoints & generated images (ignored by git)
├── scripts/ # Helper scripts (download / utils)
│
├── src/
│ ├── data/ # Dataset classes & preprocessing
│ ├── models/ # Generator, Discriminator, Critics
│ ├── training/ # train_gan.py, train_cdiag.py, train_cseg.py
│ └── evaluation/ # evaluation scripts & metrics
│
├── requirements.txt
├── README.md
└── .gitignore

````

---

## ⚙️ Quick Start

### Prerequisites

- Linux / macOS (recommended). Works on Windows with WSL.
- Python 3.8+  
- CUDA-enabled GPU (recommended) and compatible PyTorch build.

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/saibharath954/Clinically-Aware-Fair-GAN.git
cd Clinically-Aware-Fair-GAN

# 2. Create virtual environment (venv example)
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
````

> If you prefer `conda`:
>
> ```bash
> conda create -n cafgan python=3.9 -y
> conda activate cafgan
> pip install -r requirements.txt
> ```

---

## 💾 Dataset Preparation

This project uses the **MIMIC-CXR** dataset (or MIMIC-CXR-JPG). MIMIC-CXR is protected and requires credentialed access via PhysioNet.

1. **Get access** — create an account on [PhysioNet](https://physionet.org/) and request access to MIMIC-CXR.

2. **Download required files** (DICOM/JPG + metadata + precomputed CheXpert labels). Example files you might need:

   * `mimic-cxr-2.0.0-chexpert.csv`
   * `cxr-record-list.csv`
   * patient/demographic csvs as required

3. **Organize & preprocess**

Place raw files in `data/` then run preprocessing scripts to create a consolidated metadata CSV and prepare image lists.

```bash
# Example pipeline (adapt to your scripts/configs)
python src/data/preprocessing.py        # produces master CSV for your subset
python src/data/generate_download_list.py
bash scripts/download_images.sh
python src/data/generate_masks.py       # generate segmentation masks (for Cseg)
```

> **Note:** Update paths and config values in `configs/*.yaml` before running scripts.

---

## 🚀 Training

All training scripts accept a `--config` YAML path (see `configs/`).

### 1) Train Fairness Critic (Cdiag)

```bash
python src/training/train_cdiag.py --config configs/train_cdiag.yaml
# outputs saved to outputs/cdiag/
```

### 2) Train Clinical Critic (Cseg)

```bash
python src/training/train_cseg.py --config configs/train_cseg.yaml
# outputs saved to outputs/cseg/
```

### 3) Train CAF-GAN

```bash
python src/training/train_gan.py --config configs/train_gan.yaml
# checkpoints & generated samples -> outputs/gan/
```

Common config options you may tune:

* learning rates, batch size, image resolution
* adversarial loss weights, lambda\_fair, lambda\_clinic
* critic pre-trained checkpoint paths

---

## 📊 Evaluation

After training a GAN checkpoint, evaluate synthetic data utility & fairness:

```bash
python src/evaluation/evaluate.py --gan_checkpoint outputs/gan/best_generator.pth
```

Evaluation pipeline (high-level):

1. Generate synthetic dataset with trained generator.
2. Train a downstream classifier on synthetic data (from scratch).
3. Evaluate on held-out real test set for utility (AUC, F1) and fairness (e.g., Equal Opportunity Difference).

---

## 📈 Results (Example)

| Model                | Test AUC | Equal Opportunity Difference |
| -------------------- | -------: | ---------------------------: |
| Baseline (Real Data) |     0.85 |                         0.15 |
| Standard WGAN-GP     |     0.82 |                         0.12 |
| **CAF-GAN (Ours)**   | **0.84** |                     **0.04** |

> These are illustrative numbers — run experiments on your dataset/config to reproduce results.

---

## 📜 Citation

If you use this code or ideas from this project, please cite:

<!-- ```bibtex
@article{your_paper_citation,
  title={CAF-GAN: A Clinically-Aware and Fair Generative Adversarial Network},
  author={Sai Bharath Pediredla},
  journal={Conference or Journal Name},
  year={2025},
  url={https://arxiv.org/abs/xxxx.xxxxx}  # replace with actual link if available
}
``` -->

---

## 🧾 License

This project is released under the **MIT License**. See [LICENSE](./LICENSE) for details.

---

## 🙏 Acknowledgements & Contact

* This work uses the **MIMIC-CXR** dataset (PhysioNet).
* Thanks to the authors of **CheXpert** labeler and available lung segmentation tools that the project builds upon.
* If you have questions or want to collaborate, contact: **saibharath1675@gmail.com** 

---

Thank you for checking out **CAF-GAN** — feel free to open issues or PRs if you find bugs, want features, or want to reproduce/extend experiments. 🚀

