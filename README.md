# Clinically-Aware and Fair GAN (CAF-GAN) for Medical Image Generation

This repository contains the official PyTorch implementation for the paper "CAF-GAN: A Clinically-Aware and Fair Generative Adversarial Network." The project focuses on generating realistic, high-fidelity medical images (specifically chest X-rays) that are also fair across different demographic groups and clinically plausible.

![GitHub Banner](https://user-images.githubusercontent.com/your_github_id/your_image_id.png)  ## 📜 Table of Contents
- [Introduction](#-introduction)
- [Project Objectives](#-project-objectives)
- [Architecture Overview](#-architecture-overview)
- [Repository Structure](#-repository-structure)
- [Setup and Installation](#-setup-and-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Results](#-results)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

## 📖 Introduction

Generative Adversarial Networks (GANs) have shown great promise in synthesizing realistic medical images, which can be used for data augmentation, anonymization, and education. However, standard GANs can inherit and even amplify biases present in the training data, leading to models that perform inequitably across different patient populations (e.g., race, sex).

This project implements **CAF-GAN**, a novel framework that introduces two specialized "critic" models to guide the generator. These critics ensure that the generated images are not only realistic but also fair from a diagnostic standpoint and anatomically correct.

## 🎯 Project Objectives

1.  **High Fidelity:** Generate realistic chest X-ray images that are indistinguishable from real ones.
2.  **Fairness:** Mitigate algorithmic bias by ensuring a downstream diagnostic model performs equitably across different protected demographic groups.
3.  **Clinical Plausibility:** Enforce anatomical correctness in the generated images, such as realistic lung shapes and sizes.
4.  **Downstream Utility:** Produce synthetic data that can be used to train high-performing, fair, and robust downstream classifiers.

## 🏗️ Architecture Overview

The CAF-GAN builds upon a standard WGAN-GP framework and introduces two key components that add specialized loss signals to the generator's training objective:

1.  **Fairness Critic (`Cdiag`):** A pre-trained, frozen diagnostic classifier (e.g., ResNet-50) that predicts a specific pathology (e.g., Pneumonia). The generator is penalized if the `Cdiag` model's performance (e.g., True Positive Rate) differs significantly across demographic groups for the generated images.
2.  **Clinical Critic (`Cseg`):** A pre-trained, frozen segmentation model (e.g., U-Net) that identifies key anatomical structures (e.g., lungs). The generator is penalized if the segmented anatomy from its generated images is implausible (e.g., incorrect size, shape, or contiguity).

The total generator loss is a weighted sum of the adversarial loss, the fairness loss, and the clinical plausibility loss.

$$ L_G^{\text{total}} = L_{\text{WGAN}} + \lambda_{\text{fair}} L_{\text{fairness}} + \lambda_{\text{clinic}} L_{\text{clinical}} $$

## 📂 Repository Structure

caf-gan-mimic-cxr/
│
├── data/              # Raw and processed data (ignored by git)
├── configs/           # Configuration files for training and data
├── notebooks/         # Jupyter notebooks for exploration and testing
├── outputs/           # Saved model checkpoints and generated images (ignored by git)
├── scripts/           # Helper scripts (e.g., environment setup)
│
├── src/
│   ├── data/          # Data loading (Dataset) and preprocessing scripts
│   ├── models/        # Model definitions (Generator, Discriminator, Critics)
│   ├── training/      # Main training scripts for critics and the GAN
│   └── evaluation/    # Scripts for evaluating fairness and utility
│
├── requirements.txt   # Python dependencies
├── README.md          # This file
└── .gitignore


## ⚙️ Setup and Installation

Follow these steps to set up the environment and install the necessary dependencies.

1. Clone the repository:
git clone [https://github.com/saibharath954/Clinically-Aware-Fair-GAN.git](https://github.com/saibharath954/Clinically-Aware-Fair-GAN.git)
cd Clinically-Aware-Fair-GAN

2. Create a Python Virtual Environment:
It's highly recommended to use a virtual environment (e.g., venv or conda).

python3 -m venv venv
source venv/bin/activate

3. Install Dependencies:
Install all required packages using the requirements.txt file.

pip install -r requirements.txt

💾 Dataset Preparation
This project uses the MIMIC-CXR dataset. Due to its protected nature, you must first gain credentialed access on PhysioNet.

1. Download Data:
Follow the instructions on PhysioNet to download the DICOM images and the relevant metadata files (cxr-record-list.csv, patients.csv, admissions.csv). You will also need the pre-computed CheXpert labels from the MIMIC-CXR-JPG dataset (mimic-cxr-2.0.0-chexpert.csv).

2. Organize Data:
Place the downloaded files into the data/ directory according to the structure expected by the preprocessing script.

3. Run Preprocessing:
First, create a final metadata file that contains all the necessary information (image paths, labels, and demographic data). Then, download the required subset of images.

# Step 1: Create the master CSV file for a 2,000 image subset
python src/data/preprocessing.py

# Step 2: Generate the list of image URLs to download
python src/data/generate_download_list.py # Assuming you create this script

# Step 3: Run the download script
bash scripts/download_images.sh

4. Generate Segmentation Masks:
After downloading the images, generate the pseudo-ground-truth masks for the clinical critic.

python src/data/generate_masks.py

🚀 Training
The training process is divided into three main stages. All training scripts can be configured using the YAML files in the configs/ directory.

1. Pre-train the Fairness Critic (Cdiag):
This trains the diagnostic classifier on the real data.

python src/training/train_cdiag.py --config configs/train_cdiag.yaml
The best model will be saved in the outputs/cdiag/ directory.

2. Pre-train the Clinical Critic (Cseg):
This trains the segmentation model on the real data and the generated masks.

python src/training/train_cseg.py --config configs/train_cseg.yaml
The best model will be saved in the outputs/cseg/ directory.

3. Train the CAF-GAN:
This is the main training script, which loads the frozen critics and trains the generator and discriminator.

python src/training/train_gan.py --config configs/train_gan.yaml
Checkpoints and sample generated images will be saved periodically to outputs/gan/.

📊 Evaluation
After training the CAF-GAN, run the evaluation script to assess its performance. The script will:

Generate a synthetic dataset using the trained generator.

Train a new downstream classifier from scratch on this synthetic data.

Evaluate the synthetically-trained classifier on the real test set for both utility (AUC, F1-Score) and fairness (Equal Opportunity Difference).

python src/evaluation/evaluate.py --gan_checkpoint outputs/gan/best_generator.pth

📈 Results

Example Table: Fairness and Utility Comparison

Model	Test AUC	Equal Opportunity Difference
Baseline (Real Data)	0.85	0.15
Standard WGAN-GP	0.82	0.12
CAF-GAN (Ours)	0.84	0.04

©️ Citation
If you use this code or the ideas from this project in your research, please cite the original paper:
<!-- Code snippet

@article{your_paper_citation,
  title={CAF-GAN: A Clinically-Aware and Fair Generative Adversarial Network},
  author={Your Name, et al.},
  journal={Journal or Conference Name},
  year={2025}
} -->

📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.

🙏 Acknowledgements

This work relies on the publicly available MIMIC-CXR Dataset.
I thank the authors of the CheXpert labeler and Chest X-ray (LungSegmentation) for their valuable tools.

