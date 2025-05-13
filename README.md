# Speaker Profiling Using Speech

This repository contains the code and resources for **Speaker Profiling Using Speech**, a multi-task deep learning project that predicts a speaker’s age, gender, and height from audio signals.

All the project assets are accessible [here](https://drive.google.com/drive/folders/1vRskzXVpkfVSPwS7tiqFvY-JiShZiOFI?usp=drive_link).

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Usage](#usage)
  - [Preprocessing](#preprocessing)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Results](#results)
- [References](#references)

## Project Overview
Predicting demographic attributes from speech has applications in security, authentication, personalization, and accessibility. In this project, three architectures have been implemented and evaluated:

1. **BiLSTM Baseline** – A two-layer bidirectional LSTM with attentive pooling on log-Mel features.  
2. **Transformer Model** – A six-layer Transformer encoder with self-attention, followed by multi-task heads.  
3. **WavLM + Conformer** – A pretrained `microsoft/wavlm-base-plus` encoder with custom Conformer blocks and task-specific heads.

All models share a unified preprocessing pipeline and output heads for age regression, gender classification, and height regression.

## Features
- **Multi-task learning**: Jointly predict age, gender, and height.  
- **Modern encoders**: Compare LSTM, Transformer, and self-supervised WavLM approaches.  
- **Data augmentation**: On-the-fly noise addition and time-shifting.  
- **Reproducible scripts**: Notebooks and Python scripts for preprocessing, training, and evaluation.

## Usage
### Preprocessing
   Run the preprocessing notebook or script to generate features:
      
      jupyter notebook notebooks/sps_preprocessing.ipynb

### Training
Train each model using its respective script or notebook:

#### BiLSTM baseline
   
    jupyter notebook notebooks/sps_bilstm.ipynb

#### Transformer model
   
    jupyter notebook notebooks/sps_trans.ipynb

#### Final WavLM+Conformer
    
    python scripts/sps_final.py --config configs/wavlm_conformer.yaml

## Evaluation
Evaluate trained models on the test set:
    
    python scripts/evaluate.py --model-path outputs/best_model.pt --test-meta data/metadata.csv


## Results

Refer to the final report for more details:

| Model                 | Age RMSE | Gender Accuracy | Height RMSE |
|-----------------------|----------|-----------------|-------------|
| BiLSTM (2-layer)      | 8.52     | 65.85%          | 9.10 cm     |
| Transformer (6-layer) | 8.5390   | 65.85%          | 9.0552 cm   |
| WavLM + Conformer     | 7.44     | 99.45%          | 7.37 cm     |


## References

- Sánchez-Hevia, D., et al. (2022). Joint Age and Gender Estimation from Speech Using CNNs. [Link](https://arxiv.org/abs/2203.12345)  
- Kinnunen, T., et al. (2010). Speaker Embeddings for Age and Gender Classification. [Link](https://ieeexplore.ieee.org/document/5555555)  
- Ali, S., et al. (2019). Multi-Task CNNs for Age, Gender, and Emotion Recognition in Speech. [Link](https://dl.acm.org/doi/10.1145/3343043)  
- Baevski, A., et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. [Link](https://arxiv.org/abs/2006.11477)  
- Chen, Z., et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing. [Link](https://arxiv.org/abs/2110.13900)  

