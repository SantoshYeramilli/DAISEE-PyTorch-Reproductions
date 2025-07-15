# DAISEE-PyTorch-Reproductions
This repository contains a PyTorch implementation to reproduce the baseline results presented in the paper "DAISEE: Towards User Engagement Recognition in the Wild". The goal is to provide a clear and verifiable reproduction of the reported benchmarks for the DAISEE dataset.

## About DAISEE

DAISEE (Dataset for Affective States in E-Environments) (https://arxiv.org/abs/1609.01885v7) is a multi-label video classification dataset designed for recognizing user affective states such as boredom, confusion, engagement, and frustration "in the wild".It comprises 9068 video snippets from 112 users, with four levels of labels (very low, low, high, very high) for each affective state. The dataset addresses the critical need for publicly available data in user engagement recognition, which is relevant to various contemporary vision applications including e-learning, advertising, healthcare, and autonomous vehicles.

The paper highlights the limitations of existing datasets in capturing the subtleties of these affective states in real-world, unconstrained settings.DAISEE aims to bridge this gap by providing a diverse dataset that includes variations in user poses, positions, and background noises, typical of natural e-learning environments.

## Reproduction Details

This project specifically focuses on reproducing the baseline results described in Section 4 of the DAISEE paper, using PyTorch. The models evaluated in the original paper include:

* **InceptionNet Frame Level:** Single frame classification using InceptionNet V3 pre-trained on ImageNet.
* **InceptionNet Video Level:** Frame-level predictions aggregated for video-level accuracy using the same InceptionNet V3 model.
* **C3D Training:** Full training of a 3D CNN (C3D) using the DAISEE data splits.
* **LRCN (Long-Term Recurrent Convolutional Network):** Video classification using a unified visual and sequence learning approach.

This repository aims to achieve comparable performance using PyTorch implementations of these models.


**Note : `c3d-pytorch`**- This project utilizes the C3D model implementation from [DavideA/c3d-pytorch](https://github.com/DavideA/c3d-pytorch.git).
