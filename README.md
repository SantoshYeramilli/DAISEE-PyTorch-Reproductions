# DAISEE-PyTorch-Reproductions
This repository contains a PyTorch implementation to reproduce the baseline results presented in the paper "DAISEE: Towards User Engagement Recognition in the Wild". The goal is to provide a clear and verifiable reproduction of the reported benchmarks for the DAISEE dataset.

## About DAISEE

[DAISEE (Dataset for Affective States in E-Environments)] (https://arxiv.org/abs/1609.01885v7) is a multi-label video classification dataset designed for recognizing user affective states such as boredom, confusion, engagement, and frustration "in the wild".It comprises 9068 video snippets from 112 users, with four levels of labels (very low, low, high, very high) for each affective state. The dataset addresses the critical need for publicly available data in user engagement recognition, which is relevant to various contemporary vision applications including e-learning, advertising, healthcare, and autonomous vehicles.

The paper highlights the limitations of existing datasets in capturing the subtleties of these affective states in real-world, unconstrained settings.DAISEE aims to bridge this gap by providing a diverse dataset that includes variations in user poses, positions, and background noises, typical of natural e-learning environments.

## Reproduction Details

This project specifically focuses on reproducing the baseline results described in Section 4 of the DAISEE paper, using PyTorch. The models evaluated in the original paper include:

* **InceptionNet Frame Level:** Single frame classification using InceptionNet V3 pre-trained on ImageNet.
* **InceptionNet Video Level:** Frame-level predictions aggregated for video-level accuracy using the same InceptionNet V3 model.
* **C3D Training:** Full training of a 3D CNN (C3D) using the DAISEE data splits.
* **LRCN (Long-Term Recurrent Convolutional Network):** Video classification using a unified visual and sequence learning approach.

This repository aims to achieve comparable performance using PyTorch implementations of these models.

## Getting Started
### Prerequisites

* Python 3.x
* PyTorch
* Other common deep learning libraries(as listed in `requirements.txt`)
* `c3d-pytorch` This project utilizes the C3D model implementation from [DavideA/c3d-pytorch](https://github.com/DavideA/c3d-pytorch.git).

### Dataset Download

The DAISEE dataset and the associated paper are publicly available. You can find the paper [here](https://arxiv.org/abs/1609.01885v7). Please follow the instructions on their official website at https://people.iith.ac.in/vineethnb/resources/daisee/index.html to download and prepare the dataset. It is recommended to place the downloaded and processed dataset at a path like `DAiSEE/` or similar, which you will then specify as your `data_root`.

### Installation

1.  Clone the repository and its submodules (if using submodules):
    ```bash
    git clone --recurse-submodules https://github.com/SantoshYeramilli/DAISEE-PyTorch-Reproductions.git
    cd DAISEE-PyTorch-Reproductions
    ```
2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Experiments

1. Open `main.py` in a text editor.

2. Uncomment the imports for the specific dataloader, model, train, and test functions corresponding to the model you wish to run (e.g., for C3D, uncomment the lines under #C3D import and comment out others).

3. Uncomment and select the model instance (e.g., model = C3D(4)).

4. Manually update the path variables to point to the absolute paths of your DAISEE dataset on your local machine.

5. Save the changes and run the script:
```bash
python main.py
```
### Repository Structure
```bash
.
├── C3D/
│   ├── dataloader.py
│   ├── test.py
│   └── train.py
├── Frame_Level/
│   ├── dataloader.py
│   ├── inceptionnet_v3_model.py
│   ├── test.py
│   └── train.py
├── LRCN/
│   ├── dataloader.py
│   ├── LRCN_model.py
│   ├── test.py
│   └── train.py
├── Video level/
│   ├── dataloader.py
│   ├── inceptionet_v3_model.py
│   ├── test.py
│   └── train.py
├── .gitignore
├── main.py
├── README.md
├── requirements.txt
└── models/
    └── c3d-pytorch/         # Git submodule for C3D model implementation [https://github.com/DavideA/c3d-pytorch.git]
```
