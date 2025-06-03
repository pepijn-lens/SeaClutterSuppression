# End-to-End Target Detection in Range-Doppler Maps with Temporal U-Nets

This repository contains the code developed as part of my BSc thesis at **Leiden University** and internship at **TNO**, titled:

**_End-to-End Target Detection in Range-Doppler Maps with Temporal U-Nets: Deep Learning Approaches for Maritime Radar_**

## 📘 Overview

The aim of this project is to evaluate and compare the performance of deep learning models in maritime radar target detection, specifically focusing on sea clutter suppression. The pipeline includes:

- Image classification using a **Swin Transformer** and a **CNN**
- Transition to image segmentation via a **U-Net-based** architecture
- Synthetic radar and sea clutter simulation

This work demonstrates how AI, particularly deep learning, can be applied to real-world signal processing problems in radar technology.

## 🚀 Getting Started

### Installation

- Create a virtual environment and install the required packages:
- python -m venv venv
- source venv/bin/activate # or venv\Scripts\activate on Windows
- pip install -r requirements.txt


### Dependencies

- Python
- PyTorch
- NumPy
- SciPy
- StoneSoup

## ⚙️ Usage

### Synthetic Data Generation

- **Without clutter**: Run `radar.py`
- **With sea clutter**: Use the scripts in the `sea_clutter/` directory

### Model Training

Refer to the `training/` directory for training pipelines using CNNs, Swin Transformers, and U-Nets.

## 📂 Project Structure
```plaintext
.
├── data/             # Synthetic and processed radar data  
├── evaluation/       # Results and metrics of model performance  
├── models/           # Saved model weights and architectures  
├── optuna/           # Hyperparameter tuning logs  
├── sea_clutter/      # Sea clutter simulation scripts  
├── training/         # Training pipelines for different models  
├── helper.py         # Utility functions  
├── radar.py          # Synthetic radar data generation (no clutter)  
└── requirements.txt  # Project dependencies  
```

## 📄 Documentation

Refer to my thesis for detailed background, methodology, and experimental results:

📄 **[Thesis Link](#)** 

## 🧠 Built With

- Python
- PyTorch
- NumPy
- SciPy
- StoneSoup

## 📜 License

This repository is under a **proprietary license**. Please contact the author for usage inquiries.

## 🙌 Acknowledgments

I would like to thank my supervisors:

- Bas Jacobs
- Giuseppe Papari
- Peter van der Putten
- Daan Pelt

## 🔭 Future Work

- Integration of hybrid Swin Transformer U-Net architecture
- Training on real-world radar datasets
- Data augmentation with GANs
