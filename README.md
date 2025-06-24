# End-to-End Target Detection in Range-Doppler Maps with Temporal U-Nets

This repository contains the code developed as part of my BSc thesis at **Leiden University** and internship at **TNO**, titled:


**_End-to-End Target Detection in Range-Doppler Maps with Temporal U-Nets: Deep Learning Approaches for Maritime Radar_**

## Overview

The aim of this project is to evaluate and compare the performance of deep learning models in maritime radar target detection, specifically focusing on sea clutter suppression. The pipeline includes:

- Image segmentation via a **U-Net-based** architecture
- Synthetic radar and sea clutter simulation

This work demonstrates how AI, particularly deep learning, can be applied to real-world signal processing problems in radar technology.

## Getting Started

### With pip install 

- Create a virtual environment and install the required packages:
- python -m venv venv
- source venv/bin/activate # or venv\Scripts\activate on Windows
- pip install -r requirements.txt

### With uv sync
- Make sure to have poetry and uv installed on the device
- Do uv sync

##  Usage

### Marimo Notebook usage
This repository includes a Marimo notebook which allows its users to design their own sea cluttered Range Doppler maps, generate a dataset and finally train a Unet on it. Use the following command to run: 
```bash
marimo run app.py
```

## Citation

If you use this work in your research, please cite this repository as described below or see the CITATION.cff file for citation formats:

```bibtex
@software{Lens_SeaClutterSuppression_2025,
  author = {Pepijn Lens},
  title = {End-to-End Target Detection in Range-Doppler Maps with Temporal U-Nets: Deep Learning Approaches for Maritime Radar},
  url = {https://github.com/pepijn-lens/SeaClutterSuppression},
  year = {2025},
  note = {BSc thesis at Leiden University and internship at TNO}
}
```

## Acknowledgments

I would like to thank my supervisors:

- Bas Jacobs
- Giuseppe Papari
- Peter van der Putten
- Daan Pelt

## Future Work
- Training on real-world radar datasets

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
