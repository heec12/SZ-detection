# Deep learning-based tracking of subduction zones in mantle convection models
A deep learning tool for semantic segmentation of subduction zones from numerical mantle convection model outputs.

## Overview
This repository provides a deep learning framework for detecting subduction zones (SZs) using Fully Convolutional Networks (FCNs) for semantic segmentation of input images. The example inputs included are based on temperature, vertical velocity, and fineness (inverse grain size) fields. However, the framework is flexible and allows users to substitute or modify the input fields according to their own datasets. The code is written in Python.

## Getting started: Try it instantly in Google Colab
If you’d like to explore the workflow and run examples without installing anything locally, check out the Google Colab notebook. This notebook walks you through the complete SZ detection workflow.

We provide two Colab notebooks:

- **SAM.ipynb:**  
  Learn how to generate the training dataset using the Segment Anything Model (SAM) from Meta. This notebook guides you through preparing annotated training images for subduction zone detection.

- **FCN_SZ_detection2.ipynb:**  
  This notebook covers the deep learning workflow. It includes data loading, model training, inference, and visualization using the FCN-based subduction zone detection framework.

Necessary sample RGB images are under `rgb_images` folder.


## Installation

No package installation is required beyond standard Python libraries.

To get started with example runs:

1. **Clone the repository**:
```bash
git clone https://github.com/heec12/SZ-detection.git
cd SZ_detection_FCN
```

2. **Install dependencies**:
   
This project requires Python 3.7+ and the following Python libraries:
- numpy
- matplotlib
- torch (PyTorch)

You can install the dependencies using pip:

```bash
pip install numpy matplotlib torch
```

## Usage

### Run subduction zone detection on example data

After installation, you can run the detection program using the provided example files:

```bash
python conv_tracker_fcn.py
```
This will load the pre-trained model and example data, and output the predicted subduction zones and tracking results. Ensure the file paths for input RGB images and output files are specified correctly before running. 


### Output

- The main output is saved as a `.npz` file (e.g., `sz_tracker_output_<modelname>.npz`).
- This file contains arrays with subduction zone tracking indices and, optionally, continent margin locations.
- You can load and analyze the `.npz` results in Python using `numpy.load`.

**Contents:**
- `conv_tracker_index`: Subduction zone tracking indices across all time steps
- `cont_tracker_index`: (Optional) Continent margin locations, if calculated

---
*See Choi and Foley (2025) for details on output interpretation and further analysis.*


## Citation

If you use this code or approach in your research, please cite:

> Choi, H., & Foley, B. (2025).  
> Deep learning-based tracking of subduction zones in mantle convection models.  
> *Submitted to Journal of Geophysical Research: Solid Earth.*


## Questions or issues?

Open an issue on GitHub or contact hchoi342@gatech.edu
