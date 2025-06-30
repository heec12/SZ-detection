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


## Installation

No package installation is required beyond standard Python libraries.

To get started with example runs:

1. **Clone the repository**:
```bash
git clone https://github.com/heec12/SZ-detection.git
cd 
