# Deep learning-based tracking of subduction zones in mantle convection models
A deep learning tool for semantic segmentation of subduction zones from numerical mantle convection model outputs.

## Overview
This repository provides a deep learning framework for detecting subduction zones (SZs) using Fully Convolutional Networks (FCNs) for semantic segmentation of input images. The method performs semantic segmentation on RGB composites to identify SZs at each timestep and then tracks their longevity across time. The example inputs included are based on temperature, vertical velocity, and fineness (inverse grain size) fields. However, the framework is flexible and allows users to substitute or modify the input fields according to their own datasets. The code is written in Python.
> This repo is accompanied by **Choi & Foley (2025)**. See the **Cite this work** section below.


**Why this tool?**
- Automates SZ detection beyond threshold-based heuristics
- Tracks SZs across timesteps for longevity/consistency analyses
- Flexible inputs: swap any scalar fields to build your own RGB composites

**What’s included**
- **Colab notebooks** for dataset prep (SAM-based masks) and FCN training/inference
- Example **RGB images** and a simple **end-to-end run** on sample data
- Outputs saved as `.npz` with SZ indices (and optional continent margins) for downstream analysis


## Detection & tracking: Demos + Scripts

There are two ways to use this project:

### 1) Colab demos (learn & reproduce)

The Colab notebooks walk through the entire workflow on provided example data: how we generated ground-truth masks, trained the network, evaluated different architectures, and ran inference. These notebooks are meant for transparency and reproducibility — you can run them in Google Colab without installing anything locally.

**Notebooks**
- `SAM.ipynb` — Generate ground-truth masks with Segment Anything (SAM) from RGB composites.
- `FCN_SZ_detection3.ipynb` — Data prep, training, validation, inference, and visualization on provided tensors. This notebook also benchmarks multiple model depths — **FCN4, FCN6, FCN8, FCN10**.
- `Alt_FCN_SZ_detection3.ipynb` — Try alternate RGB field combinations (vertical velocity, modified temperature, viscosity) and compare results.

**Training / evaluation data provided in this repo:**
- **RGB inputs used for training / validation / evaluation:**  
  `SZ-detection/rgb_images/*.pt`  
  These `.pt` files are PyTorch tensors containing the 3-channel composite images (R = e.g. vertical velocity, G = fineness, B = modified temperature) used as input to the FCN.

- **Ground-truth masks (from SAM):**  
  `SZ-detection/boolean_images/*.o`

- **Best performing trained FCN model (FCN8):**  
  `SZ-detection/SZ_detection_FCN/trained_model_fcn_best.pt`

**What you can do directly in Colab:**
1. Inspect how the ground-truth masks were created.
2. Re-train the FCN on the provided `rgb_images/*.pt` data.
3. Run inference with the trained model and visualize predicted subduction zones.
4. Export predictions for later tracking/analysis (tracking is handled by the Python scripts in this repo outside of Colab).

These notebooks are primarily for demonstration and reproducibility.  
For batch inference and longevity tracking on your own model outputs, use the standalone Python scripts in this repository.


### 2) Python scripts (run the workflow for real)

The `.py` scripts implement the **production pipeline** used for batch runs and paper figures. Use these when you want repeatable, large-scale experiments beyond the Colab demos.

**Prereqs**
```bash
git clone https://github.com/heec12/SZ-detection.git
cd SZ-detection
pip install numpy matplotlib torch
```

**What’s provided**
- **Raw mantle convection outputs (100 timesteps):**  
  `SZ-detection/example/dam1_diss_1.2_new_icmobR_crust095_vis100/`  
  *(limited to 100 timesteps due to storage constraints — ask below for more)*

- **RGB generator script:**  
  `SZ-detection/rgb_images/generate_rgb_ha_contd.py`  
  Converts raw fields (e.g., vertical velocity, fineness, modified temperature) into RGB tensors used by the FCN.

- **Main pipeline scripts:**  
  - **Entry point:** `SZ-detection/SZ_detection_FCN/conv_tracker_fcn.py`  
  - **Helpers:** `SZ-detection/SZ_detection_FCN/conv_tracker_fcn_func.py`,  
    `SZ-detection/SZ_detection_FCN/conv_tracker_trck_func.py`

**A) Generate RGB inputs from the raw outputs**
```bash
# From the repo root
python rgb_images/generate_rgb_ha_contd.py
```

**B) Run FCN detection on the RGB tensors and track SZs through time**
```bash
python SZ_detection_FCN/conv_tracker_fcn.py
```

**Need more than 100 timesteps?**
Open an issue or contact the maintainer to request additional datasets. Due to the storage limit, the example case is provided only up to 100 timestep.

**Output**
- The main output is saved as a `.npz` file (e.g., `sz_tracker_output_<modelname>.npz`).
- This file contains arrays with subduction zone tracking indices and, optionally, continent margin locations.
- You can load and analyze the `.npz` results in Python using `numpy.load`.


## Post-processing & analysis

Two lightweight scripts read the tracked results (`.npz`) and generate summary figures:

- **Histograms** of subduction-zone longevity
- **Box-and-whisker plots** comparing groups/runs

### Usage

**A) Longevity histograms**
```bash
python post_plot/post_plot_histogram.py 
````

**B) Box-and-whisker plots**
```bash
python post_plot/post_plot_box_whisker.py 
````

**Inputs**
- **Required:** `.npz` file produced by the tracking step, containing:
  - `conv_tracker_index` — SZ indices linked across timesteps
  - `cont_tracker_index` — continent boundary indices per timestep

**Outputs**
- Figure files (PNG/PDF/SVG) written to your chosen path.

---
*See Choi and Foley (2025) for details on output interpretation and further analysis.*


## Cite this work

If you use this code or approach in your research, please cite:

> Choi, H., & Foley, B. (2025).  
> Deep learning-based tracking of subduction zones in mantle convection models.  
> *Submitted to Journal of Geophysical Research: Solid Earth.*


## Questions or issues?

Open an issue on GitHub or contact hchoi342@gatech.edu
