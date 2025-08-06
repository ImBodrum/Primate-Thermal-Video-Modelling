# Primate-Thermal-Video-Modelling
In this project, a new data processing pipeline is implemented that integrates multiple inpainting techniques and preprocessing strategies to optimize feature extraction and obstruction mitigation.

# Primate Thermal Video Modelling: Fence Removal and Feature Tracking

This repository contains the code and resources for a Computer Science final year dissertation focused on removing occlusions (e.g., fences) and tracking facial features (specifically the nose) of primates in thermal video footage. The system utilizes a combination of classical and deep learning-based image inpainting techniques, along with multiple tracking algorithms.

##  Project Structure

- `preprocessing.py`: Handles preprocessing of thermal frames (CLAHE, thresholding, mask generation).
- `inpainting_and_tracking.py`: Contains inpainting pipelines (EdgeConnect, SIREN, etc.) and nose tracking algorithms (Lucas-Kanade variants, CSRT).
- `utils.py`: Utility functions for file I/O, visualization, and logging.

##  Methods

- **Inpainting:** EdgeConnect, SIREN, Telea, Navier-Stokes
- **Tracking:** Lucas–Kanade (base), weighted median, limited shift, morphological local maximum, CSRT
- **Evaluation:** PSNR, SSIM, CSNR, Temperature Extraction

## 📁 Folder Structure


## 📝 Citation

If you use this codebase in your research, please cite the original report and relevant papers.

## 🧠 Author

- Maksut O'Connor | University of Sussex | 2025


