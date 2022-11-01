# Disclaimer
This training and inference pipeline was developed by NVIDIA. It is based on a segmentation model developed by NVIDIA researchers in conjunction with the NIH. 

# Model Overview
A pre-trained model for volumetric (3D) segmentation of bone lesions from CT images.

## Workflow
The model is trained using a 3D SegResNet network [1]. 

## Data
CT volumes of 297 subjects

- Target: Lung
- Task: Segmentation
- Modality: CT  
- Size: 297 3D volumes (70% Training, 15% Validation, 15% Testing)
- Challenge: Large ranging foreground size

# Training configuration
The training was performed with the following:

- Script: train.sh
- GPU: at least 16GB of GPU memory. 
- AMP: True
- Optimizer: Adam
- Learning Rate: 1e-4
- Loss: DiceLoss

## Input
Input: 1 channel CT image with intensity in HU and arbitary spacing

1. Resampling spacing to (1, 1, 1) mm
2. Clipping intensity to [-500, 1300] HU
3. Converting to channel first
4. Randomly cropping the volume to a fixed size (128,128,128)
5. Randomly applying spatial flipping
6. Randomly applying spatial rotation
6. Randomly shifting intensity of the volume

## Output
Output: 2 channels
- Label 0: everything else
- Label 1: bone lesion

# Intended Use
The model needs to be used with NVIDIA hardware and software. For hardware, the model can run on any NVIDIA GPU with memory greater than 16 GB. For software, this model is usable only as part of Transfer Learning & Annotation Tools in Clara Train SDK container.  Find out more about Clara Train at the [Clara Train Collections on NGC](https://ngc.nvidia.com/catalog/collections/nvidia:claratrainframework).

**The Clara pre-trained models are for developmental purposes only and cannot be used directly for clinical procedures.**

# License
[End User License Agreement](https://developer.nvidia.com/clara-train-eula) is included with the product. Licenses are also available along with the model application zip file. By pulling and using the Clara Train SDK container and downloading models, you accept the terms and conditions of these licenses.

# References
[1] Milletari F, Navab N, Ahmadi S-A. V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. 2016 Fourth International Conference on 3D Vision (3DV). 2016. p. 565–571. doi: 10.1109/3DV.2016.79
