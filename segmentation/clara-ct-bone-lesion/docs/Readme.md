# Disclaimer
This training and inference pipeline was developed by NVIDIA. It is based on a segmentation and classification model developed by NVIDIA researchers in conjunction with the NIH. 

# Model Overview
A pre-trained model for volumetric (3D) segmentation of the lung from CT images.

## Workflow
The model is trained using a 3D anisotropic hybrid network [1]. 

![Diagram showing the flow from model input, through the model architecture, and to model output](http://developer.download.nvidia.com/assets/Clara/Images/clara_pt_covid19_ct_lung_segmentation_workflow.png)

## Data
This model was trained on a global dataset with a large experimental cohort collected from across the globe. The CT volumes of 120 independent subjects are provided by NIH with experts’ lung region annotations.

- Target: Lung
- Task: Segmentation
- Modality: CT  
- Size: 120 3D volumes (90 Training, 10 Validation, 20 Testing)
- Challenge: Large ranging foreground size

# Training configuration
The training was performed with the following:

- Script: train.sh
- GPU: at least 16GB of GPU memory. 
- Actual Model Input: 224 x 224 x 32
- AMP: True
- Optimizer: Adam
- Learning Rate: 1e-4
- Loss: DiceLoss

## Input
Input: 1 channel CT image with intensity in HU and arbitary spacing

1. Resampling spacing to (0.8, 0.8, 5) mm
2. Clipping intensity to [-1500, 500] HU
3. Converting to channel first
4. Randomly cropping the volume to a fixed size (224, 224, 32)
5. Randomly applying spatial flipping
6. Randomly applying spatial rotation
6. Randomly shifting intensity of the volume

## Output
Output: 2 channels
- Label 0: everything else
- Label 1: lung

# Model Performance
Dice score is used for evaluating the performance of the model. On the test set, the trained model achieved score of ~0.96 for lung.

## Training Performance
Training acc over 1250 epochs. 

![Graph that shows training acc over 1250 epochs](http://developer.download.nvidia.com/assets/Clara/Images/clara_pt_covid19_ct_lung_segmentation_train.png)

## Validation Performance
Validation mean dice over 1250 epochs.  

![Graph that shows validation mean dice getting higher over 1250 epochs until converging around 0.97](http://developer.download.nvidia.com/assets/Clara/Images/clara_pt_covid19_ct_lung_segmentation_val.png)

# Intended Use
The model needs to be used with NVIDIA hardware and software. For hardware, the model can run on any NVIDIA GPU with memory greater than 16 GB. For software, this model is usable only as part of Transfer Learning & Annotation Tools in Clara Train SDK container.  Find out more about Clara Train at the [Clara Train Collections on NGC](https://ngc.nvidia.com/catalog/collections/nvidia:claratrainframework).

**The Clara pre-trained models are for developmental purposes only and cannot be used directly for clinical procedures.**

# License
[End User License Agreement](https://developer.nvidia.com/clara-train-eula) is included with the product. Licenses are also available along with the model application zip file. By pulling and using the Clara Train SDK container and downloading models, you accept the terms and conditions of these licenses.

# References
[1] Liu, Siqi, et al. "3D Anisotropic Hybrid Network: Transferring Convolutional Features from 2D Images to 3D Anisotropic Volumes." In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 851-858). Springer, Cham. https://arxiv.org/pdf/1711.08580.pdf
