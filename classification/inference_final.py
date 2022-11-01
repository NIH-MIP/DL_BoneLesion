#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import scipy
import os
import numpy as np
import cc3d
import torch
import torch.nn.functional as F
from torch.nn.utils import prune
import nibabel as nib
import torch.nn as nn
import skimage
import monai
from monai.transforms import Compose, LoadImaged ,ScaleIntensityRanged, Spacingd
from monai.data import Dataset, DataLoader



def cubize(mask_array,img_array,cube_stride):
    ########################### 
    ### Sliding Window Mask ###
    ###########################

    mask_windows = np.lib.stride_tricks.sliding_window_view(mask_array,window_shape=window_shape,writeable=True) #(H,W,D,i,j,k)
    mask_windows_sml = mask_windows[::cube_stride, ::cube_stride, ::int(cube_stride/2), ...] # stride masked windows
    mask_windows_sml = np.reshape(mask_windows_sml, [-1]+list(mask_windows_sml.shape[3:])) # Reshape to (HxWxD,i,j,k)
    mask_labels = np.max(mask_windows_sml,axis=(1,2,3)) # retrieve the class label for each cube (HxWxD,label/class)
    
    # create an empy sliding window matrix with the same size as the segmentation mask, this will be used for reconstructing the predictions
    zero_img = np.zeros(shape=mask_array.shape)
    zero_windows = np.lib.stride_tricks.sliding_window_view(zero_img,window_shape=window_shape,writeable=True) #(H,W,D,i,j,k)
    zero_windows_sml = zero_windows[::cube_stride, ::cube_stride, ::int(cube_stride/2), ...]
    zero_windows_sml = np.reshape(zero_windows_sml, [-1]+list(zero_windows_sml.shape[3:])) # Reshape to (HxWxD,i,j,k)
    
    
    ############################ 
    ### Sliding Window Image ###
    ############################    
    # Import image, apply sliding window, set stride, reshape        
    bin_mask_array = mask_array.copy()
    bin_mask_array[mask_array>0] = 1
    bin_mask_array = scipy.ndimage.binary_dilation(bin_mask_array,iterations=10)
    img_array_masked = img_array*bin_mask_array
    img_windows = np.lib.stride_tricks.sliding_window_view(img_array_masked,window_shape=window_shape,writeable=True)
    
    img_windows_sml = img_windows[::cube_stride, ::cube_stride, ::int(cube_stride/2), ...]
    img_windows_sml = np.reshape(img_windows_sml, [-1]+list(img_windows_sml.shape[3:])) # Reshape to (HxWxD,i,j,k)

    
    # Get Cube voxel criteria
    voxel_windows = np.lib.stride_tricks.sliding_window_view(bin_mask_array,window_shape=window_shape,writeable=True) #(H,W,D,i,j,k)
    voxel_windows_sml = voxel_windows[::cube_stride, ::cube_stride, ::int(cube_stride/2), ...]
    voxel_windows_sml = np.reshape(voxel_windows_sml, [-1]+list(voxel_windows_sml.shape[3:])) # Reshape to (HxWxD,i,j,k)
    cube_voxel_sum = np.sum(voxel_windows_sml,axis=(1,2,3)) # tells you how many voxels of segmentation are within each cube, this aids in size filtering
    
    
    return mask_labels, mask_windows , mask_windows_sml, img_windows, img_windows_sml, cube_voxel_sum, bin_mask_array, zero_windows, zero_windows_sml, zero_img



def lesion_agg(Priority_mask):
    
    labels_in = Priority_mask.astype(np.int32)
    labels_in[labels_in > 0] = 1
    connectivity = 6 #(2D) = 4,8 OR (3D) = 26,18,6
    delta = 0 # If delta=0, standard high speed processing. If delta>0, then
    # neighbor voxel values <= delta are considered the same component
    labels_out,N = cc3d.connected_components(labels_in,delta=delta,connectivity=connectivity,return_N=True)    
    
    
    
    for segid in range(1,N+1):
        extracted_mask = labels_out*(labels_out == segid) #binary mask from connected components
        extracted_mask[extracted_mask > 0] = 1
        masked_mask = Priority_mask.copy()*extracted_mask #get masked priority mask
        mask_info = []
        x,y,z = masked_mask.shape
        
        # Get the slice specific labels/classification for each Z increment
        for z_parse in range(z-1):
            slice_parse = masked_mask[:,:,z_parse]
            seg_tf=slice_parse.sum()

            if seg_tf != 0: # if the segmentation exists
                exists_mask = masked_mask[:,:,z_parse]
                Slice_labels = np.unique(exists_mask)
                try:
                    if len(Slice_labels) == 3: # if all 3 classes (cancer, benign, and background) appear in one segmentation, pick the max
                        x_df = pd.DataFrame(np.unique(exists_mask,return_counts=True)).transpose().iloc[1:,:]
                        Dom_Slice_label = int(x_df[x_df.iloc[:,1] == x_df.iloc[:,1].max()][0])
                        
                        if Dom_Slice_label == 0: # if the greatest sum of voxels come froms class 0 or background, pick the second highest/max class
                            Dom_Slice_label = int(x_df[x_df.iloc[:,1] != x_df.iloc[:,1].max()][0])
                        mask_info.append(Dom_Slice_label)
                        

                    elif len(Slice_labels) == 2: # if there are two separate classification predictions within a lesion, choose whichever has the most sum voxels
                        x_df = pd.DataFrame(np.unique(exists_mask,return_counts=True)).transpose()
                        Dom_Slice_label = int(x_df[x_df.iloc[:,1] == x_df.iloc[:,1].max()][0])

                        if Dom_Slice_label == 0: # if the greatest sum of voxels come froms class 0 or background, pick the second highest/max class
                            Dom_Slice_label = int(x_df[x_df.iloc[:,1] != x_df.iloc[:,1].max()][0])
                        mask_info.append(Dom_Slice_label)
                        

                    elif len(Slice_labels) == 1: # if there is only one class, it is obvious to pick the one
                        Dom_Slice_label = int(Slice_labels)
                        mask_info.append(Dom_Slice_label)
                        
                except: # if there are any weird exceptions or errors, default to labeling a lesion as benign (safe)
                        Dom_Slice_label = 1
                        mask_info.append(Dom_Slice_label)                    
                    
                    
        if (len(np.unique(mask_info)) == 0): # if there are no segmentation classes assigned to a segmentation, skip case
            continue
        else:
            # Aggregate all labels/classifications from the Z direction into a single lesion class
            Dom_Seg_label = np.unique(mask_info)[-1] # if any Z slice is predicted as metastases, label the entire lesion as met
            Priority_mask[masked_mask >= 1]
        
    return Priority_mask


def lesion_agg_final(Priority_mask):
    
    labels_in = Priority_mask.astype(np.int32)
    labels_in[labels_in > 0] = 1
    connectivity = 6 #(2D) = 4,8 OR (3D) = 26,18,6
    delta = 0 # If delta=0, standard high speed processing. If delta>0, then
    # neighbor voxel values <= delta are considered the same component
    labels_out,N = cc3d.connected_components(labels_in,delta=delta,connectivity=connectivity,return_N=True)    
    
    for segid in range(1,N+1):
        extracted_mask = labels_out*(labels_out == segid) #binary mask from connected components
        extracted_mask[extracted_mask > 0] = 1
        masked_mask = Priority_mask.copy()*extracted_mask #get masked priority mask
        mask_info = []
        x,y,z = masked_mask.shape
        for z_parse in range(z-1):
            slice_parse = masked_mask[:,:,z_parse]
            seg_tf=slice_parse.sum()

            if seg_tf != 0:
                exists_mask = masked_mask[:,:,z_parse]
                Slice_labels = np.unique(exists_mask)
                mask_info.append(Slice_labels[-1]) # dominant label is going to be the highest/max class in the matrix            
                    
        Dom_Seg_label = np.array(mask_info).max()
        Priority_mask[masked_mask >= 1] = Dom_Seg_label
    return Priority_mask



######################
# Creation data dict #
######################
# creation the data dictionary to feed into AI model
# merge_dict = {'image':  path_to_image,
#               'label': path_to_segmentation/mask}



###########################
# Augmentation/Transforms #
###########################

# Define transforms
valtest_transforms = Compose([LoadImaged(keys=['image','label']),
                        EnsureTyped(keys=["image", "label"]),
                        ScaleIntensityRanged(keys=['image'],a_min=-500,a_max=1300,b_min=0,b_max=1,clip=True),
                        Spacingd(keys=["image","label"], pixdim=1)])
# Define image dataset
merge_ds = Dataset(merge_dict, transform=valtest_transforms) # input dictionary of files
# create a data loader
merge_loader = DataLoader(merge_ds, batch_size=1, num_workers=4,shuffle=False) # this sliding window technique has not been adapted to batch sizes larger than 1, keep batch_size as 1



###################################
#### MODEL CREATION/PARAMETERS ####
###################################

#### DEFINE MODEL/TRAINING PARAMETERS ####
LR = 3e-5
dropout_prob = .4
num_prediction_classes = 2
window_shape = (64,64,32)
cube_stride = 32

##### DEFINE MODEL #####
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=num_prediction_classes,progress=True,dropout_prob=dropout_prob).to(device)
loss_function = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), LR)

###############################
##### LOAD MODEL WEIGHTS ######
##
##        
##        
##


#########################
#### MODEL INFERENCE ####
#########################
with torch.no_grad():
    model.eval()

    ## Data loader
    for batch_data in merge_loader:
        mask_array = np.array(batch_data['label'][0]).astype(np.int32)
        img_array = np.array(batch_data['image'][0])
        # convert both image and mask array into sliding window cubes
        mask_labels, mask_windows , mask_windows_sml, img_windows, img_windows_sml, cube_voxel_sum, bin_mask_array, zero_windows, zero_windows_sml, zero_img = cubize(mask_array,img_array,cube_stride)
        
        
        #########################
        ### AI LOGIC/WORKFLOW ###
        #########################
        # Iterate through every single cube and: 
        # (1) does the cube meet inclusion criteria (has a true label, large enough). If true, move to (2) 
        # (2) feed cube into model to get cube-level prediction
        # (3) reconstruct the sliding window with cube-level predictions
        # (4) mask with original segmentations and aggregate cube-level predictions into lesion-level predictions
        # (5) resample to original CT spacing, final aggregation similar to step (4),and done!
        
        
        iteration_shape = mask_windows_sml.shape
        for HWD in range(iteration_shape[0]):
            mask_parse = mask_windows_sml[HWD,...]
            cube_parse = img_windows_sml[HWD,...]
            label_parse = int(mask_labels[HWD,...]) # Grab the label of the mask cube
            voxelsum_parse = cube_voxel_sum[HWD,...]


            if (label_parse != 0) and (voxelsum_parse>1500) and (label_parse != 5): # (1) does the cube meet inclusion criteria (has a true label, large enough). If true, move to (2) 

                cube_tensor = torch.tensor(cube_parse).type(torch.cuda.FloatTensor).to(device)
                outputs = model(cube_tensor[None,None,...]) # (2) feed cube into model to get cube-level prediction
                pred_outputs = outputs
                outputs_sig = torch.sigmoid(pred_outputs).type(torch.FloatTensor).to(device) #map predictions to between 0 and 1 using sigmoid function
                pred_class = int(outputs_sig.argmax())+1 #prediction class is equal to its matrix location plus 1 (location 0 = benign, location 1 = metastatic)
                

                #############################################
                #### Write prediction to empty cube array ###
                #############################################
                mask_copy = mask_parse.copy()
                mask_copy[mask_copy>0] = pred_class
                
                iteration_shape = mask_windows.shape
                match = 0 # (3) reconstruct the sliding window with cube-level predictions
                for H in range(0,iteration_shape[0],int(cube_stride)):
                    for W in range(0,iteration_shape[1],int(cube_stride)):
                        for D in range(0,iteration_shape[2],int(cube_stride/2)):
                            
                            if match == 0: # Look through the CT image for the matching location
                                if (mask_windows[H,W,D,...] == mask_parse).all(): # we will parse through the entire image and look for the CT cube which is a perfect match (gives us H,W,D to assign the prediction cube)
                                    zero_windows[H,W,D,...] = mask_copy
                                    match += 1
                                else:
                                    continue
                            else:
                                break
                                  
        output_array = zero_img*bin_mask_array #remask the output
        output_array = lesion_agg(output_array) #aggregate lesion predictions # (4) aggregate cube-level predictions into lesion-level predictions
        output_array = skimage.transform.resize(output_array, output_shape = (XYspacing,XYspacing,Zspacing), clip=True, preserve_range=True) # transform to original spacing #(5)
        output_array = lesion_agg_final(output_array) #aggregate any extra segmentation added from resampling  
        new_mask = nib.Nifti1Image(output_array, np.eye(4)) # creation of Nifti image