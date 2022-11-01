import pandas as pd
from glob import glob
import scipy
import os
import numpy as np
import cc3d
import nibabel as nib
import skimage
import monai
from monai.transforms import Compose, LoadImaged ,ScaleIntensityRanged, Spacingd, Rand3DElasticd
from monai.data import Dataset, DataLoader


###############################################
#### LOAD TRAINING/VALIDATION/TESTING DATA ####
###############################################
patient_level_df = pd.read_csv('/path/to/csv/') #contains 1 column for with location of CT.nii.gz and 1 column for location of Lesion_mask.nii.gz
Train_dict = [{"image": image_name, "label": label_name} for image_name, label_name in zip(patient_level_df.CT_path, Train_mask_input.mask_path)]
cube_save_dir = '/path/to/cube/dir/'


###########################
# Augmentation/Transforms #
###########################
train_transforms = Compose([LoadImaged(keys=['image','label']),
                            ScaleIntensityRanged(keys=['image'],a_min=-500,a_max=1300,b_min=0,b_max=1,clip=True),
                            Rand3DElasticd(keys = ['label'],prob=.75,sigma_range=(3,5),magnitude_range=(1,2),padding_mode='zeros'),
                           ])


#########################################
# Training/Validation/Test Data Loaders #
#########################################

# Define image dataset
train_ds = Dataset(Train_dict, transform=train_transforms)
# create a data loader
train_loader = DataLoader(train_ds, batch_size=1, num_workers=4)

train_stride =16
window_shape = (64,64,32)
iterations = 5
dilate = 10
cube_cntr = 0

for iteration in range(iterations):
    for batch_data in train_loader:

        ####################################################
        print('Start New Sliding Window Training')
        filename = batch_data['image_meta_dict']['filename_or_obj'][0] # get the filename from the imaging metdata
        anon_path = patient_level_df[patient_level_df[CT_col] == filename]['anon_path'].item()
        print(filename)    


        # Import mask, apply sliding window, set stride, reshape
        #distort_transform(batch_data)
        mask_array = np.array(batch_data['label'][0])
        mask_windows = np.lib.stride_tricks.sliding_window_view(mask_array,window_shape=window_shape,writeable=True) #(H,W,D,i,j,k)
        mask_windows_sml = mask_windows[::train_stride, ::train_stride, ::int(train_stride/2), ...]
        mask_windows_sml = np.reshape(mask_windows_sml, [-1]+list(mask_windows_sml.shape[3:])) # Reshape to (HxWxD,i,j,k)

        mask_labels = np.max(mask_windows_sml,axis=(1,2,3)).astype(int)
        label_filter = np.array(mask_labels>0)
        filtered_mask_windows = mask_windows_sml[label_filter,...]
        filtered_labels = np.max(filtered_mask_windows,axis=(1,2,3))

        img_array = np.array(batch_data['image'][0])
        # Import image, apply sliding window, set stride, reshape        
        bin_mask_array = mask_array.copy()
        bin_mask_array[mask_array>0] = 1

        # Import image, apply sliding window, set stride, reshape        
        bin_mask_array_dilated = scipy.ndimage.binary_dilation(bin_mask_array,iterations=dilate)
        img_array_masked = img_array*bin_mask_array_dilated
        img_windows = np.lib.stride_tricks.sliding_window_view(img_array_masked,window_shape=window_shape)
        img_windows = img_windows[::train_stride, ::train_stride, ::int(train_stride/2), ...] 
        img_windows = np.reshape(img_windows, [-1]+list(img_windows.shape[3:])) # Reshape to (HxWxD,i,j,k)
        filtered_img_windows = img_windows[label_filter,...]


        for img_window_parse in range(len(filtered_img_windows)):

            window_save = np.array(filtered_img_windows[img_window_parse])
            voxels = (window_save>0).sum()

            nifti_window = nib.Nifti1Image(window_save,np.eye(4))
            cube_name = "TrainCube_"+str(img_window_parse)+'_Dilation_'+str(dilate) +'_Class_'+str(int(filtered_labels[img_window_parse])) + '_VoxelNum_' +str(voxels)+'_Iteration_'+str(iteration)+ '_CubeNum_'+ str(cube_cntr) +'_.nii.gz'
            cube_namepath = cube_save_dir + cube_name
            nib.save(nifti_window,cube_namepath)
            print(cube_namepath)
            cube_cntr += 1


        print(len(filtered_img_windows))