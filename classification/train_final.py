# Contributor: Mason Belue
# Email: air@nih.gov
# Nov 1, 2022
#
# By downloading or otherwise receiving the SOFTWARE, RECIPIENT may 
# use and/or redistribute the SOFTWARE, with or without modification, 
# subject to RECIPIENT’s agreement to the following terms:
# 
# 1. THE SOFTWARE SHALL NOT BE USED IN THE TREATMENT OR DIAGNOSIS 
# OF CANINE OR HUMAN SUBJECTS.  RECIPIENT is responsible for 
# compliance with all laws and regulations applicable to the use 
# of the SOFTWARE.
# 
# 2. The SOFTWARE that is distributed pursuant to this Agreement 
# has been created by United States Government employees. In 
# accordance with Title 17 of the United States Code, section 105, 
# the SOFTWARE is not subject to copyright protection in the 
# United States.  Other than copyright, all rights, title and 
# interest in the SOFTWARE shall remain with the PROVIDER.   
# 
# 3.	RECIPIENT agrees to acknowledge PROVIDER’s contribution and 
# the name of the author of the SOFTWARE in all written publications 
# containing any data or information regarding or resulting from use 
# of the SOFTWARE. 
# 
# 4.	THE SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED 
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT 
# ARE DISCLAIMED. IN NO EVENT SHALL THE PROVIDER OR THE INDIVIDUAL DEVELOPERS 
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
# THE POSSIBILITY OF SUCH DAMAGE.  
# 
# 5.	RECIPIENT agrees not to use any trademarks, service marks, trade names, 
# logos or product names of NCI or NIH to endorse or promote products derived 
# from the SOFTWARE without specific, prior and written permission.
# 
# 6.	For sake of clarity, and not by way of limitation, RECIPIENT may add its 
# own copyright statement to its modifications or derivative works of the SOFTWARE 
# and may provide additional or different license terms and conditions in its 
# sublicenses of modifications or derivative works of the SOFTWARE provided that 
# RECIPIENT’s use, reproduction, and distribution of the SOFTWARE otherwise complies 
# with the conditions stated in this Agreement. Whenever Recipient distributes or 
# redistributes the SOFTWARE, a copy of this Agreement must be included with 
# each copy of the SOFTWARE.

import pandas as pd
import scipy
import glob as glob
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


# #####################################################
# #### LOAD TRAINING/VALIDATION/TESTING DATA CUBES ####
# #####################################################

cubes_arr = []
cube_globs = glob('/path/to/cubes/'+'*')
for x in cube_globs:
    # sample cube names
    cube_name = x.split('/')[-1] #"TrainCube_0_Dilation_5_Class_1_VoxelNum_5143_Iteration_3_CubeNum_50_.nii.gz'
    cube_type = cube_name.split('_')[0][:-4]
    cube_class = int(cube_name.split('_')[5])
    cube_dilation = int(cube_name.split('_')[3])
    voxel_num = int(cube_name.split('_')[7])
    iteration =  int(cube_name.split('_')[9])
    cube_num =  int(cube_name.split('_')[11])

    temp = [x,cube_name,cube_num,cube_dilation,cube_type,cube_class,voxel_num,iteration]
    cubes_arr.append(temp)
        
cube_df = pd.DataFrame(cubes_arr,columns = ['anon_path','cube_name','cube_number','cube_dilation','cube_type','cube_class','','voxel_num'])
cube_train_df = cube_df[cube_df.voxel_num>1000][cube_df.cube_type == 'Train'] # pull only cubes of specific size threshold and only training cubes
cube_train_dict = [{"image": image_name, "label": label} for image_name, label in zip(cube_train_df.anon_path, cube_train_df.cube_class)]




###########################
# Augmentation/Transforms #
###########################

# Define transforms
train_transforms = Compose([LoadImaged(keys=['image']),
                            AddChanneld(keys=['image']),
                            RandGaussianNoised(keys=['image']),
                            RandFlipd(keys=['image'], prob=.5),
                            RandRotated(keys=['image'], prob=.5),
                           ])


#########################################
# Training/Validation/Test Data Loaders #
#########################################

# Define image dataset
train_ds = Dataset(cube_train_dict, transform=train_transforms)
# create a data loader
train_batch = 150
train_loader = DataLoader(train_ds, batch_size=train_batch, num_workers=4,pin_memory=True,shuffle=True)



###################################
#### MODEL CREATION/PARAMETERS ####
###################################

#### DEFINE MODEL/TRAINING PARAMETERS ####
LR = 1e-3
dropout_prob = 0
num_prediction_classes = 2
window_shape = (64,64,32)
validation_stride = 32


##### DEFINE MODEL #####

model_load_name = 'DenseNet'
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1,out_channels=num_prediction_classes,progress=True,dropout_prob=dropout_prob).to(device)
loss_function = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), LR)


# load model weights
tensor_dict = torch.load('/path/to/model/weights/',map_location=device)
model.load_state_dict(tensor_dict)
model.to(device)



###############################
#### START TRAINING EPOCHS ####
###############################
# start a typical PyTorch training
epoch_num = 500

epoch_loss_values = list()
validation_loss_values = list()
pt_loss_values = list()
metric_values = list()
writer = SummaryWriter()

TL_count = 0
for epoch in range(epoch_num):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{epoch_num}")
    model.train()
    train_loss = 0
    train_step = 0
    ### use ctr+/ to comment in/out blocks of code
    
    
    ##################
    #### TRAINING ####
    ##################
    print('Start New Sliding Window Training')
    for batch_data in train_loader:
        
        
        ####################################################
        # Import mask, apply sliding window, set stride, reshape
        batch_labels = batch_data['label']
        batch_onehot_labels = torch.nn.functional.one_hot(batch_labels).type(torch.FloatTensor).to(device)
        # Import image, apply sliding window, set stride, reshape        
        batch_images = batch_data['image'].to(device)
        outputs = model(batch_images)
        pred_outputs = outputs
        outputs_sig = torch.sigmoid(pred_outputs).type(torch.FloatTensor).to(device)
        
        loss = loss_function(outputs_sig, batch_onehot_labels)  
        item_loss = loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += item_loss
        train_step += 1
        print(item_loss,train_loss)
        
        
    train_epoch_loss = train_loss/train_step
    print(f"epoch {epoch + 1} average loss: {train_epoch_loss:.4f}")
