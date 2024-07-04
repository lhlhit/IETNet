## On Improving Accuracy of Implicit FDTD Methods with Transformer-based Deep Learning Network
Fast Algorithm Based on Deep Learning for FDTD 3D Implicit Solution Correction Network (IETNet)

The full code will be open source after the publication of the paper. If you have any questions, please contact at lhl_hit@hotmail.com

### 一、Introduction
    IETNet adopts an end-to-end training approach and the network consists of two modules:
     1. 3D Spatial Information Perception Module (3D-SIP) 
         The input format of the original 3D data is: (time phase, H*W*L, 4(X,Y,Z,F))
     2. Implicit Transformer (ImpFormer)
         The output format of the predicted data is: (time phase, H*W*L)
 ### 二、Function of Each File 
 #### 1.utils:  Stores custom-written functions 
    data_processing.py:     Used for data reading
    train_epoch_iet.py: Used for training IETNet
    valid_epoch.py: Used for validating IETNet
    utils.py: Contains utility functions such as loading configuration files, saving the trained model, calculating loss  functions, etc.
#### 2.data:   Stores the training dataset
#### 3.models: Stores models
IETNet.py: Main structure of IETNet
IETNet_utils.py: Auxiliary modules of IETNet
 
#### 4.venv:        Stores the Python virtual environment (not important)
#### 5.checkpoints:  Stores model files obtained from training
#### 6.main folder:
train_IETNet.py: Main function file for training IETNet
test.py: Used for final testing to obtain correction results
config.yaml: Parameter and basic network configuration settings
README.md: Documentation file
### 三、Training Steps
#### 1. Parameters to modify in the config.yaml 
    Train:
epochs: Number of training iterations, modify as needed
batch_size: Adjust according to GPU memory
root: Path where the training set is stored
split: Ratio for splitting the training set and validation set
    IETNet:
loc_size: Dimensions of the 3D data
n_heads: Number of heads in the multi-head attention mechanism
n_layers: Number of ImpFormer Encoders
Time_Length: Number of time sequences
#### 2.Training IETNet
Store the training set at the path specified by root
Set the GPU to be used
Run train_IETNet.py
Set the network batch_size to 16 and epochs to 50
Upon completion of training, the model IETNet.pkl will be obtained

### 四、Testing Steps
 Run test.py
Modify file_input and file_output to the paths of the test data
Modify file_predict to the path where the test data results will be saved
Modify length to the time sequence length of the test data
Output the loss (using L1 loss)
