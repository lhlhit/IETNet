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



***********************************************************************************************
  Chinese Version
***********************************************************************************************
## 基于深度学习的快速算法到FDTD三维隐式解校正网络(IETNet)
### 一、IETNet简介
    IETNet采用端到端训练方式，网络包含两个模块：
        1. 三位隐式解空间信息感知模块：3D Spatial Information Perception Module (3D-SIP)
            输入原始三维数据格式为： (时相, H*W*L, 4(X,Y,Z,F))
        2. 三维隐式解特征表达模块：Implicit Transformer (ImpFormer)
            输出预测数据格式为：    (时相, H*W*L)
### 二、 各文件功能
#### 1.utils:   存放自己写好的自定义的函数
    data_processing.py:     用于数据读取
    train_epoch_iet.py:     用于IETNet训练
    valid_epoch.py:         用于IETNet验证
    utils.py:               包含一些功能函数，如加载配置文件、保存训练模型、计算损失函数等
#### 2.data:   存放训练数据集
#### 3.models: 存放模型
    IETNet.py:           IETNet主结构
    IETNet_utils.py:     IETNet的附属模块
 
#### 4.venv:         存放python虚拟环境(不重要)
#### 5.checkpoints:  存放训练得到的模型文件
#### 6.主文件夹下：
    train_IETNet.py:    IETNet训练的主函数文件
    test.py:            用于最终测试，得到校正结果
    config.yaml:        参数、网络基本配置设置
    README.md:          说明文件

### 三、训练步骤
#### 1. config.yaml配置文件中需要修改的参数：
    Train:
        epochs:               训练迭代次数，根据需要修改 
        batch_size:           根据显存调整 
        root：                训练集存放路径
        split:                训练集和验证集分割比例
    IETNet:
        loc_size:             三维数据维度  
        n_heads:              多头注意力机制头个数
        n_layers:             ImpFormer Encoder个数
        Time_Length:          时序数

#### 2.训练IETNet
    将训练集存放至root所在路径
    设置所使用的GPU
    运行  train_IETNet.py
    设置网络batchsize设置为16，epochs设置为50
    训练完成后，得到模型IETNet.pkl

### 四、测试步骤
    运行   test.py
    修改file_input和file_output为测试数据路径
    修改file_predict为测试数据结果保存路径
    修改length为测试数据时序长度
    输出loss(采用L1损失）![image](https://github.com/yikuaixigua/IETNet/assets/52653618/9fb68767-02b7-4c27-9d56-fd7f16bc3bb7)

