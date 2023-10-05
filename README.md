##基于深度学习的快速算法到FDTD三维隐式解校正网络(IETNet)
###一、IETNet简介
    IETNet采用端到端训练方式，网络包含两个模块： 
    三位隐式解空间信息感知模块：3D Spatial Information Perception Module (3D-SIP)
                输入原始三维数据格式为： (时相, H*W*L, 4(X,Y,Z,F))
    三维隐式解特征表达模块：Implicit Transformer (ImpFormer)
                输出预测数据格式为：    (时相, H*W*L)
###二、各文件功能
####1.utils:  存放自己写好的自定义的函数
    data_processing.py:     用于数据读取
    train_epoch_iet.py:     用于IETNet训练
    valid_epoch.py:         用于IETNet验证
    utils.py:               包含一些功能函数，如加载配置文件、保存训练模型、计算损失函数等
####2.data:   存放训练数据集
####3.models: 存放模型
    IETNet.py:           IETNet主结构
    IETNet_utils.py:     IETNet的附属模块
 
####4.venv:         存放python虚拟环境(不重要)
####5.checkpoints:  存放训练得到的模型文件
####6.主文件夹下：
    train_IETNet.py:    IETNet训练的主函数文件
    test.py:            用于最终测试，得到校正结果
    config.yaml:        参数、网络基本配置设置
    README.md:          说明文件

###三、训练步骤
####1. config.yaml配置文件中需要修改的参数：
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

####2.训练IETNet
    将训练集存放至root所在路径
    设置所使用的GPU
    运行  train_IETNet.py
    设置网络batchsize设置为16，epochs设置为50
    训练完成后，得到模型IETNet.pkl

###四、测试步骤
    运行   test.py
    修改file_input和file_output为测试数据路径
    修改file_predict为测试数据结果保存路径
    修改length为测试数据时序长度
    输出loss(采用L1损失）
    
    
    

    

