import argparse
import torch
import torch.nn as nn
import os
from utils.utils import load_yaml
from utils.data_processing import load_data
from utils.utils import load_state_dict
from models.ietnet import create_model_ietnet
import scipy.io
def get_args():
    parser = argparse.ArgumentParser("Test_IETNet")

    parser.add_argument('--config', default='configs.yaml', type=str,
                        help="配置文件路径")

    parser.add_argument('--file_input', default=r'E:\LHL\0804\supersure\data\0920_2', type=str,
                        help="Input file path")

    parser.add_argument('--file_output', default=r'E:\LHL\0804\supersure\data\0920_2', type=str,
                        help="Output file path")

    parser.add_argument('--file_predict', default='D:\\LHL\\data\\predict\\chushi', type=str,
                        help="Predict file path")

    parser.add_argument('--length', default=250, type=int,
                        help="选择时序长度")

    parser.add_argument('--device', default=0, type=int,
                        help="GPU id")

    return parser.parse_args()

def Test(args):
    # 加载模型
    config = load_yaml(args.config)
    device = torch.device(args.device)
    model = create_model_ietnet(config)
    model, *unuse = load_state_dict(
        config['Train']['predict_model'],
        model)
    model.eval().to(device)
    # 加载测试数据
    files = os.listdir(args.file_input)  # 扫描所有文件
    nums_ = list(sorted(set([int(f.split('_')[1]) for f in files])))
    string = "Sample_{index}_{mode}_{label}.mat"
    for n in nums_:
        if (string.format(index=n, mode="Input", label=config['Data']['in_label']) in files):
            start_time0 = time.time()
            print(string.format(index=n, mode="Input", label=config['Data']['in_label']))
            print(string.format(index=n, mode="Output", label=config['Data']['out_label']))
            data_in = load_data(os.path.join(args.file_input, string.format(index=n, mode="Input", label=config['Data']['in_label'])))[
                'SampleIn_Ez'].astype("float32")
            data_out = load_data(
                os.path.join(args.file_input, string.format(index=n, mode="Output", label=config['Data']['out_label'])))[
                'SampleOut_Ez'].astype("float32")

            data_in = torch.tensor(data_in)
            data_in = data_in.to(device)
            data_out = data_out[:, :, 3:]
            data_out = torch.tensor(data_out)
            data_out = data_out.to(device)
            data_in0 = data_in[:, :, 3:]
            loss_fn = nn.L1Loss()
            loss_1 = loss_fn(data_in0, data_out)
            print("loss_in_out  :%.8f" % loss_1)

            data_in = data_in.unsqueeze(dim=0)
            data_in_p = model(data_in)
            data_in_p = data_in_p.permute(1, 2, 0)
            loss_0 = loss_fn(data_in_p, data_out)
            print("loss_pred_out:%.8f" % loss_0)
            end_time0 = time.time()
            elapsed_time0 = end_time0 - start_time0
            print(f"预测单个样本时间：{elapsed_time0} 秒")
            print("----------------------------------------------" )



import time
if __name__ == "__main__":
    start_time = time.time()
    args = get_args()
    Test(args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程序运行时间：{elapsed_time} 秒")