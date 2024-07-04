import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from utils.utils import show_yaml,show_args, load_yaml, save_state_dict, build_save_dir, load_state_dict, get_optim, get_scheduler
from utils.data_processing import MyDataset
from utils.train_epoch_ietnet import train_epoch
from utils.valid_epoch import valid_epoch
from models.ietnet import create_model_ietnet
import xlwt
def get_args():
    """ 执行参数 """
    parser = argparse.ArgumentParser(description='IETNet')

    parser.add_argument('--cfg_path', default='configs.yaml', type=str, help="配置文件路径")

    parser.add_argument('--device', default='0', nargs='+', help="训练GPU id")

    parser.add_argument('--local_rank', default=-1, type=int, help='多GPU训练固定参数')
    print('cuda available with GPU:', torch.cuda.get_device_name(0))
    return parser.parse_args()

def main(args):
    # 加载配置文件configs.yaml
    cfgs = load_yaml(args.cfg_path)

    #设置GPU
    if isinstance(args.device, (int, str)):
        device_ids = [args.device]
    elif isinstance(args.device, (list, tuple)):
        new_list = [list(map(int, item)) for item in args.device]
        device_ids =  [int(x) for sublist in new_list for x in sublist]
    device = torch.device("cuda:{}".format(device_ids[args.local_rank]))

    #加载模型
    model = create_model_ietnet(cfgs)
    #如果有预训练模型，可以继续训练
    if cfgs['Train']['resume']:
        model, *unused = load_state_dict(cfgs['Train']['resume'], model)
    model.to(device)
    # 终端打印配置文件参数
    print('IETNet参数配置：')
    show_yaml(trace=print, args=cfgs)

    # 创建文件夹, 保存训练的权重
    if cfgs['Train']['resume']:
        save_dir = os.path.dirname(cfgs['Train']['resume'])
    else:
        #保存到 checkpoints 文件夹下
        save_dir = build_save_dir("checkpoints")


    #设置优化器optimizer
    optimizer = get_optim(cfgs['Train']['optimizer'])(params=model.parameters(), lr=cfgs['Train']['lr'],
                                                        weight_decay=cfgs['Train']['weight_decay'])
    scheduler = get_scheduler(optimizer, step_size=cfgs['Train']['scheduler_step'], gamma=cfgs['Train']['scheduler_gamma'])

    #加载训练和验证数据集
    dataset = MyDataset(cfgs, train=True)
    train_len = int(cfgs['Data']['split'] * len(dataset))

    train_set, valid_set = random_split(dataset, [train_len, len(dataset) - train_len])

    train_sampler = None
    shuffle = True
    train_loader = DataLoader(train_set, shuffle=shuffle, batch_size=cfgs['Train']['batch_size'],
                              num_workers=cfgs['Train']['n_worker'], pin_memory=False,
                              sampler=train_sampler)
    valid_loader = DataLoader(valid_set, shuffle=shuffle, batch_size=cfgs['Train']['batch_size'],
                              num_workers=cfgs['Train']['n_worker'], pin_memory=False)

    #加载训练参数
    epochs = cfgs['Train']['epochs']
    args.epochs = epochs
    best_loss = 1e6
    workbook = xlwt.Workbook(encoding='utf-8')
    # 创建一个worksheet存储loss
    worksheet = workbook.add_sheet('loss_value')
    excelnum1 = 1
    excelnum2 = 1

    #开始训练
    for epoch in range(epochs):
        print('*' * 10, ' epoch [{}/{}] '.format(epoch, epochs), '*' * 10)
        #训练
        train_loss, model, optimizer, excelnum1 = train_epoch(excelnum1, worksheet, args=args, model=model,
                                                     loader=train_loader,
                                                     epoch=epoch, device=device,
                                                     optimizer=optimizer)
        scheduler.step()

        #验证
        val_loss, excelnum2 = valid_epoch(excelnum2, worksheet,args=args,model=model,
                                          loader=valid_loader, epoch=epoch, device=device)


        workbook.save('loss_ietnet.xls')
        # 保存最优模型
        if val_loss.avg < best_loss:
            best_loss = val_loss.avg
            save_state_dict(
                path=os.path.join(save_dir, 'IETNet.pkl'),
                model=model,
                cfgs=args,
                epoch=epoch,
                optim=optimizer,
                min_loss=best_loss)
            print("save to: '{}', min loss: {:.8f}".format(
                os.path.join(save_dir, 'IETNet.pkl'), best_loss))
        else:
            print("valid min loss: {:8f}".format(best_loss))
        print()


if __name__ == '__main__':
    args = get_args()
    show_args(trace=print, args=args)
    main(args)