import os.path as osp
import random
import time
import datetime
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
import platform
import os
import torch.distributed as dist
import warnings
warnings.filterwarnings('ignore')


           
### 命令行运行
##  python -m torch.distributed.launch --nproc_per_node=4 p_2_7_SSD_training_DDP.py
##  python distributed.py -bk nccl -im tcp://10.10.10.1:12345 -rn 1 -ws 2
##  torchrun --nproc_per_node=4 p_2_7_SSD_training_DDP.py
##  sudo lsof -i:1234

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 设置分布式环境变量
world_size = 4
rank = int(os.environ.get('RANK', '0'))
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '1234'
# os.environ['RANK'] = str(rank)
# os.environ['WORLD_SIZE'] = str(world_size)
# os.environ['NCCL_SOCKET_IFNAME'] = 'eno1'
# os.environ['OMP_NUM_THREADS']=str(1)

# 初始化分布式环境
# init_method = 'tcp://{}:{}'.format('localhost', '1234')  
# torch.distributed.init_process_group(backend='nccl', init_method=init_method,world_size=4, rank=rank)

dist.init_process_group("nccl")
rank = dist.get_rank()
print(f"Start running basic DDP example on rank {rank}.")
device_id = rank % torch.cuda.device_count()
# print('device_id'+str(device_id))


from utils.ssd_model import make_datapath_list, VOCDataset, DataTransform, Anno_xml2list, od_collate_fn


# ファイルパスのリストを取得
rootpath = "../0_datasets/2_VOC2012/VOCdevkit/VOC2012/"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(
    rootpath)

# Datasetを作成
voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']
color_mean = (104, 117, 123)  # (BGR)の色の平均値
input_size = 300  # 画像のinputサイズを300×300にする

train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(
    input_size, color_mean), transform_anno=Anno_xml2list(voc_classes))

val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(
    input_size, color_mean), transform_anno=Anno_xml2list(voc_classes))


train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=True)

# DataLoaderを作成する
batch_size = 8

train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size*4, shuffle=(train_sampler is None),sampler=train_sampler, collate_fn=od_collate_fn,num_workers=6,pin_memory=True)

val_dataloader = data.DataLoader(
    val_dataset, batch_size=batch_size*4, shuffle=(val_sampler is None),sampler=val_sampler, collate_fn=od_collate_fn,num_workers=6,pin_memory=True)

# 辞書オブジェクトにまとめる
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

# DistributedSampler是PyTorch中用于在多个GPU或多个节点上进行分布式训练时对数据进行划分和加载的一个工具。
# 它的作用是将训练数据集分成多个子集，并且在每个子集上的数据加载器中只加载这个子集上的数据，从而避免在不同GPU或节点之间重复加载相同的数据。
# train_sampler = DistributedSampler(train_dataset)
# 创建分布式采样器
# 获取当前进程的排名和总进程数
world_size = 4
rank = int(os.environ.get('RANK', '0'))



from utils.ssd_model import SSD

# SSD300の設定
ssd_cfg = {
    'num_classes': 21,  # 背景クラスを含めた合計クラス数
    'input_size': 300,  # 画像の入力サイズ
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
    'feature_maps': [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ
    'steps': [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
    'min_sizes': [30, 60, 111, 162, 213, 264],  # DBOXの大きさを決める
    'max_sizes': [60, 111, 162, 213, 264, 315],  # DBOXの大きさを決める
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
}

# SSDネットワークモデル
net = SSD(phase="train", cfg=ssd_cfg)

# SSDの初期の重みを設定
# ssdのvgg部分に重みをロードする
vgg_weights = torch.load('../0_pre_model/vgg16_reducedfc.pth')
net.vgg.load_state_dict(vgg_weights)

# ssdのその他のネットワークの重みはHeの初期値で初期化


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:  # バイアス項がある場合
            nn.init.constant_(m.bias, 0.0)


# Heの初期値を適用
net.extras.apply(weights_init)
net.loc.apply(weights_init)
net.conf.apply(weights_init)

# GPUが使えるかを確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from utils.ssd_model import MultiBoxLoss

# 損失関数の設定
criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device_id)

# 最適化手法の設定
optimizer = optim.SGD(net.parameters(), lr=1e-3,
                      momentum=0.9, weight_decay=5e-4)



def reduce_val(val):
    world_size = dist.get_world_size()
    with torch.no_grad():
        dist.all_reduce(val, async_op=True)
        val /= world_size
    return val

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    
    time_name = datetime.datetime.now().strftime(f'%Y-%m-%d-%H-%M')

    # GPU是否可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用cuda：", device)

    # 网络放入GPU中
    net = net.to(device_id)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[device_id])
    # net.to(device)parallel.DistributedDataParallel

    # 网络稳定后，开启告诉运算
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # 迭代计数器
    iteration = 1
    epoch_train_loss = 0.0  # epochの損失和
    epoch_val_loss = 0.0  # epochの損失和
    logs = []

    # epochのループ
    for epoch in range(num_epochs+1):

        # 開始時刻 保存
        t_epoch_start = time.time()
        t_iter_start = time.time()

        print('-------------')
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        # 以epoch为单位进行训练和验证
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
                train_sampler.set_epoch(epoch)
                print('（train）')
            else:
                if((epoch+1) % 10 == 0):
                    net.eval()  # 每10轮进行一次验证
                    val_sampler.set_epoch(epoch)
                    print('-------------')
                    print('（val）')
                else:
                    # 検証は10回に1回だけ行う
                    continue
            
            
            # 加载数据
            for images, targets in dataloaders_dict[phase]:

                # GPU 
                images = images.to(device_id)
                targets = [ann.to(device_id)
                           for ann in targets]  # リストの各要素のテンソルをGPUへ

                # optimizer初始化
                optimizer.zero_grad()

                # 正向传播计算
                with torch.set_grad_enabled(phase == 'train'):
                    # 正向forward计算
                    outputs = net(images)

                    # 计算损失值
                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c

                    # 训练时候，开启反向传播
                    if phase == 'train':
                        loss.backward()  # 计算梯度
                        
                        # 如果梯度太大，计算会变得不稳定，因此，使用clip将梯度固定在2.0以内
                        nn.utils.clip_grad_value_(
                            net.parameters(), clip_value=2.0)

                        optimizer.step()  # 更新优化器参数

                        if (iteration % 10 == 0):  # 每10次迭代，显示一次loss
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print('迭代 {} || Loss: {:.4f} || 10iter: {:.4f} sec.'.format(
                                iteration, loss.item(), duration))
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item()
                        iteration += 1

                    # 验证时
                    else:
                        epoch_val_loss += loss.item()

        # 以epoch的phase为单位的loss和准确率
        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(
            epoch+1, epoch_train_loss, epoch_val_loss)) # # 每10轮进行一次验证
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        # 日志保存
        log_epoch = {'epoch': epoch+1,
                     'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("log_output.csv")

        epoch_train_loss = 0.0  # 重置每个epoch的损失值
        epoch_val_loss = 0.0  # 重置每个epoch的损失值

        save_path_dir = './weights/{}/'.format(time_name)
        if not os.path.exists(save_path_dir):
            os.makedirs(save_path_dir)
        # 保存网络
        if ((epoch+1) % 100 == 0):
            torch.save(net.module.state_dict(), save_path_dir+'ssd300_' +
                       str(epoch+1) + '.pth')
            
 

if __name__ == '__main__':
    num_epochs= 500
    train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

