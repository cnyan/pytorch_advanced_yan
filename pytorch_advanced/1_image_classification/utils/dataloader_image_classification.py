import glob
import os.path as osp
import torch.utils.data as data
from torch.utils.data import Dataset,DataLoader
from torchvision import models, transforms
from PIL import Image

import platform
sys = platform.system()

# 图像预处理类
class ImageTransform():
    
    def __init__(self, resize, mean, std) -> None:
        
        self.data_transform = {
            # 经过transforms操作，即使同一张图片，在每个epoch中都会自动生成稍有不同的图像，简单且高效的提升网络性能
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                # scale将图像在0.5~1.0的范围进行放大或者缩小，同时将图像的宽高比在3/4~4/3变化，对图像进行拉伸处理，再按照resize进行剪裁工作
                transforms.RandomHorizontalFlip(),  # 以50%的概率对图像进行左右翻转处理
                transforms.ToTensor(),  # 图像转为张量，并归化到0-1，由[h,w,c]转为[c,h,w]
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
        
    def __call__(self, img, phase='train') :
        return self.data_transform[phase](img)
    

def make_datapath_list(phase='train')->list:
    rootpath = '../0_datasets/1_hymenoptera/hymenoptera_data/'
    target_path = osp.join(rootpath+phase+'/**/*.jpg')
    path_list = glob.glob(target_path)
        
    return path_list



class HymenopteraDataSet(Dataset):
    """_summary_

    Args:
        Dataset (_type_): 创建DataSet，对图像进行预处理、getitem()、len()
    """

    def __init__(self,file_list,transform=None,phase='train') -> None:
        """_summary_

        Args:
            file_list (_type_): 图片路径
            transform (_type_, optional): 预处理类的实例
            phase (str, optional): train or val
        """
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        
        
    def __len__(self)->int:
        return len(self.file_list)
    
    def __getitem__(self, index) -> tuple:
        image_path = self.file_list[index] # 文件路径
        img = Image.open(image_path)
        
        img_transformed = self.transform(img,self.phase) # torch.size([3,224,224])
       
        # 获取标签
        # if self.phase == 'train':
        #     y_lable = str(image_path[30:34]) 
        # elif self.phase == 'val':
        #     y_lable = str(image_path[28:32])
        # else:
        #     y_lable = 'None'
        
        # 获取标签
        if sys == 'Windows':
            y_lable = image_path.split('\\')[-2]
        else:
            y_lable = image_path.split('/')[-2]
        
        #将标签转为数字
        if y_lable == 'ants':
            y_lable = 0
        elif y_lable == 'bees':
            y_lable = 1
        else:
            y_lable = -1
            
        return img_transformed,y_lable
    
