from torch.utils.data import Dataset
import albumentations as A
import cv2
import os
import torch
import torch.nn.functional as F
import numpy as np
class ManipulationDataset(Dataset):
    def __init__(self):
        self.maskpath=[]
        self.image_size=512
        p1='/home/hexingyang/MMFusion-IML-main/data/Casiav2/mask'
        for f in os.listdir(p1):
            self.maskpath.append(os.path.join(p1,f))
            self.maskpath.append('/home/hexingyang/MMFusion-IML-main/data/zeromask.png')
        p2='/home/hexingyang/MMFusion-IML-main/data/FantasticReality/masks'
        for f in os.listdir(p2):
            self.maskpath.append(os.path.join(p2,f))
        self._init_transforms()
    def __len__(self):
        return len(self.maskpath)
    
    def _init_transforms(self):
        self.image_transforms_train = A.Compose([
            A.RandomScale(scale_limit=(-0.5, 0.5), p=0.5),
            A.PadIfNeeded(min_height=self.image_size, min_width=self.image_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1),
            A.RandomCrop(height=self.image_size, width=self.image_size, p=1),
        ])


    def __getitem__(self, index):

        mask = cv2.imread(self.maskpath[index], cv2.IMREAD_GRAYSCALE)
        res = self.image_transforms_train(image=np.zeros_like(mask), mask=mask)
        mask = res['mask']
        mask = mask / 255.0

        mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
        
        mask = (mask > 0.1).astype(int)
        
        # if np.random.rand()<0.5:
        #     noise_prob=  np.random.rand() * 0.05
        #     noise = np.random.rand(128, 128) < noise_prob
            
        #     noise = noise.astype(int)  # 确保噪声矩阵是整数类型

        #     # 使用 XOR 操作添加噪声
        #     noisy_mask = mask | noise 
            
        #     cv2.imwrite('00.png',noisy_mask*255)

        #     return torch.tensor(noisy_mask)
        # else:
        #     return torch.tensor(mask)
        return torch.tensor(mask)