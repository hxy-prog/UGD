# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from PIL import Image
import os
from .builder import DATASETS
from .custom import CustomDataset
import random
##sudo nvidia-smi drain -p 0000:0D:00.0 -m 1
@DATASETS.register_module()
class CIMLDataset(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = ('bg', 'fg')

    PALETTE = [[0, 0, 0], [255, 255, 255]]
    
    
    
    def __init__(self, **kwargs):
        super(CIMLDataset, self).__init__(
            reduce_zero_label=True,
            **kwargs)

    def count_non_empty_lines(self,filename):
        with open(filename, 'r', encoding='utf-8') as file:
            non_empty_lines = sum(1 for line in file if line.strip())
        return non_empty_lines
    def random_sample_lines(self,filename, sample_size):
        with open(filename, 'r', encoding='utf-8') as file:
            lines = [line for line in file if line.strip()]
        sampled_lines = random.sample(lines, sample_size)
        return sampled_lines
    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                        split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """
        if self.test_mode:   
            img_infos=[]
            for fn in os.listdir(os.path.join(img_dir,'img')):
                ##'filename': 'FantasticReality/ColorFakeImages/IMG_0016577_IMG_0007136.jpg',
                img_info = dict(filename='img/'+fn)
                img_info['realimgname']='realimg/'+fn

                img_info['ann'] = dict(seg_map='gt/'+fn.replace(img_suffix,seg_map_suffix))

                img_infos.append(img_info)

            return img_infos
        
        img_infos_list=[]
        txtfile=[
        '/home/hexingyang/DDP/segmentation/data/cimltxt/CASIA_v2.txt',
        '/home/hexingyang/DDP/segmentation/data/cimltxt/cm_COCO.txt',
        '/home/hexingyang/DDP/segmentation/data/cimltxt/sp_COCO.txt',
        '/home/hexingyang/DDP/segmentation/data/cimltxt/IMDspg.txt',
        '/home/hexingyang/DDP/segmentation/data/cimltxt/IMDsdg.txt'
        ]
        
        self.smallest = min([self.count_non_empty_lines(f) for f in txtfile])
        self.setnums=len(txtfile)
        
        for filename in txtfile:
            img_infos=[]
            with open(filename,'r') as f:
                for line in f:
                    txt=line.strip()
                    if not txt:
                        continue
                    img=txt.split(' ')[0]
                    seg_map=txt.split(' ')[1]
                    ##'filename': 'FantasticReality/ColorFakeImages/IMG_0016577_IMG_0007136.jpg',
                    img_info = dict(filename=img)
                    img_info['realimgname']=txt.split(' ')[2]

                    img_info['ann'] = dict(seg_map=seg_map)

                    img_infos.append(img_info)
                    
            img_infos_list.append(img_infos)
        return img_infos_list
        
        
        
        
        # img_infos=[]
        # for filename in txtfile:
        #     with open(filename,'r') as f:
        #         for line in f:
        #             txt=line.strip()
        #             if not txt:
        #                 continue
        #             img=txt.split(' ')[0]
        #             seg_map=txt.split(' ')[1]
        #             ##'filename': 'FantasticReality/ColorFakeImages/IMG_0016577_IMG_0007136.jpg',
        #             img_info = dict(filename=img)
        #             img_info['realimgname']=txt.split(' ')[2]

        #             img_info['ann'] = dict(seg_map=seg_map)

        #             img_infos.append(img_info)

        # return img_infos
    
    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)
        
        
    def __len__(self):
        """Total number of samples of data."""
        if self.test_mode:
            return super().__len__()
        return self.smallest*self.setnums
    
    def shuffle(self):
        ##[{'filename':1.jpg,'ann':{'seg_map':1.jpg}},]

        for imginfo in self.img_infos:
            random.shuffle(imginfo)
        random.shuffle(self.img_infos)
        
    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        if self.test_mode:
            return super().get_ann_info(idx)
        return self.img_infos[idx//self.smallest][idx%self.smallest]['ann']
    
    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        if self.test_mode:
            return super().prepare_train_img(idx)
        img_info = self.img_infos[idx//self.smallest][idx%self.smallest]

        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)

        self.pre_pipeline(results)

        return self.pipeline(results)
    
