# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings

import mmcv
import numpy as np
import torch
from PIL import Image

from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from datetime import datetime
from .imlmetrics import computeLocalizationMetrics
import cv2
import os

from scipy.ndimage import label
import sys
sys.path.append('/home/hexingyang/FastSAM') 
# from fastsam import FastSAM, FastSAMPrompt
import cv2
import torch
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from sklearn.metrics import roc_auc_score

import random
from PIL import Image
import matplotlib.pyplot as plt
import copy
from mmseg.utils import build_ddp, build_dp, get_device, setup_multi_processes
# smodel = FastSAM('/home/hexingyang/FastSAM/weights/FastSAM-x.pt')
DEVICE =get_device()

def calculate_iou(gt, pred):
    
    gt=(gt >= 0.5).astype(np.uint8)
    pred=(pred >= 0.5).astype(np.uint8)
    # 将输入转为布尔类型
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    # 求交集
    intersection = np.logical_and(gt, pred).sum()
    
    # 求并集
    union = np.logical_or(gt, pred).sum()

    # 计算IoU
    iou = intersection / union if union != 0 else 0
    
    return iou

def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.
    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5,
                    pre_eval=False,
                    format_only=False,
                    format_args={}):
    """Test with single GPU by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.
    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler
    # f1 = []
    rocauc=[]
    f1th = []
    f1=[]
    iouth=[]
    for batch_indices, data in zip(loader_indices, data_loader):

        with torch.no_grad():

            result = model(return_loss=False, rescale=True,**data)
        # saveresult=result[0]
        # output = Image.fromarray(saveresult.astype(np.uint8)*255)
        # output.save('gram.png')
        # exit(0)
        gt=data_loader.dataset.get_gt_seg_map_by_idx(batch_indices[0])
        if result[0].shape[1]!=gt.shape[1] or result[0].shape[0]!=gt.shape[0]:
            gt=cv2.resize(gt.astype(float),(result[0].shape[1],result[0].shape[0]))
            gt[gt>=0.5]=1
            gt[gt<0.5]=0
        if 'DSO' in data['img_metas'][0].data[0][0]['filename']:
            gt=1-gt
        # 计算AUC
        # print(result[:,1,:,:][0].flatten())
        output = Image.fromarray((result[0]*255).astype(np.uint8))
        output.save('reuslt.png')
        # print(gt.shape)
        # print(result[0].shape)
        # cv2.imwrite('congt.png',gt*255)
        # exit(0)
        auc = roc_auc_score(gt.flatten(),result[0].flatten())
        rocauc.append(auc)
        F1_best, F1_th = computeLocalizationMetrics(result[0], gt)
        f1.append(F1_best)
        f1th.append(F1_th)
        
        iou=calculate_iou(gt,result[0])
        iouth.append(iou)
        
        ##sam
        # random_points=[]
        # labeled_array, num_features = label(result[0])
        # for i in range(1, num_features + 1):
        #     # 获取当前连通区域的坐标
        #     region_indices = np.argwhere(labeled_array == i)
            
        #     # 随机选择一个点的坐标
        #     random_point = region_indices[random.randint(0, len(region_indices) - 1)]
            
        #     # 将选中的点添加到结果列表中
        #     random_points.append(tuple(random_point))

        # IMAGE_PATH=data['img_metas'][0].data[0][0]['filename']
        # cv2.imwrite('/home/hexingyang/DDP/segmentation/vis/1/'+IMAGE_PATH.split('/')[-1],result[0]*255)
        # everything_results = smodel(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
        # prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)
        # annotations=prompt_process.point_prompt(points=[[random_point[1],random_point[0]] for random_point in random_points], pointlabel=[1 for _ in random_points])
        # if len(annotations)>0:
        #     result[0] = np.logical_or(annotations[0], result[0]).astype(float)
        # cv2.imwrite('/home/hexingyang/DDP/segmentation/vis/2/'+IMAGE_PATH.split('/')[-1],result[0]*255)

        # gt=gt_seg_maps[batch_indices[0]]
        # gt=cv2.resize(gt.astype(float),(result[0].shape[1],result[0].shape[0]))
        # if 'DSO' in data['img_metas'][0].data[0][0]['filename']:
        #     gt=1-gt
        # F1_best, F1_th = computeLocalizationMetrics(result[0], gt)
        # f1.append(F1_best)
        # f1th.append(F1_th)
        
        # if show or out_dir:
        #     img_tensor = data['img'][0]
        #     img_metas = data['img_metas'][0].data[0]
        #     imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        #     assert len(imgs) == len(img_metas)

        #     for img, img_meta in zip(imgs, img_metas):
        #         h, w, _ = img_meta['img_shape']
        #         img_show = img[:h, :w, :]

        #         ori_h, ori_w = img_meta['ori_shape'][:-1]
        #         img_show = mmcv.imresize(img_show, (ori_w, ori_h))

        #         if out_dir:
        #             out_file = osp.join(out_dir, img_meta['ori_filename'])
        #         else:
        #             out_file = None

        #         model.module.show_result(
        #             img_show,
        #             result,
        #             palette=dataset.PALETTE,
        #             show=show,
        #             out_file=out_file,
        #             opacity=opacity)

        # if efficient_test:
        #     result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        # if format_only:
        #     result = dataset.format_results(
        #         result, indices=batch_indices, **format_args)
        # if pre_eval:
        #     # TODO: adapt samples_per_gpu > 1.
        #     # only samples_per_gpu=1 valid now
        #     result = dataset.pre_eval(result, indices=batch_indices)
        #     results.extend(result)
        # else:
        #     results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    with open('/home/hexingyang/DDP/segmentation/test/result.txt','a+') as f:
        f.write("F1 - fixed: {},F1 - best: {},AUC:{},iou:{}".format(np.nanmean(f1th),np.nanmean(f1),np.nanmean(rocauc),np.nanmean(iouth)))
        f.write('\n')
    return results


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False,
                   pre_eval=False,
                   format_only=False,
                   format_args={},
                   savep=None):
    """Test model with multiple gpus by progressive mode.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. The same path is used for efficient
            test. Default: None.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.

    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx

    # batch_sampler based on DistributedSampler, the indices only point to data
    # samples of related machine.
    loader_indices = data_loader.batch_sampler

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    saveid=0
    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        saveresult=result[0]
        if saveid<10:
            saveresult = saveresult
            output = Image.fromarray((saveresult*255).astype(np.uint8))
            pp=savep.split('/')[-1]
            if not osp.exists("/home/hexingyang/DDP/segmentation/val/"+pp):
                os.mkdir("/home/hexingyang/DDP/segmentation/val/"+pp)
            png_filename="/home/hexingyang/DDP/segmentation/val/"+pp+"/"+data['img_metas'][0].data[0][0]['filename'].split('/')[-1].replace('jpg','png')
            output.save(png_filename)
        
            saveid+=1

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = dataset.format_results(
                result, indices=batch_indices, **format_args)
        pre_eval=False
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(result, indices=batch_indices)

        results.extend(result)

        if rank == 0:
            batch_size = len(result) * world_size
            for _ in range(batch_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results
