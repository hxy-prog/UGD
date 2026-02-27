import argparse
import json
import os
import time
from pathlib import Path
import sys
import numpy as np
import cv2
import copy
import xml.etree.cElementTree as ET
import torch
import torch.backends.cudnn as cudnn
from numpy import random
sys.path.insert(0,'./yolov7')
sys.path.append('.')
from PyQt5.QtWidgets import QProgressBar,QWidget,QProgressDialog
from PyQt5.QtCore import *
import math
from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from tracker.mc_bot_sort import BoTSORT,STrack
from tracker import matching
from tracker.tracking_utils.timer import Timer
from scipy import interpolate


def euclidean_distance(p1, p2):
    '''
    计算两个点的欧式距离
    '''
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


class BBox:
    def __init__(self, x, y, r, b):
        '''
        定义框，左上角及右下角坐标
        '''
        self.x, self.y, self.r, self.b = x, y, r, b

    def __xor__(self, other):
        '''
        计算box和other的IoU
        '''
        cross = self & other
        union = self | other
        return cross / (union + 1e-6)

    def __or__(self, other):
        '''
        计算box和other的并集
        '''
        cross = self & other
        union = self.area + other.area - cross
        return union

    def __and__(self, other):
        '''
        计算box和other的交集
        '''
        xmax = min(self.r, other.r)
        ymax = min(self.b, other.b)
        xmin = max(self.x, other.x)
        ymin = max(self.y, other.y)
        cross_box = BBox(xmin, ymin, xmax, ymax)
        if cross_box.width <= 0 or cross_box.height <= 0:
            return 0
        return cross_box.area

    def boundof(self, other):
        '''
        计算box和other的边缘外包框，使得2个box都在框内的最小矩形
        '''
        xmin = min(self.x, other.x)
        ymin = min(self.y, other.y)
        xmax = max(self.r, other.r)
        ymax = max(self.b, other.b)
        return BBox(xmin, ymin, xmax, ymax)

    def center_distance(self, other):
        '''
        计算两个box的中心点距离
        '''
        return euclidean_distance(self.center, other.center)

    def bound_diagonal_distance(self, other):
        '''
        计算两个box的bound的对角线距离
        '''
        bound = self.boundof(other)
        return euclidean_distance((bound.x, bound.y), (bound.r, bound.b))

    @property
    def center(self):
        return (self.x + self.r) / 2, (self.y + self.b) / 2

    @property
    def area(self):
        return self.width * self.height

    @property
    def width(self):
        return self.r - self.x  # + 1

    @property
    def height(self):
        return self.b - self.y  # + 1
def IoU(a, b):
    return a ^ b
def GIoU(a, b):
    bound_area = a.boundof(b).area
    union_area = a | b
    return IoU(a, b) - (bound_area - union_area) / bound_area
def DIoU(a, b):
    d = a.center_distance(b)
    c = a.bound_diagonal_distance(b)
    return IoU(a, b) - (d ** 2) / (c ** 2)
def CIoU(a, b):
    v = 4 / (math.pi ** 2) * (math.atan(a.width / a.height) - math.atan(b.width / b.height)) ** 2
    iou = IoU(a, b)
    alpha = v / (1 - iou + v)
    return DIoU(a, b) - alpha * v

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    print('save results to {}'.format(filename))

def detect(averageacc):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # print('aaaaaaaaaaaaaa\n')
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    # Create tracker
    tracker = BoTSORT(opt, frame_rate=30.0)
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    # t0 = time.time()
    txtpath = str(save_dir) +"/det.txt"
    file = open(txtpath, 'w')
    j=0
    save_path = str(save_dir )+'/v.mp4'  # img.jpg
    fps=30
    sumacc=0
    num=0
    truthpath=os.path.dirname(opt.source)+'/det/det.txt'
    truthfile=open(truthpath,'r')
    truthlist = truthfile.readline().split(',')
    # Create Kalman filter model matrices.
    for path, img, im0s, vid_cap in dataset:
        j+=1
        btlbr=[]
        atlbr=[]
        while truthlist[0]==str(j):
            btlbr.append(np.array([float(truthlist[2]),float(truthlist[3]),float(truthlist[2])+float(truthlist[4]),float(truthlist[3])+float(truthlist[5])]))
            truthlist = truthfile.readline().split(',')
        if j==1:
            w=im0s.shape[1]
            h=im0s.shape[0]
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        print(j)
        # idfile
        idlabelname = str(save_dir) + '/' + str(j) + '_id.txt'
        idfile = open(idlabelname, 'w')
        ###
        if j%3==1000:
            id=[]
            im0=im0s
            try:
                tracko=copy.deepcopy(tracker.tracked_stracks)
                for t in tracko:
                    t.mean[4:]=t.mean[4:]/2
                STrack.multi_predict(tracko)
                # STrack.multi_predict(tracker.tracked_stracks)
                # for t in tracker.tracked_stracks:
                for t in tracko:
                    # print('c')
                    # print(t.mean)
                    tid=t.track_id
                    tlwh=t.tlwh
                    tlbr=t.tlbr
                    if tlwh[2] * tlwh[3] > opt.min_box_area and t.score>0.9:
                        atlbr.append(np.array([tlbr[0], tlbr[1], tlbr[2], tlbr[3]]))
                        file.write(str(j)+','+str(tlwh[0])+','+str(tlwh[1])+','+str(tlwh[2])+','+str(tlwh[3])+','+str(t.score)+'\n')
                        id.append(t.track_id)
                        plot_one_box(tlbr, im0, color=colors[int(tid) % len(colors)], line_thickness=2)
                vid_writer.write(im0)
                idfile.write(str(id))
            except Exception as e:
                print(e)
                exit(0)
        else:
            #奇数
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]
            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()
            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            for i, det in enumerate(pred):  # detections per image


                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                # Run tracker
                detections = []
                if len(det):
                    boxes = scale_coords(img.shape[2:], det[:, :4], im0.shape)
                    boxes = boxes.cpu().numpy()
                    detections = det.cpu().numpy()
                    detections[:, :4] = boxes
                # print(detections)
                # exit(0)
                id=[]
                if j>=3:
                    for t in tracker.tracked_stracks:
                        t.lastmean=copy.deepcopy(t.mean)
                online_targets = tracker.update(detections, im0)

                online_tlwhs = []
                online_ids = []
                online_scores = []
                online_cls = []
                for t in online_targets:
                    tlwh = t.tlwh#左上x，y，w，h
                    tlbr = t.tlbr#左上，右下坐标
                    tid = t.track_id
                    tcls = t.cls
                    if tlwh[2] * tlwh[3] > opt.min_box_area and t.score>0.9:
                        atlbr.append(np.array([tlbr[0], tlbr[1], tlbr[2], tlbr[3]]))
                        id.append(tid)
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        online_cls.append(t.cls)
                        file.write(str(j) + ',' + str(tlwh[0]) + ',' + str(tlwh[1]) + ',' + str(tlwh[2]) + ',' + str(
                            tlwh[3]) + ',' + str(t.score) + '\n')
                        plot_one_box(tlbr, im0, color=colors[int(tid) % len(colors)], line_thickness=2)
                vid_writer.write(im0)
                        # if save_img or view_img:  # Add bbox to image
                        #     if opt.hide_labels_name:
                        #         label = f'{tid}, {int(tcls)},{t.score:.2f}'
                        #     else:
                        #         label = f'{tid}, {names[int(tcls)]},{t.score:.2f}'
                        #     plot_one_box(tlbr, im0, label=label, color=colors[int(tid) % len(colors)], line_thickness=2)
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                idfile.write(str(id))
                idfile.close()
                # Print time (inference + NMS)
                # print(f'{s}Done. ({t2 - t1:.3f}s)')
                # Stream results
                if view_img:
                    cv2.imshow('BoT-SORT', im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)
        # if save_txt:
        #     file.write(str(results) + '\n')
        iou_list=np.zeros((len(atlbr),len(btlbr)))
        for a in range(len(atlbr)):
            for b in range(len(btlbr)):
                x=atlbr[a]
                y=btlbr[b]
                iou_list[a][b]=CIoU(BBox(x[0],x[1],x[2],x[3]),BBox(y[0],y[1],y[2],y[3]))
        # iou_list=matching.ious(atlbr,btlbr)
        if len(atlbr)>0 and len(btlbr)>0:
            for i in range(len(iou_list)):
                num+=1
                ma=max(iou_list[i])
                sumacc+=ma
    print(sumacc/num)
    averageacc.append(sumacc/num)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    file.close()
    # print(f'Done. ({time.time() - t0:.3f}s)')
    # mainwindow.default_save_dir = str(save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--format', nargs='+', type=str, default='json', help='model.pt path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.09, help='object confidence threshold')

    parser.add_argument('--iou-thres', type=float, default=0.7, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', default=False,action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', default=True,action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    parser.add_argument('--hide-labels-name', default=False, action='store_true', help='hide labels')

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.7, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
                        help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')#拒绝iou小于给定值的匹配
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')#拒绝外观小于给定值的匹配

    opt = parser.parse_args()

    opt.jde = False
    opt.ablation = False
    opt.classes=0
    # check_requirements(exclude=('pycocotools', 'thop'))
    truthdir=r'C:\Users\26387\Desktop\dataset\MOT\MOT17\test'
    averageacc=[]
    t0 = time.time()
    with torch.no_grad():
        for dir in os.listdir(truthdir):
            strr=dir.split('-')
            #总共6个
            if strr[2]=='FRCNN' :
                opt.source=truthdir+'/'+dir+'/'+'img1'
                print(opt)
                detect(averageacc)
                print('\n\ndone\n\n')
    # with torch.no_grad():
    #     opt.source=r'C:\Users\26387\Desktop\MOT20\test\MOT20-06\img1'
    #     print(opt)
    #     detect(averageacc)
    print(f'Done. ({time.time() - t0:.3f}s)')
    print(sum(averageacc)/len(averageacc))
    # with torch.no_grad():
    #     if opt.update:  # update all models (to fix SourceChangeWarning)
    #         for opt.weights in ['yolov7.pt']:
    #             detect()
    #             strip_optimizer(opt.weights)
    #     else:
    #         detect()
