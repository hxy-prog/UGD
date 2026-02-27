import argparse
import json
import time
from pathlib import Path
import sys
import numpy as np
import cv2
import xml.etree.cElementTree as ET
import torch
import copy
import torch.backends.cudnn as cudnn
from numpy import random
sys.path.insert(0,'./yolov7')
sys.path.append('.')
from PyQt5.QtWidgets import QProgressBar,QWidget,QProgressDialog
from PyQt5.QtCore import *
from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from tracker.mc_bot_sort import BoTSORT,STrack
from tracker.tracking_utils.timer import Timer

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

def detect(opt,mainwindow):
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
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device,weights_only=False)['model']).to(device).eval()

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
    t0 = time.time()

    txtpath = str(save_dir) +"/classes.txt"
    file = open(txtpath, 'w')
    for x in names:
        file.write(x+'\n')
    file.close()
    j=0
    progressDialog = QProgressDialog('进度','取消', 0,dataset.nframes)
    progressDialog.resize(300,100)
    progressDialog.setWindowTitle('QProgressDialog')
    progressDialog.show()
    for path, img, im0s, vid_cap in dataset:
        j+=1
        #进度条
        progressDialog.setValue(j)
        QCoreApplication.processEvents()
        if progressDialog.wasCanceled():
            print('cancel\n')
            return
        #创建label文件
        if opt.format=='json':
            labelfilename = str(save_dir) + '/' + str(j) + '.json'
        elif opt.format=='txt':
            labelfilename = str(save_dir) + '/' + str(j) + '.txt'
        elif opt.format=='xml':
            labelfilename = str(save_dir) + '/' + str(j) + '.xml'
            # 创建XML根元素
            root = ET.Element("annotation")
            # 添加文件夹名称元素
            folder = ET.SubElement(root, "folder")
            folder.text = path.split("\\")[-2]
            # 添加文件名元素
            filename = ET.SubElement(root, "filename")
            filename.text = path.split("\\")[-1]
            # 添加文件路径元素
            pathn = ET.SubElement(root, "path")
            pathn.text = path
            # 添加数据来源元素
            source = ET.SubElement(root, "source")
            database = ET.SubElement(source, "database")
            database.text = "Unknown"
            # 添加图像大小元素
            size = ET.SubElement(root, "size")
            width = ET.SubElement(size, "width")
            width.text = str(im0s.shape[1])
            height = ET.SubElement(size, "height")
            height.text = str(im0s.shape[0])
            depth = ET.SubElement(size, "depth")
            depth.text = str(im0s.shape[2])
            # 添加分割元素
            segmented = ET.SubElement(root, "segmented")
            segmented.text = "0"
        labelfile = open(labelfilename, 'w')
        #idfile
        idlabelname = str(save_dir) + '/' + str(j) + '_id.txt'
        idfile = open(idlabelname, 'w')
        ###抽取样本帧，非样本帧预测
        if j%2!=1:
            id=[]
            im0=im0s
            tracko=copy.deepcopy(tracker.tracked_stracks)
            for t in tracko:
                t.mean[4:]=t.mean[4:]/2
            STrack.multi_predict(tracko)
            # STrack.multi_predict(tracker.tracked_stracks)
            annotion=[]
            for t in tracko:
            # for t in tracker.tracked_stracks:
                tid=t.track_id
                tlwh=t.tlwh
                tlbr=t.tlbr
                if tlwh[2] * tlwh[3] > opt.min_box_area and t.score > mainwindow.atuolabelthresh:
                    id.append(t.track_id)
                    xcenter = float((tlbr[0] + tlbr[2]) / 2)
                    ycenter = float((tlbr[1] + tlbr[3]) / 2)
                    if opt.format == 'json':
                        keyvalue = {"label": names[int(t.cls)],
                                    "coordinates": {"x": xcenter, "y": ycenter, "width": float(tlwh[2]),
                                                    "height": float(tlwh[3])}}
                        annotion.append(keyvalue)
                # if tlwh[2] * tlwh[3] > opt.min_box_area:
                    if opt.hide_labels_name:
                        label = f'{tid}, {int(t.cls)},{t.score:.2f}'
                    else:
                        label = f'{tid}, {names[int(t.cls)]},{t.score:.2f}'
                    plot_one_box(tlbr, im0, label=label, color=colors[int(tid) % len(colors)], line_thickness=2)
            results = [{"image": str(j) + ".jpg", "verified": False, "annotations": annotion}]
            json.dump(results, labelfile)
            savepath = str(save_dir) + '/' + str(j) + '.jpg'
            cv2.imwrite(savepath, im0)
            labelfile.close()
            idfile.write(str(id))
            idfile.close()
            vid_writer.write(im0)
            continue
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
            online_targets = tracker.update(detections, im0)

            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_cls = []
            annotion=[]
            id=[]
            for t in online_targets:
                tlwh = t.tlwh#左上x，y，w，h
                tlbr = t.tlbr#左上，右下坐标
                tid = t.track_id
                tcls = t.cls
                if tlwh[2] * tlwh[3] > opt.min_box_area and t.score>mainwindow.atuolabelthresh:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    online_cls.append(t.cls)
                    # save results
                    id.append(tid)
                    xcenter=float((tlbr[0]+tlbr[2])/2)
                    ycenter=float((tlbr[1]+tlbr[3])/2)
                    if opt.format == 'json':
                        keyvalue = {"label": names[int(t.cls)],
                                    "coordinates": {"x": xcenter, "y": ycenter, "width": float(tlwh[2]),
                                                    "height": float(tlwh[3])}}
                        annotion.append(keyvalue)
                    elif opt.format == 'txt':
                        keyvalue=str(int(t.cls))+' '+str(float(xcenter)/im0.shape[1])+' '+str(float(ycenter)/im0.shape[0])+' '+str(float(tlwh[2]/im0.shape[1]))+' '+str(float(tlwh[3]/im0.shape[0]))+'\n'
                        labelfile.write(keyvalue)
                    elif opt.format == 'xml':
                        object = ET.SubElement(root, "object")
                        name = ET.SubElement(object, "name")
                        name.text = names[int(t.cls)]
                        pose = ET.SubElement(object, "pose")
                        pose.text = "Unspecified"
                        truncated = ET.SubElement(object, "truncated")
                        truncated.text = "0"
                        difficult = ET.SubElement(object, "difficult")
                        difficult.text = "0"
                        bndbox = ET.SubElement(object, "bndbox")
                        xmin = ET.SubElement(bndbox, "xmin")
                        xmin.text = str(tlbr[0])
                        ymin = ET.SubElement(bndbox, "ymin")
                        ymin.text = str(tlbr[1])
                        xmax = ET.SubElement(bndbox, "xmax")
                        xmax.text = str(tlbr[2])
                        ymax = ET.SubElement(bndbox, "ymax")
                        ymax.text = str(tlbr[3])
                    if save_img or view_img:  # Add bbox to image
                        if opt.hide_labels_name:
                            label = f'{tid}, {int(tcls)},{t.score:.2f}'
                        else:
                            label = f'{tid}, {names[int(tcls)]},{t.score:.2f}'
                        plot_one_box(tlbr, im0, label=label, color=colors[int(tid) % len(colors)], line_thickness=2)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')
            idfile.write(str(id))
            idfile.close()
            if opt.format == 'json':
                results = [{"image": str(j) + ".jpg", "verified": False, "annotations": annotion}]
                json.dump(results, labelfile)
            elif opt.format == 'txt':
                pass
            elif opt.format == 'xml':
                tree=ET.ElementTree(root)
                tree.write(labelfilename)
            labelfile.close()
            savepath=str(save_dir)+'/'+str(j)+'.jpg'
            cv2.imwrite(savepath,im0)
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
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    file.close()
    print(f'Done. ({time.time() - t0:.3f}s)')
    mainwindow.last_open_dir=str(save_dir)
    mainwindow.import_dir_images(str(save_dir))
    # mainwindow.default_save_dir = str(save_dir)
class Parse:
    def __init__(self):
        self.weights='yolov7.pt'
        self.source='inference/images'
        self.format='json'
        self.img_size=1920
        self.conf_thres=0.09
        self.iou_thres=0.7
        self.device=''
        self.view_img=False
        self.save_txt=False
        self.save_conf=False
        self.nosave=False
        self.classes=None
        self.agnostic_nms=True
        self.augment=False
        self.update=False
        self.project='runs/detect'
        self.name='exp'
        self.exist_ok=False
        self.trace=False
        self.hide_labels_name=False
        self.track_high_thresh=0.3
        self.track_low_thresh=0.05
        self.new_track_thresh=0.4
        self.track_buffer=30
        self.match_thresh=0.7
        self.aspect_ratio_thresh=1.6
        self.min_box_area=10
        self.mot20=True
        self.cmc_method="sparseOptFlow"
        #是否使用iou-reid融合,project2发现加上with-reid对idswitch影响不大，false可以减少运行时间
        self.with_reid=False
        self.fast_reid_config=r"BoT-SORT-main/fast_reid/configs/MOT17/sbs_S50.yml"
        self.fast_reid_weights=r"BoT-SORT-main/pretrained/mot17_sbs_S50.pth"
        self.proximity_thresh=0.5
        self.appearance_thresh=0.25
        self.jde = False
        self.ablation = False
def y7main(str,mainwindow):
    opt=Parse()
    (weight,sourcefile,classfilter,savepath,formatt)=str
    opt.format=formatt
    opt.source=sourcefile
    if classfilter=="all":
        opt.classes=None
    else:
        opt.classes=classfilter
    opt.weights=weight
    # check_requirements(exclude=('pycocotools', 'thop'))
    if savepath!='':
        opt.project=savepath+'/runs/detect'
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
           detect(opt,mainwindow)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
#     parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
#     parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.09, help='object confidence threshold')
#
#     parser.add_argument('--iou-thres', type=float, default=0.7, help='IOU threshold for NMS')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--view-img', action='store_true', help='display results')
#     parser.add_argument('--save-txt', default=True,action='store_true', help='save results to *.txt')
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
#     parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--update', action='store_true', help='update all models')
#     parser.add_argument('--project', default='runs/detect', help='save results to project/name')
#     parser.add_argument('--name', default='exp', help='save results to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     parser.add_argument('--trace', action='store_true', help='trace model')
#     parser.add_argument('--hide-labels-name', default=False, action='store_true', help='hide labels')
#
#     # tracking args
#     parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
#     parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
#     parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track thresh")
#     parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
#     parser.add_argument("--match_thresh", type=float, default=0.7, help="matching threshold for tracking")
#     parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
#                         help="threshold for filtering out boxes of which aspect ratio are above the given value.")
#     parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
#     parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
#                         help="fuse score and iou for association")
#
#     # CMC
#     parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")
#
#     # ReID
#     parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
#     parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
#                         type=str, help="reid config file path")
#     parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth",
#                         type=str, help="reid config file path")
#     parser.add_argument('--proximity_thresh', type=float, default=0.5,
#                         help='threshold for rejecting low overlap reid matches')#拒绝iou小于给定值的匹配
#     parser.add_argument('--appearance_thresh', type=float, default=0.25,
#                         help='threshold for rejecting low appearance similarity reid matches')#拒绝外观小于给定值的匹配
#
#     opt = parser.parse_args()
#
#     opt.jde = False
#     opt.ablation = False
#     print(opt)
#     # check_requirements(exclude=('pycocotools', 'thop'))
#
    # with torch.no_grad():
    #     if opt.update:  # update all models (to fix SourceChangeWarning)
    #         for opt.weights in ['yolov7.pt']:
    #             detect()
    #             strip_optimizer(opt.weights)
    #     else:
    #         detect()
