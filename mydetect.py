'''
Autor: dingxiaolin
Date: 2022-01-27 18:09:26
LastEditors: dingxiaolin
LastEditTime: 2022-01-27 18:09:27
https://gitee.com/mrjinfa/yolov5/blob/master/mydetect.py
'''
import time

import cv2
import torch

from models.experimental import attempt_load
from mydatasets import load_images
from utils.general import check_img_size, check_requirements, non_max_suppression, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging
from utils.torch_utils import select_device


def load_model(
        weights='yolov5s.pt',
        device='cpu'
):
    # Initialize
    set_logging()
    device = select_device(device)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    return model, names


def detect(
    img,
    model,
    names,
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    half=False,  # use FP16 half-precision inference
):

    img, im0s = LoadImages.getimg(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img)[0]
    ret_data = list()

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        im0 = im0s.copy()

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                xywh_dict = {
                    "x": int(xywh[0]),
                    "y": int(xywh[1]),
                    "w": int(xywh[2]),
                    "h": int(xywh[3]),
                    "conf": int(conf),
                    "cls": names[int(cls)]
                }
                ret_data.append(xywh_dict)
    return ret_data
