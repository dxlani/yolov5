import argparse # python的命令行解析的标准模块  可以让我们直接在命令行中就可以向程序中传入参数并让程序运行
import time
from pathlib import Path # Path将str转换为Path对象 使字符串路径易于操作的模块

import cv2 # opencv模块
import torch  # pytorch模块
import torch.backends.cudnn as cudnn # cuda模块
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(save_img=False):
    # ===================================== 1、初始化一些配置 =====================================
    # 获取输出文件夹，输入源，权重，参数等参数
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    # 是否保存预测后的图片 默认nosave=False 所以只要传入的文件地址不是以.txt结尾 就都是要保存预测后的图片的
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    # 是否是使用webcam 网页数据 一般是Fasle  因为我们一般是使用图片流LoadImages(可以处理图片/视频流文件)
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # 默认save_txt=False 所以这里一般都是新建一个 save_dir(runs/detect/expn)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize 初始化日志信息
    set_logging()
    # 获取当前主机可用的设备
    device = select_device(opt.device)
    # 如果设备为gpu，使用Float16
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # ===================================== 2、载入模型和模型参数并调整模型 =====================================
    # 2.1、加载Float32模型，确保用户设定的输入图片分辨率能整除32(如不能则调整为能整除并返回)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    # 2.2、载入一些模型参数
    # stride: 模型最大的下采样率 [8, 16, 32] 所有stride一般为32
    stride = int(model.stride.max())  # model stride
    # 确保输入图片的尺寸imgsz能整除stride=32 如果不能则调整为能被整除并返回
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    # 设置Float16 是否将模型从float32 -> float16  加速推理
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    # 设置第二次分类，默认不使用
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # ===================================== 3、加载推理数据 =====================================
    # Set Dataloader
    # 通过不同的输入源来设置不同的数据加载方式
    vid_path, vid_writer = None, None
    if webcam:
        # 一般不会使用webcam模式从网页中获取数据
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        # 一般是直接从source文件目录下直接读取图片或者视频数据
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    # 获取类别名字
    names = model.module.names if hasattr(model, 'module') else model.names
    # 设置画框的颜色
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # ===================================== 4、推理前测试 =====================================
    # 这里先设置一个全零的Tensor进行一次前向推理 判断程序是否正常
    # Run inference
    if device.type != 'cpu':
        # 进行一次前向推理,测试程序是否正常
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    # ===================================== 5、正式推理 =====================================
    t0 = time.time()
    """
    path 图片/视频路径
    img 进行resize+pad之后的图片
    img0 原size图片
    cap 当读取图片时为None，读取视频时为视频源
    """
    for path, img, im0s, vid_cap in dataset:
        # 5.1、处理每一张图片的格式
        img = torch.from_numpy(img).to(device)
        # 半精度训练 图片也设置为Float16
        img = img.half() if half else img.float()  # uint8 to fp16/32
        # 归一化 0 - 255 to 0.0 - 1.0
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # 如果图片是3维(RGB) 就在前面添加一个维度1当中batch_size=1
        # 因为输入网络的图片需要是4为的 [batch_size, channel, w, h]
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        # 5.2、对每张图片/视频进行前向推理
        t1 = time_synchronized()
        """
        前向传播 返回pred的shape是(1, num_boxes, 5+num_class)
        h,w为传入网络图片的长和宽，注意dataset在检测时使用了矩形推理，所以这里h不一定等于w
        num_boxes = h/32 * w/32 + h/16 * w/16 + h/8 * w/8
        pred[..., 0:4]为预测框坐标
        预测框坐标为xywh(中心点+宽长)格式
        pred[..., 4]为objectness置信度
        pred[..., 5:-1]为分类结果
        """
        # pred shape=[1, num_boxes, xywh+obj_conf+classes] = [1, 18900, 25]
        pred = model(img, augment=opt.augment)[0]
    
        # Apply NMS
        """
        pred:前向传播的输出
        conf_thres:置信度阈值
        iou_thres:iou阈值
        classes:是否只保留特定的类别
        agnostic:进行nms是否也去除不同类别之间的框 默认False
        经过nms之后，预测框格式：xywh-->xyxy(左上角右下角)
        pred是一个列表list[torch.tensor]，长度为batch_size
        每一个torch.tensor的shape为(num_boxes, 6),内容为box+conf+cls
        """
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        # 如果需要二次分类 就进行二次分类  一般是不需要的
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        # 对每一张图片作处理 将pred(相对img_size 640)映射回原图img0 size
        for i, det in enumerate(pred):  # detections per image
            # 如果输入源是webcam（网页），则batch_size不为1，取出dataset中的一张图片
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                # 但是大部分我们一般都是从LoadImages流读取本都文件中的照片或者视频 所以batch_size=1
                # p: 当前图片/视频的绝对路径 如 F:\yolo_v5\yolov5-U\data\images\bus.jpg
                # s: 输出信息 初始为 ''
                # im0: 原始图片 letterbox + pad 之前的图片
                # frame: 初始为0  可能是当前图片属于视频中的第几帧？
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # 设置保存图片/视频的路径
            save_path = str(save_dir / p.name)  # img.jpg
            # 设置保存框坐标txt文件的路径
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            # 设置打印信息(图片长宽)
            s += '%gx%g ' % img.shape[2:]  # print string
            # normalization gain gn = [w, h, w, h]  用于后面的归一化
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                # 将预测信息（相对img_size 640）映射回原图 img0 size
                # 调整预测框的坐标：基于resize+pad的图片的坐标-->基于原size图片的坐标
                # 此时坐标格式为xyxy
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # 打印检测到的类别数量
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # 保存预测信息: txt、img0上画框、crop_img
                for *xyxy, conf, cls in reversed(det):
                    # 将每个图片的预测信息分别存入save_dir/labels下的xxx.txt中 每行: class_id+score+xywh
                    if save_txt:  # Write to file
                        # 将xyxy(左上角+右下角)格式转为xywh(中心点+宽长)格式，并除上w，h做归一化，转化为列表再保存
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    # 在原图上画框
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            # 打印前向传播+nms时间
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            # 是否需要显示我们预测后的结果  img0(此时已将pred结果可视化到了img0中)
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # 是否需要保存图片或视频（检测后的图片/视频 里面已经被我们画好了框的） img0
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

    # ===================================== 6、推理结束, 保存结果, 打印信息 =====================================
    # 保存预测的label信息 xywh等   save_txt
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    # 打印总时间
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images/', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--source', type=str, default='rtsp.txt', help='source') 
    # parser.add_argument('--source', type=str, default='rtsp://admin:jhx12345@192.168.1.249:554/Streaming/Channels/1', help='source')  
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # 置信度阈值
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    # 做nms的iou阈值
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 是否展示预测之后的图片/视频
    parser.add_argument('--view-img', action='store_true', help='display results',default=True)
    # 是否将预测的框坐标以txt文件格式保存 默认True 会在runs/detect/expn/labels下生成每张图片预测的txt文件
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # 是否保存预测每个目标的置信度到预测tx文件中 默认True
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # 是否不要保存预测后的图片  默认False 就是默认要保存预测后的图片
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # 设置只保留某一部分类别
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    # 进行nms是否也去除不同类别之间的框
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # 推理的时候进行多尺度，翻转等操作(TTA)推理
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # 如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息
    parser.add_argument('--update', action='store_true', help='update all models')
    # 是否将optimizer从ckpt中删除  更新模型  默认False
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    # 当前测试结果放在run/detect下的文件名  默认是exp
    parser.add_argument('--name', default='exp', help='save results to project/name')
    # exist-ok: 是否存在当前文件 默认False 一般是 no exist-ok 连用  所以一般都要重新创建文件夹
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                # 去除pt文件中的优化器等信息
                strip_optimizer(opt.weights)
        else:
            detect()
