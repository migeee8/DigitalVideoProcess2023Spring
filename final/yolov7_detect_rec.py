import argparse
import time
import os
import copy
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from plate_recognition.plate_rec import get_plate_result, init_model, cv_imread
from utils.datasets import letterbox
from utils.cv_puttext import cv2ImgAddText


def crop_plate_from_image(img, coordinate):  # 裁剪区域
    x_values = [int(point[0]) for point in coordinate]
    y_values = [int(point[1]) for point in coordinate]

    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)

    plate_img = img[min_y:max_y, min_x:max_x]
    return plate_img


def four_point_transform(image, pts):  # 透视变换
    rect = pts.astype("float32")

    # 进行透视变换的四个点的坐标
    (tl, tr, br, bl) = rect

    # 计算四条边的长度并比较，确定要变换的目标尺寸
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")  # 定义透视变换后图像的目标区域
    M = cv2.getPerspectiveTransform(rect, dst)  # 变换矩阵M
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))  # 应用变换矩阵
    return warped


def get_plate_rec_landmark(img, xyxy, conf, landmarks, class_num, device, plate_rec_model):
    result_dict = {}

    # 从 xyxy 中提取边界框坐标
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])

    # 根据提取的关键点坐标进行车牌图像的透视变换
    landmarks_np = np.zeros((4, 2))
    rect = [x1, y1, x2, y2]
    for i in range(4):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        landmarks_np[i] = np.array([point_x, point_y])
    roi_img = four_point_transform(img, landmarks_np)  # 透视变换得到车牌小图
    # cv2.imshow("img",roi_img)
    # 对车牌小图进行识别
    plate_number, rec_prob, plate_color, color_conf = get_plate_result(roi_img, device, plate_rec_model)

    result_dict['rect'] = rect  # 车牌区域对应矩形坐标
    result_dict['landmarks'] = landmarks_np.tolist()  # 四个点的坐标
    result_dict['plate_no'] = plate_number     # 车牌号
    result_dict['rec_conf'] = rec_prob  # 每个字符的概率
    result_dict['plate_color'] = plate_color    # 车牌颜色
    result_dict['color_conf'] = color_conf   # 车牌颜色置信度
    result_dict['roi_height'] = roi_img.shape[0]    # 车牌区域高度
    result_dict['score'] = conf     # 置信度
    result_dict['label'] = int(class_num)
    return result_dict


def detect_Recognition_plate(model, orgimg, device, plate_rec_model, img_size):
    conf_thres = 0.3
    iou_thres = 0.5
    dict_list = []

    # 对原始图像进行预处理，调整大小(letterbox)和将颜色通道从BGR转换为RGB格式。
    im0 = copy.deepcopy(orgimg)
    imgsz = (img_size, img_size)
    img = letterbox(im0, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x640X640
    img = torch.from_numpy(img).to(device)
    img = img.float()  # 将数据类型从uint8转换为浮点型
    img /= 255.0  # 将像素值归一化为0.0 - 1.0的范围
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # 增加一个维度，将图像准备为模型推断的批处理输入

    # 预测 得到边界框、类别和置信度得分
    pred = model(img)[0]
    # 应用非最大抑制来过滤重叠和低置信度的边界框
    pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, kpt_label=4, agnostic=True)

    # 对于每个检测结果(det)，提取并处理边界框坐标和其他细节
    for i, det in enumerate(pred):
        if len(det):  # 如果det中不为空
            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
            scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=4, step=3)
            # 对每个检测到的目标
            for j in range(det.size()[0]):
                # 获取目标信息
                xyxy = det[j, :4].view(-1).tolist()
                # 计算车牌区域大小和比例
                plate_area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                ratio = (xyxy[2] - xyxy[0]) / (xyxy[3] - xyxy[1])
                # 设置车牌大小以及长宽比的阈值
                min_plate_area = 2000
                max_plate_area = 15000
                min_ratio = 2

                full_img_area = im0.shape[0] * im0.shape[1]
                plate_area_ratio = plate_area / full_img_area * 10
                if min_plate_area < plate_area < max_plate_area and ratio > min_ratio:  # 如果车牌区域大小和比例合适 继续识别
                    print(plate_area_ratio)
                    conf = det[j, 4].cpu().numpy()  # 目标的置信度
                    landmarks = det[j, 6:].view(-1).tolist()  # 提取关键点坐标并转换为列表格式
                    landmarks = [landmarks[0], landmarks[1], landmarks[3], landmarks[4], landmarks[6], landmarks[7],
                                 landmarks[9], landmarks[10]]
                    class_num = det[j, 5].cpu().numpy()  # 目标的类别编号
                    # 目标（车牌）识别
                    result_dict = get_plate_rec_landmark(orgimg, xyxy, conf, landmarks, class_num, device,
                                                         plate_rec_model)
                    dict_list.append(result_dict)

    return dict_list


def draw_result(orgimg, dict_list):
    result_str = ""
    # 对于每个检测的目标
    for result in dict_list:
        # 定义绘制的矩形区域
        rect_area = result['rect']

        x, y, w, h = rect_area[0], rect_area[1], rect_area[2] - rect_area[0], rect_area[3] - rect_area[1]
        padding_w = 0.05 * w
        padding_h = 0.11 * h  # 矩形框的padding设置
        rect_area[0] = max(0, int(x - padding_w))  # 边界比较
        rect_area[1] = max(0, int(y - padding_h))
        rect_area[2] = min(orgimg.shape[1], int(rect_area[2] + padding_w))
        rect_area[3] = min(orgimg.shape[0], int(rect_area[3] + padding_h))
        rect_area = [int(x) for x in rect_area]  # 更新矩形区域

        # 绘制边界框
        cv2.rectangle(orgimg, (rect_area[0], rect_area[1]), (rect_area[2], rect_area[3]), (0, 0, 255), 2)

        # 设置显示的文本
        result_p = result['plate_no'] + " " + result['plate_color']
        result_str += result_p + " "

        labelSize = cv2.getTextSize(result_p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        if rect_area[0] + labelSize[0][0] > orgimg.shape[1]:
            rect_area[0] = int(orgimg.shape[1] - labelSize[0][0])  # 防止显示的文字越界

        # 绘制矩形放置识别结果的文本信息
        orgimg = cv2.rectangle(orgimg, (rect_area[0], int(rect_area[1] - round(1.6 * labelSize[0][1]))),
                               (int(rect_area[0] + round(1.2 * labelSize[0][0])), rect_area[1] + labelSize[1]),
                               (255, 255, 255), cv2.FILLED)

        # 4个关键点的绘制
        landmarks = result['landmarks']
        marksColors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        if len(result) > 1:
            for i in range(4):
                cv2.circle(orgimg, (int(landmarks[i][0]), int(landmarks[i][1])), 5, marksColors[i], -1)
            orgimg = cv2ImgAddText(orgimg, result_p, rect_area[0], int(rect_area[1] - round(1.6 * labelSize[0][1])),
                                   (0, 0, 0), 21)

    print(result_str)
    return orgimg


if __name__ == '__main__':
    # 加载检测模型
    detect_model = 'weights/yolov7-lite-s.pt'
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model = attempt_load(detect_model, map_location=device)

    # 加载识别模型
    rec_model = 'weights/plate_rec_color.pth'
    plate_rec_model = init_model(device, rec_model)

    output = 'result'
    source = 'imgs/test.png'
    img_size = 640

    if not os.path.exists(output):
        os.mkdir(output)

    time_b = time.time()

    img = cv_imread(source)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    dict_list = detect_Recognition_plate(model, img, device, plate_rec_model, img_size)
    for result in dict_list:
        print('\n')
        print(result['plate_no'])
    ori_img = draw_result(img, dict_list)
    save_img_path = os.path.join(output, "result.png")
    cv2.imwrite(save_img_path, ori_img)
    print(f"elasted time is {time.time() - time_b} s")
