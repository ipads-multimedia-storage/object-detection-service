import sys
import os
import cv2
import math
import numpy as np

# filter used to detect color in LAB color space
# XXX: but it seems too loose, it can be tighter
color_range = {
    'red': [(0, 151, 100), (255, 255, 255)], 
    'green': [(0, 0, 0), (255, 115, 255)], 
    'blue': [(0, 0, 0), (255, 255, 110)], 
    'black': [(0, 0, 0), (56, 255, 255)], 
    'white': [(193, 0, 0), (255, 250, 255)], 
}

color_to_bgr = {
    'red':   (0, 0, 255),
    'blue':  (255, 0, 0),
    'green': (0, 255, 0),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
}

# size = (640, 480)
size = (1080, 1440)
target_color = ('red', 'green', 'blue')

#找出面积最大的轮廓
#参数为要比较的轮廓的列表
def get_max_contour_and_area(contours) :
    max_contour = None
    max_contour_area = 0

    for c in contours : #历遍所有轮廓
        tmp = math.fabs(cv2.contourArea(c))  #计算轮廓面积
        if tmp > max_contour_area:
            max_contour_area = tmp
            if tmp > 300:  #只有在面积大于300时，最大面积的轮廓才是有效的，以过滤干扰
                max_contour = c

    return max_contour, max_contour_area

def getROI(box):
    x_min = min(box[0, 0], box[1, 0], box[2, 0], box[3, 0])
    x_max = max(box[0, 0], box[1, 0], box[2, 0], box[3, 0])
    y_min = min(box[0, 1], box[1, 1], box[2, 1], box[3, 1])
    y_max = max(box[0, 1], box[1, 1], box[2, 1], box[3, 1])

    return (x_min, x_max, y_min, y_max)
    
def detect(image):
    frame_resize = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
    frame_gb = cv2.GaussianBlur(frame_resize, (11, 11), 11)
    # it seems that this is not necessary
    #如果检测到某个区域有识别到的物体，则一直检测该区域直到没有为止
    # if get_roi and not start_pick_up:
    #     get_roi = False
    #     frame_gb = getMaskROI(frame_gb, roi, size)      
    frame_lab = cv2.cvtColor(frame_gb, cv2.COLOR_BGR2LAB)  # 将图像转换到LAB空间

    # the max contour
    max_contour = None
    # the color of the max contour
    max_contour_color = None
    # the area of the max contour
    max_contour_area = 0
    
    for color in target_color:
        frame_mask = cv2.inRange(frame_lab, color_range[color][0], color_range[color][1])  #对原图像和掩模进行位运算
        opened = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, np.ones((6,6),np.uint8))  #开运算
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((6,6),np.uint8)) #闭运算
        contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  #找出轮廓
        max_contour_tmp, max_contour_area_tmp = get_max_contour_and_area(contours)  #找出最大轮廓
        if max_contour_tmp is not None:
            if max_contour_area_tmp > max_contour_area:
                max_contour = max_contour_tmp
                max_contour_color = color
                max_contour_area = max_contour_area_tmp

    if max_contour_area > 2500:  # 有找到最大面积
        # rect: [(center_x, center_y), (width, height), angle]
        rect = cv2.minAreaRect(max_contour)
        box = np.int0(cv2.boxPoints(rect))
        
        roi = getROI(box) #获取roi区域
        # get_roi = True
        img_centerx, img_centery = rect[0] # 获取木块中心坐标
        # img_centerx, img_centery = getCenter(rect, roi, size, square_length)  # 获取木块中心坐标
        # print("rect:      ({})".format(rect[0]))
        # print("getCenter: ({}, {})".format(img_centerx, img_centery))
            
        # world_x, world_y = convertCoordinate(img_centerx, img_centery, size) #转换为现实世界坐标
        
        cv2.drawContours(image, [box], -1, color_to_bgr[max_contour_color], 2)
        cv2.imwrite("images/tmp.jpg", image)
        # cv2.putText(image, '(' + str(world_x) + ',' + str(world_y) + ')', (min(box[0, 0], box[2, 0]), box[2, 1] - 10),
                # cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_to_bgr[max_contour_color], 1)  #绘制中心点
        return image, img_centerx, img_centery, rect[2], max_contour_color
        # return image, world_x, world_y, rect[2], max_contour_color
    
    return False