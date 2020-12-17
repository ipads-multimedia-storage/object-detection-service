import lib
import cv2

od_client = lib.ObjectDetectionClient('localhost:30307')
img = cv2.imread('images/image1.jpg')
od_client.upload(img)