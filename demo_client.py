import lib
import cv2

od_client = lib.ImageClient('localhost:30307')
img = cv2.imread('images/image2.jpg')
od_client.upload(img)