import lib
import cv2

img = cv2.imread('images/image1.jpg')
od_client = lib.ObjectDetectionClient('localhost:30307')

result = od_client.upload(img)
if result is not False:
    processed_img, x, y, color, angle = result
    processed_img_path = 'images/tmp.jpg'
    cv2.imwrite(processed_img_path, processed_img)
    print('coord: ({}, {}), color: {}, angle: {}'.format(x, y, color, angle))