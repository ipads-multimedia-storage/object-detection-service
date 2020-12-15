import sys
import os
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir + "/proto")
from io import BytesIO
from concurrent import futures

import numpy as np
import grpc
import time

import image_pb2
import image_pb2_grpc
import detector

def ndarray_to_image(nda):
    nda_bytes = BytesIO()
    np.save(nda_bytes, nda, allow_pickle=False)
    return image_pb2.Image(ndarray=nda_bytes.getvalue())

def image_to_ndarray(image):
    nda_bytes = BytesIO(image.ndarray)
    return np.load(nda_bytes, allow_pickle=False)

class ImageClient:
    def __init__(self, address):
        options = [('grpc.max_send_message_length', 512 * 1024 * 1024), \
            ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
        channel = grpc.insecure_channel(address, options=options)
        self.stub = image_pb2_grpc.ImageServerStub(channel)
    
    def upload(self, ndarray):
        response = self.stub.upload(ndarray_to_image(ndarray))
        if response.valid == False:
            return False
        else:
            img = image_to_ndarray(response.processed_image)
            return img, response.x, response.y, response.color, response.angle

class ImageServer(image_pb2_grpc.ImageServerServicer):
    def __init__(self):

        class ImageServicer(image_pb2_grpc.ImageServerServicer):
            def __init__(self):
                self.count = 0

            def upload(self, image, context):
                result = detector.detect(image_to_ndarray(image))
                if result == False:
                    # no object detected
                    return image_pb2.Reply(valid=False)
                else:
                    img, x, y, angle, color = result
                    img = ndarray_to_image(img)
                    return image_pb2.Reply(valid=True, processed_image=img,
                        color=color, x=x, y=y, angle=angle)
        
        options = [('grpc.max_send_message_length', 512 * 1024 * 1024), \
            ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=options)
        image_pb2_grpc.add_ImageServerServicer_to_server(ImageServicer(), self.server)
    
    def start(self, port):
        self.server.add_insecure_port(f'[::]:{port}')
        self.server.start()

        try:
            while True:
                time.sleep(60 * 60 * 24)
        except KeyboardInterrupt:
            self.server.stop(0)