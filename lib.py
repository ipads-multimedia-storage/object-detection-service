import os, sys
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir + "/proto")

import time
import numpy as np
from io import BytesIO
from concurrent import futures

import grpc
import object_detection_pb2
import object_detection_pb2_grpc
import detector

def ndarray_to_bytes(nda):
    nda_bytes = BytesIO()
    np.save(nda_bytes, nda, allow_pickle=False)
    return object_detection_pb2.Image(payload=nda_bytes.getvalue())

def bytes_to_ndarray(_bytes):
    nda_bytes = BytesIO(_bytes.payload)
    return np.load(nda_bytes, allow_pickle=False)

class ObjectDetectionClient:
    def __init__(self, address):
        options = [('grpc.max_send_message_length', 512 * 1024 * 1024), \
            ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
        channel = grpc.insecure_channel(address, options=options)
        self.stub = object_detection_pb2_grpc.ObjectDetectionServerStub(channel)
    
    def upload(self, ndarray):
        response = self.stub.upload(ndarray_to_bytes(ndarray))
        if response.object_detected == False:
            return False
        else:
            img = bytes_to_ndarray(response.processed_image)
            return img, response.x, response.y, response.color, response.angle

class ObjectDetectionServer(object_detection_pb2_grpc.ObjectDetectionServerServicer):
    def __init__(self):

        class ObjectDetectionServicer(object_detection_pb2_grpc.ObjectDetectionServerServicer):
            def upload(self, image, context):
                result = detector.detect(bytes_to_ndarray(image))
                if result == False:
                    # no object detected
                    return object_detection_pb2.DetectionResult(object_detected=False)
                else:
                    img, x, y, angle, color = result
                    img = ndarray_to_bytes(img)
                    return object_detection_pb2.DetectionResult(object_detected=True, processed_image=img,
                        color=color, x=x, y=y, angle=angle)
        
        options = [('grpc.max_send_message_length', 512 * 1024 * 1024), \
            ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=options)
        object_detection_pb2_grpc.add_ObjectDetectionServerServicer_to_server(
            ObjectDetectionServicer(), self.server)
    
    def start(self, port):
        self.server.add_insecure_port(f'[::]:{port}')
        self.server.start()

        try:
            while True:
                time.sleep(60 * 60 * 24)
        except KeyboardInterrupt:
            self.server.stop(0)