# # Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# # SPDX-License-Identifier: MIT-0
# import sys
# import datetime
# import time

# while True:
    
#     message = f"Hello, {sys.argv[1]}! Current time: {str(datetime.datetime.now())}."
    
#     # Print the message to stdout.
#     print(message)
    
#     # Append the message to the log file.
#     with open('/tmp/Greengrass_HelloWorld.log', 'a') as f:
#         print(message, file=f)
        
#     time.sleep(1)



# # Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# # SPDX-License-Identifier: MIT-0
# import time
# import datetime
# import json
# import awsiot.greengrasscoreipc
# from awsiot.greengrasscoreipc.model import (
#     PublishToTopicRequest,
#     PublishMessage,
#     JsonMessage
# )
# from dummy_sensor import DummySensor


# TIMEOUT = 10
# publish_rate = 1.0

# ipc_client = awsiot.greengrasscoreipc.connect()

# sensor = DummySensor()

# topic = "my/topic"





TIMEOUT = 10

import dlr
import os
import json
import numpy as np
import glob
import time
from datetime import datetime
from random import randint
from io import BytesIO
from PIL import Image

import awsiot.greengrasscoreipc
from awsiot.greengrasscoreipc.model import (
    PublishToTopicRequest,
    PublishMessage,
    JsonMessage
)
from awsiot.eventstreamrpc import Connection, LifecycleHandler, MessageAmendment
import awsiot.greengrasscoreipc.client as client


def send_message(ipc_client, message, topic, publish_rate=1.0):
    request = PublishToTopicRequest()
    request.topic = topic
    request.topic_name = "neo-detect"
    
    publish_message = PublishMessage()
    publish_message.json_message = JsonMessage()
    publish_message.json_message.message = message

    request.publish_message = publish_message
    operation = ipc_client.new_publish_to_topic()
    operation.activate(request)
    future = operation.get_response()
    future.result(TIMEOUT)

    print("publish")
    time.sleep(1/publish_rate)


def softmax(x):
    x_exp = np.exp(x - np.max(x))
    f_x = x_exp / np.sum(x_exp)
    return f_x

def get_label_map(label_file):
    label_map = {}
    labels = open(label_file, 'r')
    
    for line in labels:
        line = line.rstrip("\n")
        ids = line.split(',')
        label_map[int(ids[0])] = ids[2] 
        
    return label_map


def get_label_map_imagenet(label_file):
    label_map = {}
    with open(label_file, 'r') as f:
        for line in f:
            key, val = line.strip().split(':')
            label_map[key] = val.replace(',', '')
    return label_map


def prepare_img(img_filepath, img_size=224, verbose=0):
    # Prepare image
    img = Image.open(img_filepath)
    img = img.resize((img_size, img_size))
    img = np.asarray(img).astype('float32')
    img /= 255.0
    img = np.transpose(img, (2,0,1))
    img  = np.expand_dims(img, axis=0)
    
    if verbose == 1:
        print(img.shape)

    return img

def predict(image_data, label_map, verbose=0):
    r"""
    Predict image with DLR.
    :param image_data: numpy array of the image_data
    :label_map: class mapping dictionary
    """
    try:
        model_output = dlr_model.run(image_data)
        probs = softmax(model_output[0][0])
        pred_cls_idx = np.argmax(model_output)
        pred_score = np.max(probs)
        pred_cls_str = label_map[str(pred_cls_idx)].strip()
        
        # sort_classes_by_probs = np.argsort(probs)[::-1]
        # max_no_of_results = 3
        # for i in sort_classes_by_probs[:max_no_of_results]:
        #     print("[ Class: {}, Score: {} ]".format(label_map[str(i)], probs[i]))

        message = '{"class_id":"' + str(pred_cls_idx) + '"' + ',"class":"' + pred_cls_str + '"' + ',"score":"' + str(pred_score) +'"}'

        payload = {
             "message": message,
             "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        }
        
        if verbose == 1:        
            print(json.dumps(payload, sort_keys=True, indent=4))
        
        return payload

    except Exception as e:
        print("Exception occurred during prediction: %s", e)
        

#hostname = os.getenv("AWS_GG_NUCLEUS_DOMAIN_SOCKET_FILEPATH_FOR_COMPONENT")
#print("hostname=", hostname)
#print("svcid=", os.getenv("SVCUID"))
#ipc_client = awsiot.greengrasscoreipc.connect()


topic = "my/topic"
model_path = 'model'
test_path = 'test'
log_filepath = '/tmp/Greengrass_img.log'
label_file = 'imagenet1000_clsidx_to_labels.txt'
label_map = get_label_map_imagenet(label_file)
img_filelist = glob.glob(f"{test_path}/*.jpg")

# Load model
dlr_model = dlr.DLRModel(model_path, 'cpu', 0)

os.system("echo {}".format("Using dlr from '{}'.".format(dlr.__file__)))
os.system("echo {}".format("Using numpy from '{}'.".format(np.__file__)))
os.system("echo {}".format("GGV2 Log path '{}'.".format(log_filepath)))

while True:
    # Prepare image
    idx = randint(0, len(img_filelist)-1)
    img_filepath = img_filelist[idx]
    img = prepare_img(img_filepath)
    
    # Predict
    payload = predict(img, label_map, verbose=1)

    # Append the message to the log file.
    with open(log_filepath, 'a') as f:
        print(payload, file=f)

    # send_message(payload, topic)
    
    time.sleep(2)

# def predict(image_data):
#     r"""
#     Predict image with DLR.
#     :param image: numpy array of the Image inference with.
#     """
#     try:
#         # Run DLR to perform inference with DLC optimized model
#         model_output = dlr_model.run(image_data)
#         max_score_id = np.argmax(model_output)
#         max_score = np.max(model_output)
#         print("max score id:",max_score_id)
#         print("class:",labels[max_score_id])
#         print("max score",str(max_score))
#         probabilities = model_output[0][0]
#         sort_classes_by_probability = np.argsort(probabilities)[::-1]
#         results_file = "{}/{}.log".format(results_directory,os.path.basename(os.path.realpath(model_path)))
#         message = '{"class":"' + labels[max_score_id] + '"' + ',"confidence":"' + str(max_score) +'"}'
#         payload = {
#             "message": message,
#             "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
#         }
#         topic = "demo/topic"
#         if enableSendMessages:
#           ipc_client.new_publish_to_iot_core().activate(
#               request=PublishToIoTCoreRequest(topic_name=topic, qos='0',
#                                             payload=json.dumps(payload).encode()))

#         with open(results_file, 'a') as f:
#             print("{}: Top {} predictions with score {} or above ".format(str(
#                 datetime.now()), max_no_of_results, score_threshold), file=f)
#             for i in sort_classes_by_probability[:max_no_of_results]:
#                 if probabilities[i] >= score_threshold:
#                     print("[ Class: {}, Score: {} ]".format(
#                         labels[i], probabilities[i]), file=f)

#     except Exception as e:
#         print("Exception occurred during prediction: %s", e)


# def predict_from_image(image):
#     r"""
#     reshape the captured image and predict using it.
#     """
#     #cvimage = cv2.resize(image, reshape)
#     predict(image)


# def send_mqtt_message(message):
#     request = PublishToIotCoreRequest()
#     request.topic_name = "neo-detect"
#     request.payload = bytes(message, "utf-8")
#     request.qos = QOS.AT_LEAST_ONCE
#     operation = ipc_client.new_publish_to_iot_core()
#     operation.activate(request)
#     future = operation.get_response()
#     future.result(TIMEOUT)


# def predict_from_cam():
#     if camera is None:
#         print("Unable to support camera")
#         return
#     if platform.machine() == "armv7l":  # RaspBerry Pi
#         stream = io.BytesIO()
#         camera.start_preview()
#         time.sleep(2)
#         camera.capture(stream, format='jpeg')
#         # Construct a numpy array from the stream
#         data = np.fromstring(stream.getvalue(), dtype=np.uint8)
#         # "Decode" the image from the array, preserving colour
#         cvimage = cv2.imdecode(data, 1)
#     elif platform.machine() == "aarch64":  # Nvidia Jetson TX
#         if camera.isOpened():
#             ret, cvimage = camera.read()
#             cv2.destroyAllWindows()
#         else:
#             raise RuntimeError("Cannot open the camera")
#     elif platform.machine() == "x86_64":  # Deeplens
#         ret, cvimage = camera.getLastFrame()
#         if ret == False:
#             raise RuntimeError("Failed to get frame from the stream")
#     return predict_from_image(cvimage)



# # Passed arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("--accelerator",
#                     "-a",
#                     default="gpu",
#                     help="gpu/cpu/opencl")
# parser.add_argument("--modelPath",
#                     "-m",
#                     help="path to model")
# parser.add_argument("--mlRootPath",
#                     "-p",
#                     help="path to inference result and images")
# parser.add_argument("--imageName",
#                     "-i",
#                     help="image name")
# parser.add_argument("--interval",
#                     "-s", default=60,
#                     help="prediction interval in seconds")

# args = parser.parse_args()

# model_path = args.modelPath
# context = args.accelerator
# mlRootPath = args.mlRootPath
# imageName = args.imageName
# prediction_interval_secs = args.interval
# reshape = (224, 224)
# score_threshold = 0.3
# max_no_of_results = 5
# camera = None
# image_data = None
# sample_image = (mlRootPath + "/images/" + imageName).format(
#     os.path.dirname(os.path.realpath(__file__)))
# results_directory = mlRootPath + "/inference_log/"
# # Create the results directory if it does not exist already
# os.makedirs(results_directory, exist_ok=True)

# # Initialize example Resnet model
# dlr_model = dlr.DLRModel(model_path, context)

# os.system("echo {}".format("Inference logs can be found under the directory '{}' in the name of the model used. ".format(
#     results_directory)))
# # Load image based on the format - support jpg,jpeg,png and npy.
# if imageName.endswith(".jpg", -4,) or imageName.endswith(".png", -4,) or imageName.endswith(".jpeg", -5,):
#     image = bytearray(open(sample_image, 'rb').read())
#     image_data = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
#     image_data = cv2.resize(image_data, (224,224))
#     print("loaded image:",imageName)
# elif imageName.endswith(".npy", -4,):
#     # the shape for the resnet18 model is [1,3,224,224]
#     image_data = np.load(sample_image).astype(np.float32)

# # enable_camera()

# while True:
#     # predict_from_cam()
#     # comment the below if-else statements and uncomment the above line after enabling the camera to predict images from the camera
#     if image_data is not None:
#         predict_from_image(image_data)
#     else:
#         os.system("Images of format jpg,jpeg,png and npy are only supported.")
#     time.sleep(int(prediction_interval_secs))