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
import logging, sys
from os import environ, path
from sys import stdout
import config_utils
import IPCUtils as ipc_utils
import awsiot.greengrasscoreipc.client as client


try:
    ipc = ipc_utils.IPCUtils()
    ipc_client = client.GreengrassCoreIPCClient(ipc.connect())
    config_utils.logger.info("Created IPC client...")
except Exception as e:
    config_utils.logger.error(
        "Exception occured during the creation of an IPC client: {}".format(e)
    )
    exit(1)


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


def prepare_img(img_filepath, img_shape=(224,224), verbose=0):
    # Prepare image
    img = Image.open(img_filepath)
    img = img.resize(img_shape)
    img = np.asarray(img).astype('float32')
    img /= 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
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
        
        sort_classes_by_probs = np.argsort(probs)[::-1]
        max_no_of_results = 3
        # for i in sort_classes_by_probs[:max_no_of_results]:
        #     print("[ Class: {}, Score: {} ]".format(label_map[str(i)], probs[i]))

        message = '{"class_id":"' + str(pred_cls_idx) + '"' + ',"class":"' + pred_cls_str + '"' + ',"score":"' + str(pred_score) +'"}'
        payload = {
             "message": message,
             "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        }

        config_utils.logger.info(f"predict={payload}")        

        if config_utils.ENABLE_SEND_MESSAGE:
            ipc.publish_results_to_cloud(ipc_client, payload)
            
        if verbose == 1:        
            print(json.dumps(payload, sort_keys=True, indent=4))
        
        
        return payload

    except Exception as e:
        print("Exception occurred during prediction: %s", e)
        

label_map = get_label_map_imagenet(config_utils.LABEL_FILE)
extensions = (f"{config_utils.SAMPLE_IMAGE_DIR}/*.jpg", f"{config_utils.SAMPLE_IMAGE_DIR}/*.jpeg")
img_filelist = [f for f_ in [glob.glob(e) for e in extensions] for f in f_]
config_utils.logger.info(img_filelist)

# Load model
dlr_model = dlr.DLRModel(config_utils.MODEL_DIR, 'cpu', 0)

os.system("echo {}".format("Using dlr from '{}'.".format(dlr.__file__)))
os.system("echo {}".format("Using numpy from '{}'.".format(np.__file__)))

idx = 0
num_imgs = len(img_filelist)

while True:
    # Prepare image
    img_filepath = img_filelist[idx]
    img = prepare_img(img_filepath)
    
    # Predict
    payload = predict(img, label_map, verbose=1)
    
    idx += 1
    if idx % num_imgs == 0:
        idx = 0
        print(idx)

    # Append the message to the log file.
    #with open(log_filepath, 'a') as f:
    #    print(payload, file=f)
    
    time.sleep(config_utils.DEFAULT_PREDICTION_INTERVAL_SECS)
