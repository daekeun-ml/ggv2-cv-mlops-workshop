import ast
import datetime
import io
import json
import os
import platform
import sys
import time

import config_utils
import cv2
import IPCUtils as ipc_utils
import numpy as np
import tflite_runtime.interpreter as tflite

config_utils.logger.info("Using tflite from '{}'.".format(sys.modules[tflite.__package__].__file__))
config_utils.logger.info("Using np from '{}'.".format(np.__file__))
config_utils.logger.info("Using cv2 from '{}'.".format(cv2.__file__))


def get_bbox_abs_coordinate(box, ih, iw):
    x = int(box[0] * iw)
    y = int(box[1] * ih)
    w = int(box[2] * iw - x)
    h = int(box[3] * ih - y)
    return x, y, w, h


def get_label_map(label_file):
    label_map = {}
    labels = open(label_file, 'r')
    
    for line in labels:
        line = line.rstrip("\n")
        ids = line.split(',')
        label_map[int(ids[0])] = ids[2] 
        
    return label_map


# Read labels file
label_path = os.path.join(config_utils.MODEL_DIR, config_utils.LABEL_FILE_NAME)
config_utils.logger.info("class info path: '{}'.".format(label_path))
label_map = get_label_map(label_path)
label_list = list(label_map.values())

try:
    interpreter = tflite.Interpreter(
        model_path=os.path.join(config_utils.MODEL_DIR, config_utils.MODEL_FILE_NAME)
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    config_utils.logger.info("Exception occured during the allocation of tensors: {}".format(e))
    exit(1)


def predict_from_cam():
    r"""
    Captures an image using camera and sends it for prediction
    """
    cvimage = None
    if config_utils.CAMERA is None:
        config_utils.logger.error("Unable to support camera.")
        exit(1)
    if platform.machine() == "armv7l":  # RaspBerry Pi
        stream = io.BytesIO()
        config_utils.CAMERA.start_preview()
        time.sleep(2)
        config_utils.CAMERA.capture(stream, format="jpeg")
        # Construct a numpy array from the stream
        data = np.fromstring(stream.getvalue(), dtype=np.uint8)
        # "Decode" the image from the array, preserving colour
        cvimage = cv2.imdecode(data, 1)
    elif platform.machine() == "aarch64":  # Nvidia Jetson Nano
        if config_utils.CAMERA.isOpened():
            ret, cvimage = config_utils.CAMERA.read()
            cv2.destroyAllWindows()
        else:
            raise RuntimeError("Cannot open the camera")
    elif platform.machine() == "x86_64":  # Deeplens
        ret, cvimage = config_utils.CAMERA.getLastFrame()
        if ret == False:
            raise RuntimeError("Failed to get frame from the stream")
    if cvimage is not None:
        return predict_from_image(cvimage)
    else:
        config_utils.logger.error("Unable to capture an image using camera")
        exit(1)


def load_image(image_path):
    r"""
    Validates the image type irrespective of its case. For eg. both .PNG and .png are valid image types.
    Also, accepts numpy array images.

    :param image_path: path of the image on the device.
    :return: a numpy array of shape (1, input_shape_x, input_shape_y, no_of_channels)
    """
    # Case insenstive check of the image type.
    img_lower = image_path.lower()
    if (
        img_lower.endswith(
            ".jpg",
            -4,
        )
        or img_lower.endswith(
            ".png",
            -4,
        )
        or img_lower.endswith(
            ".jpeg",
            -5,
        )
    ):
        try:
            image_data = cv2.imread(image_path)
        except Exception as e:
            config_utils.logger.error(
                "Unable to read the image at: {}. Error: {}".format(image_path, e)
            )
            exit(1)
    elif img_lower.endswith(
        ".npy",
        -4,
    ):
        image_data = np.load(image_path)
    else:
        config_utils.logger.error("Images of format jpg,jpeg,png and npy are only supported.")
        exit(1)
    return image_data


def predict_from_image(img):
    #cv image
    r"""
    Resize the image to the trained model input shape and predict using it.

    :param image: numpy array of the image passed in for inference
    """
    ih, iw, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (config_utils.MODEL_INPUT_SIZE, config_utils.MODEL_INPUT_SIZE))
    img = np.asarray(img).astype('float32')
    img = img / 255. 
    img = np.expand_dims(img, axis=0) 

    config_utils.logger.info("image shape after resizing: '{}'.".format(img.shape))    
    predict(img, ih, iw)
    
    
def enable_camera():
    r"""
    Checks of the supported device types and access the camera accordingly.
    """
    if platform.machine() == "armv7l":  # RaspBerry Pi
        import picamera

        config_utils.CAMERA = picamera.PiCamera()
    elif platform.machine() == "aarch64":  # Nvidia Jetson TX
        config_utils.CAMERA = cv2.VideoCapture(
            "nvarguscamerasrc ! video/x-raw(memory:NVMM),"
            + "width=(int)1920, height=(int)1080, format=(string)NV12,"
            + "framerate=(fraction)30/1 ! nvvidconv flip-method=2 !"
            + "video/x-raw, width=(int)1920, height=(int)1080,"
            + "format=(string)BGRx ! videoconvert ! appsink"
        )
    elif platform.machine() == "x86_64":  # Deeplens
        import awscam

        config_utils.CAMERA = awscam
        
        
def batch_iou(boxes, box):
    """Compute the Intersection-Over-Union of a batch of boxes with another box.
    Args:
        box1: 2D array of [cx, cy, width, height].
        box2: a single array of [cx, cy, width, height]
    Returns:
        ious: array of a float number in range [0, 1].
    """
    lr = np.maximum(
          np.minimum(boxes[:,0]+0.5*boxes[:,2], box[0]+0.5*box[2]) - \
          np.maximum(boxes[:,0]-0.5*boxes[:,2], box[0]-0.5*box[2]),
          0
    )
    tb = np.maximum(
          np.minimum(boxes[:,1]+0.5*boxes[:,3], box[1]+0.5*box[3]) - \
          np.maximum(boxes[:,1]-0.5*boxes[:,3], box[1]-0.5*box[3]),
          0
    )
    inter = lr*tb
    union = boxes[:,2]*boxes[:,3] + box[2]*box[3] - inter
    return inter/union


def nms(boxes, class_ids, probs, threshold=0.8):
    """Non-Maximum supression.
    Args:
        boxes: array of [cx, cy, w, h] (center format)
        class_ids: array of classes
        probs: array of probabilities
    threshold: two boxes are considered overlapping if their IOU is largher than this threshold
    form: 'center' or 'diagonal'
    Returns:
        boxes: boxes to be selected by non-max suppression
        class_ids: classes to be selected by non-max suppression
        class_ids: probabilities to be selected by non-max suppression
        keep: array of True or False.
    """
    order = probs.argsort()[::-1]
    keep = [True]*len(order)

    for i in range(len(order)-1):
        ovps = batch_iou(boxes[order[i+1:]], boxes[order[i]])
        for j, ov in enumerate(ovps):
            if ov > threshold:
                keep[order[j+i+1]] = False
                
    return boxes[keep], class_ids[keep], probs[keep], keep


def filter_boxes(bboxes, pred_classes, model_input_size=416, score_threshold=0.4):
    boxes = []
    class_ids = []
    for i, box in enumerate(bboxes):
        if pred_classes[i][1] >= score_threshold:
            x1 = (box[0] - box[2]/2) / model_input_size
            y1 = (box[1] - box[3]/2) / model_input_size
            x2 = (box[0] + box[2]/2) / model_input_size
            y2 = (box[1] + box[3]/2) / model_input_size
            boxes.append([x1,y1,x2,y2]) 
            class_ids.append(pred_classes[i])
    
    classes, probs = zip(*class_ids)
    classes = np.array(classes)
    probs = np.array(probs)
    boxes = np.array(boxes)
    return boxes, classes, probs


def predict(image_data, ih, iw):
    r"""
    Performs object detection and predicts using the model.

    :param image_data: numpy array of the resized image passed in for inference.
    """
    PAYLOAD = {}
    PAYLOAD["timestamp"] = str(datetime.datetime.now())
    PAYLOAD["inference-type"] = "object-detection"
    PAYLOAD["inference-description"] = "Top {} predictions with score {} or above ".format(
        config_utils.MAX_NO_OF_RESULTS, config_utils.SCORE_THRESHOLD
    )
    PAYLOAD["inference-results"] = []
    
    try:
        # Get prediction results from tflite
        interpreter.set_tensor(input_details[0]["index"], image_data)
        interpreter.invoke()
        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        
        # Get classes and bounding boxes
        bboxes = np.array([tuple(x) for x in pred[0][0]])
        pred_classes = []
        for c in pred[1][0]:
            pred_class = (int(np.argmax(c)), float(np.max(c)))
            pred_classes.append(pred_class)
        boxes, class_ids, probs = filter_boxes(bboxes, pred_classes, config_utils.MODEL_INPUT_SIZE)
        boxes, class_ids, probs, keep = nms(boxes, class_ids, probs)
        
        if class_ids is not None:
            for box, cid, prob in zip(boxes, class_ids, probs):
                if prob >= config_utils.SCORE_THRESHOLD:    
                    x,y,w,h = get_bbox_abs_coordinate(box, ih, iw)
                    class_str = label_list[int(cid)]
                    #print((x,y,w,h), cid, class_str, prob)
                    result = {
                        "Label": str(class_str),
                        "Label_index": str(cid),
                        "Score": str(prob),
                        "Box": str((x,y,w,h))
                    }
                    PAYLOAD["inference-results"].append(result)

        config_utils.logger.info(json.dumps(PAYLOAD))
        if config_utils.TOPIC.strip() != "":
            ipc_utils.IPCUtils().publish_results_to_cloud(PAYLOAD)
        else:
            config_utils.logger.info("No topic set to publish the inference results to the cloud.")
    except Exception as e:
        config_utils.logger.error("Exception occured during prediction: {}".format(e))
