from logging import INFO, StreamHandler, getLogger
import os
from os import environ, path
from sys import stdout

from awsiot.greengrasscoreipc.model import QOS

# Set all the constants
SCORE_THRESHOLD = 0.25
MAX_NO_OF_RESULTS = 3
SHAPE = (224,224)
QOS_TYPE = QOS.AT_LEAST_ONCE
TIMEOUT = 10

# Intialize all the variables with default values
DEFAULT_PREDICTION_INTERVAL_SECS = 5
ENABLE_SEND_MESSAGE = True
TOPIC = "ml/example/imgclassification"

# Get a logger
logger = getLogger()
handler = StreamHandler(stdout)
logger.setLevel(INFO)
logger.addHandler(handler)

# Get the model directory and images directory from the env variables.
MODEL_DIR = path.expandvars(environ.get("MODEL_DIR"))
SAMPLE_IMAGE_DIR = path.expandvars(environ.get("SAMPLE_IMAGE_DIR"))
#MODEL_DIR = f'{os.getcwd()}/model'
#SAMPLE_IMAGE_DIR = f'{os.getcwd()}/sample_images'
LABEL_FILE = path.join(MODEL_DIR, "imagenet1000_clsidx_to_labels.txt")
