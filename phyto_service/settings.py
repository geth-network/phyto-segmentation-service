import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.resolve()
TASK_NAME = os.environ.get("TASK_NAME", "segment_task")
TASK_QUEUE = os.environ.get("TASK_QUEUE", "task_queue")

REDIS_HOST = os.environ.get("REDIS_HOST", 'localhost')
REDIS_PORT = os.environ.get("REDIS_PORT", "6379")
REDIS_DB = int(os.environ.get("REDIS_DB", 0))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)

DEBUG = bool(os.environ.get("DEBUG", "false").lower() == "true")
USE_CUDA = bool(os.environ.get("USE_CUDA", "false").lower() == "true")

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['phyto']
ACTIVATION = 'sigmoid'

MODEL_PATH = '/home/tuhas/PycharmProjects/phyto-segmentation-service/best_model.ckpt'
