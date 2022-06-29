import logging
from base64 import b64encode

import numpy as np
import torch
import segmentation_models_pytorch as smp
from cv2 import cv2
from dramatiq import actor
from PIL import Image

from phyto_service import settings
from phyto_service.decorators import done_for
from phyto_service.middlewares.initmodel import get_model_by_key
from phyto_service.utils import image_preprocessing

logger = logging.getLogger(__name__)

preprocessing_fn = smp.encoders.get_preprocessing_fn(
        settings.ENCODER, settings.ENCODER_WEIGHTS
    )


@actor(actor_name=settings.TASK_NAME, queue_name=settings.TASK_QUEUE,
       max_retries=0, store_results=True)
@done_for
def segmentation(img_b64: str, *args, **kwargs) -> str:
    target_size = (320, 320)
    image, image_original, is_padded = image_preprocessing(img_b64, target_size)
    # TODO set device
    image_trans = preprocessing_fn(x=image).transpose(2, 0, 1).astype('float32')
    device = "cpu" if not settings.USE_CUDA else "cuda"
    image_tensor = torch.from_numpy(image_trans).unsqueeze(0).to(device)
    model = get_model_by_key(settings.MODEL_PATH)
    pr_mask = model(image_tensor)
    pr_mask = (pr_mask.squeeze().cpu().detach().numpy().round())
    #pr_mask[pr_mask == 1] = 255
    #pr_mask = pr_mask.astype(np.uint8)
    if is_padded is False and pr_mask.shape[:2] != image_original.shape[:2]:
        pr_mask_pil = Image.fromarray(pr_mask)
        pr_mask_pil_resized = pr_mask_pil.resize(image_original.shape[:2][::-1],
                                                 resample=Image.Resampling.BICUBIC)
        pr_mask = np.array(pr_mask_pil_resized)

    #image_hsv = cv2.cvtColor(image_original, cv2.COLOR_BGR2HSV)
    image_mask_hsv = image_original[:, :, 2]
    image_mask_hsv[pr_mask >= 1] = 200
    image_original[:, :, 2] = image_mask_hsv
    #image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    _, encoded_image = cv2.imencode('.png', image_original)
    str_bytes = b64encode(encoded_image).decode()
    return str_bytes

