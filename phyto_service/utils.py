from base64 import b64decode
from typing import Tuple

import albumentations as albu
import numpy as np
import cv2.cv2 as cv2
from PIL import Image

from phyto_service import settings

transform = albu.Compose([albu.PadIfNeeded(320, 480)])


def image_preprocessing(img_b64: str,
                        target_size: Tuple[int, int]) -> Tuple[np.ndarray,
                                                               np.ndarray,
                                                               bool]:
    img_bytes = b64decode(img_b64)

    image_original = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), -1)
    if image_original.shape[2] == 4:
        image_original = cv2.cvtColor(image_original, cv2.COLOR_BGRA2BGR)
    image = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    if image.shape[0] > target_size[0] or image.shape[1] > target_size[1]:
        im_pil = Image.fromarray(image)
        im_pil_resized = im_pil.resize(target_size[::-1],
                                       resample=Image.Resampling.BICUBIC)
        image = np.array(im_pil_resized)
        padded = False
    else:
        padded = True
        transformed = transform(image=image)
        image = transformed["image"]
    return image, image_original, padded
