import logging

from phyto_service.decorators import alone_apply
from phyto_service import settings
from phyto_service.camvid_lightning import PhytoModel

logger = logging.getLogger(__name__)
models = {}


@alone_apply(redis_host=settings.REDIS_HOST, redis_port=settings.REDIS_PORT,
             timeout=120, prefix_key='intent-init')
def init_fpn_model(model_path: str):
    if models.get(model_path) is None:
        model = PhytoModel.load_from_checkpoint(model_path, map_location="cpu")
        if settings.USE_CUDA is True:
            model.to("cuda")
            logger.info(f"use cuda: {model_path}")
        model.eval()
        models[model_path] = model
        logger.debug(f"Finish init new intent model: {model_path}")
    else:
        logger.info(f"Intent model already loaded: {model_path}")


def get_model_by_key(key: str):
    model = models.get(key)
    if model is None:
        raise KeyError(f"Target model is not loaded: {key}")
    return model
