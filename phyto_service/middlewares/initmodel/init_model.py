from dramatiq import Middleware

from phyto_service import settings
from .tools import init_fpn_model


class InitModelMiddleware(Middleware):

    def after_process_boot(self, broker):
        init_fpn_model(settings.MODEL_PATH)


