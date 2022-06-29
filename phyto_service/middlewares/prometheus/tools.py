import logging

from dramatiq.middleware.prometheus import _run_exposition_server

logger = logging.getLogger(__name__)


def run_prometheus_fork():
    try:
        _run_exposition_server()
    except OSError:
        logger.debug("Prometheus server already started.")
