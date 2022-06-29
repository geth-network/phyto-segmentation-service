from dramatiq.middleware import Prometheus

from .tools import run_prometheus_fork


class PatchedPrometheus(Prometheus):
    @property
    def forks(self):
        return [run_prometheus_fork]
