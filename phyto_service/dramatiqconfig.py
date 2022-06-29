import dramatiq
from dramatiq.results import Results
from dramatiq.results.backends import RedisBackend
from dramatiq.brokers.redis import RedisBroker
from dramatiq.middleware import CurrentMessage, Prometheus

from phyto_service import settings
from phyto_service.middlewares import (
    InitModelMiddleware,
)

broker = RedisBroker(
    host=settings.REDIS_HOST, port=settings.REDIS_PORT,
    db=settings.REDIS_DB, password=settings.REDIS_PASSWORD
)
backend = RedisBackend(client=broker.client)
broker.middleware = list(filter(lambda x: not isinstance(x, Prometheus),
                                broker.middleware))
#broker.middleware.insert(0, PatchedPrometheus())
broker.add_middleware(Results(backend=backend))
broker.add_middleware(InitModelMiddleware())
broker.add_middleware(CurrentMessage())
dramatiq.set_broker(broker)
