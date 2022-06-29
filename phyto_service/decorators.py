import logging
import os
from time import perf_counter
from functools import wraps

from dramatiq.middleware import CurrentMessage
from redis import Redis


logger = logging.getLogger(__name__)


def alone_apply(redis_host: str = "", redis_port: str = "", redis_db: int = 0,
                redis_password: str = None, prefix_key: str = "",
                timeout: int = None):
    """Enforce only one celery task at a time."""

    def decorator(run_func):
        @wraps(run_func)
        def wrapper(*args, **kwargs):
            ret_value = None
            have_lock = False
            if os.environ.get("USE_CUDA", "false").lower() == "true":
                end_key = f':{os.environ.get("CUDA_VISIBLE_DEVICES", "0")}'
            else:
                end_key = ":cpu"
            key = prefix_key + end_key
            redis_client = Redis(host=redis_host, port=redis_port,
                                 db=redis_db, password=redis_password)
            lock = redis_client.lock(key, timeout=timeout)
            try:
                # wait until unlock
                have_lock = lock.acquire(blocking=True)
                if have_lock:
                    ret_value = run_func(*args, **kwargs)
            finally:
                if have_lock:
                    lock.release()
            redis_client.close()
            return ret_value

        return wrapper

    return decorator


def done_for(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        message = CurrentMessage.get_current_message()
        begin_at = perf_counter()
        result = fn(*args, **kwargs)
        finish_at = perf_counter()
        logger.info(f'{fn.__name__.upper()} {message.message_id} done for '
                    f'{finish_at-begin_at}')
        return result
    return wrapper
