from uuid import uuid4
from base64 import b64encode
from io import BytesIO
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor

import dramatiq
import numpy as np
from phyto_service.dramatiqconfig import backend
from dramatiq import Message
import redis


def task(img, idx):
    m = Message(
        message_id=idx,
        queue_name='task_queue',
        actor_name='segment_task',
        args=(img,), kwargs={}, options={}
    )
    dramatiq.get_broker().enqueue(m)
    #m.get_result(backend=backend, block=True, timeout=3600)


def main():
    import sys
    with open('./slice_0_100_70_6.png','rb') as f:
        img = BytesIO(f.read())
    s = []
    n = 500
    j = 500
    img = b64encode(img.getvalue()).decode()
    r = redis.Redis()
    with ThreadPoolExecutor(max_workers=10) as executor:
        for j in range(j):
            print(j)
            ids = [uuid4().hex for _ in range(n)]
            start = perf_counter()
            for i in range(n):
                executor.submit(task, img, ids[i])
            m = Message(
                message_id=ids[-1],
                queue_name='task_queue',
                actor_name='segment_task',
                args=(img,), kwargs={}, options={}
            )
            m.get_result(backend=backend, block=True, timeout=3600 * 1000)
            end = perf_counter()
            r.flushall(asynchronous=False)
            s.append(end - start)
            print(f'avg: {np.mean(s)}')
            print(f'median: {np.median(s)}')
            print(f'max: {np.max(s)}')
            print(f'min: {np.min(s)}')
            print(f'std: {np.std(s)}')
            print(f'sum: {np.sum(s)}')

    print(f'avg: {np.mean(s)}')
    print(f'median: {np.median(s)}')
    print(f'max: {np.max(s)}')
    print(f'min: {np.min(s)}')
    print(f'std: {np.std(s)}')
    print(f'sum: {np.sum(s)}')


if __name__ == '__main__':
    main()
