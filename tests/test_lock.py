import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from itertools import chain, groupby
from multiprocessing import Manager, Pool, Process
from pathlib import Path
from queue import Queue
from random import random
from threading import Barrier

import pytest

from pharmpy.lock import (
    AcquiringProcessLevelLockWouldBlockError,
    AcquiringThreadLevelLockWouldBlockError,
    RecursiveDeadlockError,
    path_lock,
)
from pharmpy.utils import TemporaryDirectoryChanger


@contextmanager
def lock(directory):
    with TemporaryDirectoryChanger(directory):
        lock = Path('lock')
        lock.touch()
        path = str(lock)
        yield path


def test_exclusive_threads_non_blocking(tmp_path):
    with lock(tmp_path) as path:

        is_done = Barrier(2)
        is_locked = Barrier(2)

        def thread_lock_first():
            with path_lock(path, shared=False, blocking=False):
                is_locked.wait()
                is_done.wait()

        def thread_lock_last():
            is_locked.wait()
            with path_lock(path, shared=False, blocking=False):
                pass

        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(thread_lock_first)
            last = executor.submit(thread_lock_last)

            with pytest.raises(AcquiringThreadLevelLockWouldBlockError):
                last.result()

            is_done.wait()


def process_lock_first(is_locked, is_done, path):
    with path_lock(path, shared=False, blocking=False):
        is_locked.wait()
        is_done.wait()


def process_lock_last(is_locked, q, path):
    is_locked.wait()
    try:
        with path_lock(path, shared=False, blocking=False):
            pass
        q.put(None)
    except Exception as e:
        q.put(e)


def test_exclusive_processes_non_blocking(tmp_path):
    with lock(tmp_path) as path:

        m = Manager()
        is_done = m.Barrier(2)
        is_locked = m.Barrier(2)
        q = m.Queue()
        Process(target=process_lock_first, args=(is_locked, is_done, path)).start()
        last = Process(target=process_lock_last, args=(is_locked, q, path))
        last.start()

        e = q.get()

        assert isinstance(e, AcquiringProcessLevelLockWouldBlockError)

        is_done.wait()


def test_many_shared_one_exclusive_threads_blocking(tmp_path):
    with lock(tmp_path) as path:

        n = 10

        are_locked = Barrier(n)

        def thread_lock_shared(results, i):
            with path_lock(path, shared=True):
                are_locked.wait()
                time.sleep(1)
                results.append(i)

        def thread_lock_exclusive(results, i):
            are_locked.wait()
            with path_lock(path, shared=False):
                results.append(i)

        with ThreadPoolExecutor(max_workers=n) as executor:
            # NOTE We run the test twice to reuse workers to catch errors where
            # some workers are left in a locked state.
            for _ in range(2):
                are_locked.reset()
                results = []

                for i in range(1, n):
                    executor.submit(thread_lock_shared, results, i)
                last = executor.submit(thread_lock_exclusive, results, 0)

                last.result()

                assert sorted(results) == sorted(range(n))
                assert results[-1] == 0


def process_lock_shared(are_locked, q, path, i):
    with path_lock(path, shared=True):
        are_locked.wait()
        time.sleep(1)
        q.put(i)


def process_lock_exclusive(are_locked, q, path, i):
    are_locked.wait()
    with path_lock(path, shared=False):
        q.put(i)


@pytest.mark.skipif(os.name == 'nt', reason="Windows shared process locks are not implemented.")
def test_many_shared_one_exclusive_processes_blocking(tmp_path):
    with lock(tmp_path) as path:

        n = 10

        with Pool(processes=n) as pool:

            m = Manager()
            results_queue = m.Queue()

            # NOTE We run the test twice to reuse workers to catch errors where
            # some workers are left in a locked state.
            for _ in range(2):

                are_locked = m.Barrier(n)

                for i in range(1, n):
                    pool.apply_async(process_lock_shared, [are_locked, results_queue, path, i])
                last = pool.apply_async(
                    process_lock_exclusive, [are_locked, results_queue, path, 0]
                )

                last.wait()

                results = []
                for i in range(n):
                    results.append(results_queue.get())

                assert sorted(results) == sorted(range(n))
                assert results[-1] == 0


def test_non_reentrant_dead_lock(tmp_path):
    with lock(tmp_path) as path:
        with path_lock(path):
            with pytest.raises(RecursiveDeadlockError):
                with path_lock(path):
                    pass


def test_reentrant(tmp_path):
    with lock(tmp_path) as path:
        with path_lock(path):
            with path_lock(path, reentrant=True):
                pass

        with path_lock(path, reentrant=True):
            with path_lock(path, reentrant=True):
                pass

        with path_lock(path):
            with path_lock(path, reentrant=True):
                with path_lock(path, reentrant=True):
                    pass


def many_exclusive_threads_and_processes_rw_process(path, indices):
    def thread_write(i):
        with path_lock(path, shared=False) as fd:
            try:
                with open(fd, closefd=False) as fp:
                    fp.seek(0)
                    done = json.load(fp)
            except json.decoder.JSONDecodeError:
                done = []

            done.append(i)
            with open(fd, 'w', closefd=False) as fp:
                fp.seek(0)
                fp.truncate(0)
                json.dump(done, fp)

    with ThreadPoolExecutor(max_workers=len(indices)) as executor:
        executor.map(thread_write, indices)


def test_many_exclusive_threads_and_processes_rw(tmp_path):
    with lock(tmp_path) as path:

        m = 10
        n = m**2
        items = list(range(n))
        partition = list(
            map(
                lambda g: list(map(lambda t: t[1], g[1])),
                groupby(enumerate(items), lambda t: t[0] // m),
            )
        )

        assert len(partition) == m
        assert sorted(chain(*partition)) == items

        with Pool(processes=len(partition)) as pool:
            pool.starmap(
                many_exclusive_threads_and_processes_rw_process,
                map(lambda part: (path, part), partition),
            )

        with open(path) as fp:
            results = json.load(fp)

        assert sorted(results) == sorted(range(n))


def test_chained_shared_one_exclusive_threads_blocking(tmp_path):
    with lock(tmp_path) as path:

        n = 10

        first_is_locked = Barrier(2)

        def thread_lock_shared(recvp, send, recvn, results, i):
            if i > 0:
                j = recvp.get()  # Wait for previous thread to be locked
                assert j == i - 1
            with path_lock(path, shared=True):
                if i == 0:
                    first_is_locked.wait()  # Allow exclusive lock attempt
                results.append(i)
                send.put(i)  # Notify next thread that we are locked
                send.put(i)  # Notify previous thread that we are locked
                if i < n - 2:
                    j = recvn.get()  # Wait for next thread to be locked
                    assert j == i + 1

        def thread_lock_exclusive(results, i):
            first_is_locked.wait()
            with path_lock(path, shared=False):
                results.append(i)

        with ThreadPoolExecutor(max_workers=n) as executor:
            # NOTE We run the test twice to reuse workers to catch errors where
            # some workers are left in a locked state.
            for _ in range(2):
                first_is_locked.reset()
                results = []

                queues = [Queue() for _ in range(n + 1)]

                for i in range(n - 1):
                    executor.submit(
                        thread_lock_shared, queues[i], queues[i + 1], queues[i + 2], results, i
                    )
                last = executor.submit(thread_lock_exclusive, results, n - 1)

                last.result()

                assert results == sorted(range(n))


def process_lock_shared_chained(path, first_is_locked, recvp, send, recvn, results, n, i):
    if i > 0:
        j = recvp.get()  # Wait for previous thread to be locked
        assert j == i - 1
    with path_lock(path, shared=True):
        if i == 0:
            first_is_locked.wait()  # Allow exclusive lock attempt
        results.put(i)
        send.put(i)  # Notify next thread that we are locked
        send.put(i)  # Notify previous thread that we are locked
        if i < n - 2:
            j = recvn.get()  # Wait for next thread to be locked
            assert j == i + 1


@pytest.mark.skipif(os.name == 'nt', reason="Windows shared process locks are not implemented.")
def test_chained_shared_one_exclusive_processes_blocking(tmp_path):
    with lock(tmp_path) as path:

        n = 10

        with Pool(processes=n) as pool:

            m = Manager()
            results_queue = m.Queue()

            # NOTE We run the test twice to reuse workers to catch errors where
            # some workers are left in a locked state.
            for _ in range(2):

                first_is_locked = m.Barrier(2)
                queues = [m.Queue() for _ in range(n + 1)]

                for i in range(n - 1):
                    pool.apply_async(
                        process_lock_shared_chained,
                        [
                            path,
                            first_is_locked,
                            queues[i],
                            queues[i + 1],
                            queues[i + 2],
                            results_queue,
                            n,
                            i,
                        ],
                    )
                last = pool.apply_async(
                    process_lock_exclusive, [first_is_locked, results_queue, path, n - 1]
                )

                last.wait()

                results = []
                for i in range(n):
                    results.append(results_queue.get())

                assert results == sorted(range(n))


def test_synchronized_reads_threads_blocking(tmp_path):

    with lock(tmp_path) as path:

        filename = 'rw'

        with open(filename, 'w') as fp:
            json.dump({'id': -1, 'counter': 0, 'copy': 0}, fp)

        n = 1000
        k = 7
        p = 10
        max_write = p - k

        def sync_read(before, after, q, i):
            before.wait()
            time.sleep(0.01 * random())
            with path_lock(path, shared=True):
                after.wait()
                with open(filename) as fp:
                    contents = json.load(fp)
            q.put({'type': 'read', 'id': i, 'contents': contents})

        def write(q, i):
            with path_lock(path, shared=False):
                with open(filename) as fp:
                    contents = json.load(fp)

                contents['id'] = i
                contents['counter'] += 1

                with open(filename, 'w') as fp:
                    json.dump(contents, fp)

                with open(filename) as fp:
                    contents = json.load(fp)

                contents['copy'] += 1

                with open(filename, 'w') as fp:
                    json.dump(contents, fp)

            q.put({'type': 'write', 'id': i, 'contents': contents})

        with ThreadPoolExecutor(max_workers=p) as executor:
            q = Queue()
            i = 0
            g = 0
            w = 0
            messages = []
            group = {}
            while i < n:
                c = min(n - i, k)
                if w >= max_write or random() < 1 / (c + 1):
                    before = Barrier(c)
                    after = Barrier(c)
                    for _ in range(c):
                        executor.submit(sync_read, before, after, q, i)
                        group[i] = g
                        i += 1
                    g += 1
                else:
                    executor.submit(write, q, i)
                    i += 1
                    w += 1

                while (
                    not q.empty() or len(messages) < i - k
                ):  # NOTE We empty the queue as much as possible
                    messages.append(q.get())
                    if messages[-1]['type'] == 'write':
                        w -= 1

            while len(messages) < n:
                messages.append(q.get())

        values = {}
        for message in messages:
            t = message['type']
            if t == 'read':
                i = message['id']
                g = group[i]
                counter = message['contents']['counter']
                # NOTE counter is the same for the whole group
                assert values.setdefault(g, counter) == counter
                # NOTE copy is identical to counter
                assert message['contents']['copy'] == counter
            else:
                assert t == 'write'
                # NOTE check nobody else wrote to file while we held the lock
                assert message['id'] == message['contents']['id']
                assert message['contents']['copy'] == message['contents']['counter']


def process_sync_read(path, filename, before, after, q, i):
    before.wait()
    time.sleep(0.01 * random())
    with path_lock(path, shared=True):
        after.wait()
        with open(filename) as fp:
            contents = json.load(fp)
    q.put({'type': 'read', 'id': i, 'contents': contents})


def process_write(path, filename, q, i):
    with path_lock(path, shared=False):
        with open(filename) as fp:
            contents = json.load(fp)

        contents['id'] = i
        contents['counter'] += 1

        with open(filename, 'w') as fp:
            json.dump(contents, fp)

        with open(filename) as fp:
            contents = json.load(fp)

        contents['copy'] += 1

        with open(filename, 'w') as fp:
            json.dump(contents, fp)

    q.put({'type': 'write', 'id': i, 'contents': contents})


@pytest.mark.skipif(os.name == 'nt', reason="Windows shared process locks are not implemented.")
def test_synchronized_reads_processes_blocking(tmp_path):

    with lock(tmp_path) as path:

        filename = 'rw'

        with open(filename, 'w') as fp:
            json.dump({'id': -1, 'counter': 0, 'copy': 0}, fp)

        n = 1000
        k = 7
        p = 10
        max_write = p - k

        with Pool(processes=p) as pool:
            m = Manager()
            q = m.Queue()
            i = 0
            g = 0
            w = 0
            messages = []
            group = {}
            while i < n:
                c = min(n - i, k)
                if w >= max_write or random() < 1 / (c + 1):
                    before = m.Barrier(c)
                    after = m.Barrier(c)
                    for _ in range(c):
                        pool.apply_async(process_sync_read, [path, filename, before, after, q, i])
                        group[i] = g
                        i += 1
                    g += 1
                else:
                    pool.apply_async(process_write, [path, filename, q, i])
                    i += 1
                    w += 1

                while (
                    not q.empty() or len(messages) < i - k
                ):  # NOTE We empty the queue as much as possible
                    messages.append(q.get())
                    if messages[-1]['type'] == 'write':
                        w -= 1

            while len(messages) < n:
                messages.append(q.get())

        values = {}
        for message in messages:
            t = message['type']
            if t == 'read':
                i = message['id']
                g = group[i]
                counter = message['contents']['counter']
                # NOTE counter is the same for the whole group
                assert values.setdefault(g, counter) == counter
                # NOTE copy is identical to counter
                assert message['contents']['copy'] == counter
            else:
                assert t == 'write'
                # NOTE check nobody else wrote to file while we held the lock
                assert message['id'] == message['contents']['id']
                assert message['contents']['copy'] == message['contents']['counter']
