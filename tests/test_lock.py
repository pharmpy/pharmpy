import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from multiprocessing import Manager, Pipe, Pool, Process
from pathlib import Path
from queue import Queue

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

        first_is_locked = Queue()
        is_done = Queue()
        last_attempt_locking = Queue()

        def thread_lock_first():
            with path_lock(path, shared=False, blocking=False):
                first_is_locked.put(0)
                is_done.get()

        def thread_lock_last():
            last_attempt_locking.get()
            with path_lock(path, shared=False, blocking=False):
                pass

        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(thread_lock_first)
            last = executor.submit(thread_lock_last)

            first_is_locked.get()
            last_attempt_locking.put(0)

            with pytest.raises(AcquiringThreadLevelLockWouldBlockError):
                last.result()

            is_done.put(0)


def process_lock_first(conn, path):
    with path_lock(path, shared=False, blocking=False):
        conn.send('')
        conn.recv()


def process_lock_last(conn, path):
    conn.recv()
    try:
        with path_lock(path, shared=False, blocking=False):
            pass
        conn.send(None)
    except Exception as e:
        conn.send(e)


def test_exclusive_processes_non_blocking(tmp_path):
    with lock(tmp_path) as path:

        parent_first_conn, first_conn = Pipe()
        parent_last_conn, last_conn = Pipe()
        Process(target=process_lock_first, args=(first_conn, path)).start()
        last = Process(target=process_lock_last, args=(last_conn, path))
        last.start()

        parent_first_conn.recv()
        parent_last_conn.send('')
        e = parent_last_conn.recv()

        assert isinstance(e, AcquiringProcessLevelLockWouldBlockError)

        parent_first_conn.send('')


def test_many_shared_one_exclusive_threads_blocking(tmp_path):
    with lock(tmp_path) as path:

        n = 10

        q = Queue()

        def thread_lock_shared(results, i):
            with path_lock(path, shared=True):
                q.put(0)
                time.sleep(1)
                results.append(i)

        def thread_lock_exclusive(results, i):
            for j in range(n - 1):
                q.get()
            with path_lock(path, shared=False):
                results.append(i)

        with ThreadPoolExecutor(max_workers=n) as executor:
            for j in range(2):
                results = []

                for i in range(1, n):
                    executor.submit(thread_lock_shared, results, i)
                last = executor.submit(thread_lock_exclusive, results, 0)

                last.result()

                assert sorted(results) == sorted(range(n))
                assert results[-1] == 0


def process_lock_shared(conn, q, path, i):
    with path_lock(path, shared=True):
        conn.put(0)
        time.sleep(2)
        q.put(i)


def process_lock_exclusive(conn, q, path, i):
    conn.recv()
    with path_lock(path, shared=False):
        q.put(i)


@pytest.mark.skipif(os.name == 'nt', reason="Windows shared process locks are not implemented.")
def test_many_shared_one_exclusive_processes_blocking(tmp_path):
    with lock(tmp_path) as path:

        n = 10

        with Pool(processes=n) as pool:

            m = Manager()
            results_queue = m.Queue()

            for j in range(2):

                locked_queue = m.Queue()
                parent_last_conn, last_conn = Pipe()

                for i in range(1, n):
                    pool.apply_async(process_lock_shared, [locked_queue, results_queue, path, i])
                last = pool.apply_async(process_lock_exclusive, [last_conn, results_queue, path, 0])

                # Waiting for at least one process to acquire a shared lock
                locked_queue.get()

                # Allow last process to attempt acquiring an exlusive lock
                parent_last_conn.send('')

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
