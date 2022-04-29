"""The lock module exposes a path_lock context manager function

Other exported functions are implementation details subject to change.

Caveats:
    If you need to lock multiple paths you should have a total order for each
    subset of paths that could be locked simultaneously by the same thread and call
    path_lock multiple times respecting this total order. For instance via
    resolved absolute path lexicographical order.

    Currently, on Windows, at the process level, shared locks are "simulated" by
    exclusive locks. Implementing true shared locks on this platform would require
    depending on userland libraries. This would be necessary, for instance, in
    situations where multiple processes need to lock shared resources and
    communicate simultaneously about the state of those resources. Note that the
    problem in this particular example can be circumvented at the cost of
    efficiency, by simulating locked shared resources through multiple rounds of
    exclusively-locked resources plus communication. I do not think our current
    usage mandates the correct implementation of shared locks. It is enough that
    the implementation is simple for Windows, and is correct on UNIX.

    Currently, on Windows, at the process level, we attempt to lock the entire file
    by passing the largest possible value to `mscvrt.locking` `nbytes` argument. If
    we want this implementation to be correct, we would need to `fo.seek(0)` before
    calling `mscvrt.locking` then perhaps restore the seek position? Also the
    largest possible value is 2^31-1 which might not be enough for files larger
    than 2GB. Again, we do not currently need this "functionality".

    It is currently not possible to specify a timeout for acquiring a lock. The
    only options currently are immediate failure or forever blocking.

    The thread-level lock is currently hard-coded as non-reentrant, but we
    could want the possibility of choice. It will require some work to figure
    out how to combine reentrant thread-level locks with process-level locks.
    For instance, Linux will happily "upgrade" an acquired exclusive lock to a
    shared lock if the same fd is lockf-ed multiple times with different
    arguments from the same process. This is not something that we want to
    happen.
"""
import os
from contextlib import contextmanager
from threading import Condition, Lock

if os.name == 'nt':
    # Windows file locking
    import msvcrt
    import sys

    # NOTE lock the entire file
    _lock_length = -1 if sys.version_info.major == 2 else int(2**31 - 1)

    def _is_process_level_lock_timeout_error(error: OSError) -> bool:
        """Check if an OSError corresponds to a blocking lock timeout error

        Note that the errno for a non-blocking lock failure would be 13 with a
        strerror of "Permission denied".
        """
        return error.errno == 36 and error.strerror == 'Resource deadlock avoided'

    def _process_level_lock(fd: int, shared: bool = False, blocking: bool = True):
        # NOTE Simulates shared lock using an exclusive lock
        if blocking:
            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_LOCK, _lock_length)
                    break
                except OSError as error:
                    if not _is_process_level_lock_timeout_error(error):
                        raise error
        else:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, _lock_length)

    def _process_level_unlock(fd: int):
        msvcrt.locking(fd, msvcrt.LK_UNLCK, _lock_length)

else:
    # UNIX based file locking
    import fcntl

    def _process_level_lock(fd: int, shared: bool = False, blocking: bool = True):
        operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        if not blocking:
            operation |= fcntl.LOCK_NB
        fcntl.lockf(fd, operation)

    def _process_level_unlock(fd: int):
        fcntl.lockf(fd, fcntl.LOCK_UN)


@contextmanager
def process_level_lock(fd: int, shared: bool = False, blocking: bool = True):
    _process_level_lock(fd, shared, blocking)

    try:
        yield

    finally:
        _process_level_unlock(fd)


class ShareableThreadLock:
    def __init__(self):
        self._condition = Condition(Lock())
        self._shared_count = 0

    def lock(self, shared: bool = False, blocking: bool = True):
        if shared:
            return self._lock_sh(blocking=blocking)
        elif blocking:
            return self._lock_ex()
        else:
            return self._lock_ex_nb()

    @contextmanager
    def _lock_sh(self, blocking: bool = True):
        self._condition.acquire(blocking=blocking)
        try:
            self._shared_count += 1
        finally:
            self._condition.release()

        try:
            yield

        finally:

            self._condition.acquire()
            try:
                self._shared_count -= 1
                if self._shared_count == 0:
                    self._condition.notifyAll()
            finally:
                self._condition.release()

    @contextmanager
    def _lock_ex(self):
        self._condition.acquire(blocking=True)
        try:
            while self._shared_count != 0:
                self._condition.wait()

            yield

        finally:
            self._condition.release()

    @contextmanager
    def _lock_ex_nb(self):
        self._condition.acquire(blocking=False)
        try:
            if self._shared_count != 0:
                raise Exception('Could not acquire lock')

            yield

        finally:
            self._condition.release()


class ThreadSafeKeyedRefPool:
    def __init__(self, lock, refs: dict, factory):
        self._lock = lock
        self._refs = refs
        self._factory = factory

    @contextmanager
    def __call__(self, key: str):
        with self._lock:
            entry = self._refs.get(key)
            if entry is None:
                # NOTE We create a new object if none exists
                obj = self._factory()
                self._refs[key] = (obj, 1)
            else:
                # NOTE Otherwise we count one ref more
                (obj, refcount) = entry
                self._refs[key] = (obj, refcount + 1)

        try:
            yield obj

        finally:
            with self._lock:
                (_, refcount) = self._refs[key]
                if refcount == 1:
                    # NOTE We remove the object from the pool since nobody
                    # else holds a reference to it.
                    del self._refs[key]
                else:
                    # NOTE Otherwise we count one ref less
                    self._refs[key] = (obj, refcount - 1)


_thread_level_lock_ref = ThreadSafeKeyedRefPool(Lock(), {}, ShareableThreadLock)


@contextmanager
def thread_level_path_lock(path: str, shared: bool = False, blocking: bool = True):
    key = os.path.normpath(path)
    with _thread_level_lock_ref(key) as ref:
        with ref.lock(shared, blocking):
            yield


@contextmanager
def process_level_path_lock(path: str, shared: bool = False, blocking: bool = True):
    fd = os.open(path, os.O_RDONLY if shared else os.O_RDWR)
    try:
        with process_level_lock(fd, shared, blocking):
            yield fd

    finally:
        os.close(fd)


@contextmanager
def path_lock(path: str, shared: bool = False, blocking: bool = True):
    """Locks a path both at the thread-level and process-level.

    Parameters
    ----------
    path : str
        The path to lock. This path will be open in read-only mode if shared is
        True, write mode otherwise. Must be the path of an existing file (not a
        directory).
    shared : bool
        Whether the lock should be shared. If not shared, the lock is
        exclusive. One can have either a single exclusive lock or any number of
        shared lock on the same resource at any given time.
    blocking : bool
        Whether lock acquisition should be blocking. If True, will raise an
        error if a lock cannot be acquired immediately. Otherwise, will block
        until the lock can be acquired.
    """
    with thread_level_path_lock(path, shared, blocking):
        with process_level_path_lock(path, shared, blocking):
            yield
