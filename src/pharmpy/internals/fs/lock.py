"""The lock module exposes a path_lock context manager function

Exception classes are part of the public API.
Other exported functions are implementation details subject to change.

Caveats:
    If you need to lock multiple paths you should have a total order for each
    subset of paths that could be locked simultaneously by the same thread and call
    path_lock multiple times respecting this total order. For instance via
    resolved absolute path lexicographical order.

    The Linux implementation relies on fcntl/lockf which is infamous for being
    hard to work with: if you close any file descriptor for a given file in a
    given process, it will release the lock held by that process, even if the
    lock was acquired through a different file descriptor. So you cannot easily
    open a lock file for reading or writing without breaking the locking
    mechanism. We implement locks in a way that uses a single fd while a lock
    exists for a given (normalized) path. That fd is opened with O_RDWR and is
    yielded by path_lock for convenience, but the user should take extra care
    not to close this fd, as it would release the lock. For instance, `open`
    must be used with the `closefd` flag as in `open(fd, closefd=False)`. This
    makes writing to the lock file, or reading several times from the lock
    file, a bit challenging but not impossible (using `fp.seek` and
    `fp.truncate`).

    Another solution would be to use flock but since that can sometimes
    fallback to the fcntl/lockf implementation, we prefer to use the latter
    implementation directly, and workaround the limitations. See
    http://0pointer.de/blog/projects/locking.html.

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

    The current implementation may not even be correct on UNIX in the scenario
    discussed above: in a multi-process, multi-thread setting, if one thread of
    process A acquires an exlusive lock at the thread-level first, then
    attempts to exclusively acquire the process-level lock, that process-level
    lock may already be acquired as shared by a thread of another process B
    that is waiting on the other threads of A to run. And vice versa, if one
    thread of process A acquires an exclusive lock at the process-level first,
    then attempts to exclusively acquire the thread-level lock, that
    thread-level lock may already be acquired as shared by other threads of
    process A, but they are waiting on a thread of process B that is in turn
    waiting for the process-level lock to be downgraded to shared. So it seems
    that composing two isolated implementations, one for threads, one for
    processes, cannot work in these "share and communicate" scenarios. One
    solution is to forbid upgrade and downgrade of a process-level lock which
    would effectively be the same as having an all-or-nothing lock (either you
    lock at both levels or you do not).

    Currently, on Windows, at the process level, we attempt to lock the entire file
    by passing the largest possible value to `mscvrt.locking` `nbytes` argument. If
    we want this implementation to be correct, we would need to `fo.seek(0)` before
    calling `mscvrt.locking` then perhaps restore the seek position? Also the
    largest possible value is 2^31-1 which might not be enough for files larger
    than 2GB. Again, we do not currently need this "functionality".

    It is currently not possible to specify a timeout for acquiring a lock. The
    only options currently are immediate failure or forever blocking.
"""
import os
import sys
from collections import Counter
from contextlib import contextmanager
from threading import Condition, Lock, RLock, get_ident
from typing import Generic, TypeVar

T = TypeVar('T')


class AcquiringLockWouldBlockError(Exception):
    """Raised when acquiring lock would block and blocking is False"""


class AcquiringProcessLevelLockWouldBlockError(AcquiringLockWouldBlockError):
    """Raised when acquiring a process-level lock would block and blocking is False"""


class AcquiringThreadLevelLockWouldBlockError(AcquiringLockWouldBlockError):
    """Raised when acquiring a thread-level lock would block and blocking is False"""


class RecursiveDeadlockError(Exception):
    """Raised when recursive dead-lock is detected."""


is_windows = os.name == 'nt'
is_mac_os = sys.platform == 'darwin'

if is_windows:
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

    def _is_process_level_lock_blocking_error(error: OSError) -> bool:
        """Check that an OSError corresponds to an error raised because
        a process-level lock cannot be acquired immediately
        """
        return (
            isinstance(error, PermissionError)
            and error.errno == 13
            and error.strerror == 'Permission denied'
        )

    def _process_level_lock(fd: int, shared: bool = False, blocking: bool = True):
        # NOTE Simulates shared lock using an exclusive lock. This
        # implementation does not allow to lock the same fd multiple times.
        # This does not matter a we make sure we do not do that.
        if blocking:
            while True:
                try:
                    msvcrt.locking(  # pyright: ignore [reportGeneralTypeIssues]
                        fd,
                        msvcrt.LK_LOCK,  # pyright: ignore [reportGeneralTypeIssues]
                        _lock_length,
                    )
                    break
                except OSError as error:
                    if not _is_process_level_lock_timeout_error(error):
                        raise error
        else:
            try:
                msvcrt.locking(  # pyright: ignore [reportGeneralTypeIssues]
                    fd, msvcrt.LK_NBLCK, _lock_length  # pyright: ignore [reportGeneralTypeIssues]
                )
            except PermissionError as error:
                if _is_process_level_lock_blocking_error(error):
                    raise AcquiringProcessLevelLockWouldBlockError()
                else:
                    raise error

    def _process_level_unlock(fd: int):
        # NOTE This implementation (Windows) will raise an error if attempting
        # to unlock an already unlocked fd. This does not matter as we make
        # sure we do not do that.
        msvcrt.locking(  # pyright: ignore [reportGeneralTypeIssues]
            fd, msvcrt.LK_UNLCK, _lock_length  # pyright: ignore [reportGeneralTypeIssues]
        )

else:
    # UNIX based file locking
    import fcntl

    def _is_process_level_lock_blocking_error(error: OSError) -> bool:
        """Check that an OSError corresponds to an error raised because
        a process-level lock cannot be acquired immediately
        """
        return (
            isinstance(error, BlockingIOError)
            and (
                (is_mac_os and error.errno == 35)  # MacOS
                or (not (is_mac_os) and error.errno == 11)  # Linux
            )
            and error.strerror == 'Resource temporarily unavailable'
        )

    def _process_level_lock(fd: int, shared: bool = False, blocking: bool = True):
        operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        if blocking:
            fcntl.lockf(fd, operation)
        else:
            try:
                fcntl.lockf(fd, operation | fcntl.LOCK_NB)
            except BlockingIOError as error:
                if _is_process_level_lock_blocking_error(error):
                    raise AcquiringProcessLevelLockWouldBlockError()
                else:
                    raise error

    def _process_level_unlock(fd: int):
        # NOTE This implementation (UNIX) will NOT raise an error if attempting
        # to unlock an already unlocked fd. This does not matter as we make
        # sure we do not do that.
        fcntl.lockf(fd, fcntl.LOCK_UN)


class ShareableProcessLock:
    """Creates a process lock for a FD shared by all threads of a process"""

    def __init__(self, fd: int):
        self._fd = fd
        self._lock = Lock()
        self._shared_by: Counter[int] = Counter()
        self._exclusively_held_by: Counter[int] = Counter()

    @contextmanager
    def lock(self, shared: bool = False, blocking: bool = True, reentrant: bool = False):
        """Locks the scoped FD

        shared : bool
            Whether the lock should be shared. If not shared, the lock is
            exclusive. Exclusively locking a file only means that we want to
            exclude other processes from locking the same file: any number of
            threads of the same process can lock the same file both shared and
            exclusively simultaneously. We keep track of which threads are
            holding what kind of locks to upgrade the process lock from shared
            to exclusive and vice versa.
        blocking : bool
            Whether lock acquisition should be blocking. If True, will raise an
            error if a lock cannot be acquired immediately. Otherwise, will block
            until the lock can be acquired.
        reentrant : bool
            Whether lock acquisition is reentrant. If True, allows each thread
            to lock the file recursively. Otherwise, the same thread
            recursively locking dead-locks.
        """
        thread_id = get_ident()
        if self._lock.acquire(blocking=blocking):
            try:
                if not reentrant and (
                    self._shared_by[thread_id] or self._exclusively_held_by[thread_id]
                ):
                    raise RecursiveDeadlockError()

                is_held_shared = bool(self._shared_by)
                is_held_exclusively = bool(self._exclusively_held_by)
                is_held = is_held_shared or is_held_exclusively

                if not is_held or (is_held_shared and not shared and not is_windows):
                    # NOTE We only lock when the first locking attempt is made, or
                    # if we want to upgrade the lock to an exclusive lock, but only on
                    # UNIX since current implementation always uses exclusive locks on
                    # Windows.
                    _process_level_lock(self._fd, shared, blocking)

                if shared:
                    self._shared_by[thread_id] += 1
                else:
                    self._exclusively_held_by[thread_id] += 1
            finally:
                self._lock.release()
        else:
            raise AcquiringProcessLevelLockWouldBlockError()

        try:
            yield

        finally:
            with self._lock:
                if shared:
                    self._shared_by[thread_id] -= 1
                    if not self._shared_by[thread_id]:
                        del self._shared_by[thread_id]
                else:
                    self._exclusively_held_by[thread_id] -= 1
                    if not self._exclusively_held_by[thread_id]:
                        del self._exclusively_held_by[thread_id]

                is_held_shared = bool(self._shared_by)
                is_held_exclusively = bool(self._exclusively_held_by)
                is_held = is_held_shared or is_held_exclusively

                if not is_held:
                    # NOTE We only release the lock once we have exhausted all
                    # locking attempts
                    _process_level_unlock(self._fd)
                elif not is_held_exclusively and not shared and not is_windows:
                    # NOTE We downgrade the lock from exclusive to shared if we
                    # are not on Windows and we just released an exclusive
                    # lock, and no exclusive lock is left.
                    _process_level_lock(self._fd, shared=True, blocking=True)


class ShareableThreadLock:
    def __init__(self):
        self._condition = Condition(RLock())
        self._acquired_by: Counter[int] = Counter()

    def lock(self, shared: bool = False, blocking: bool = True, reentrant: bool = False):
        if shared:
            return self._lock_sh(blocking=blocking, reentrant=reentrant)
        else:
            return self._lock_ex(blocking=blocking, reentrant=reentrant)

    @contextmanager
    def _lock_sh(self, blocking: bool = True, reentrant: bool = False):
        thread_id = get_ident()
        if self._condition.acquire(blocking=blocking):
            try:
                if not reentrant and self._acquired_by[thread_id]:
                    raise RecursiveDeadlockError()
                self._acquired_by[thread_id] += 1
            finally:
                self._condition.release()
        else:
            raise AcquiringThreadLevelLockWouldBlockError()

        try:
            yield

        finally:
            self._condition.acquire(blocking=True)
            try:
                self._acquired_by[thread_id] -= 1
                if not self._acquired_by[thread_id]:
                    del self._acquired_by[thread_id]  # NOTE GC
                    if not self._acquired_by:
                        self._condition.notify_all()
            finally:
                self._condition.release()

    @contextmanager
    def _lock_ex(self, blocking: bool = True, reentrant: bool = False):
        thread_id = get_ident()
        if self._condition.acquire(blocking=blocking):
            acquired = False
            try:
                this_thread_count = Counter({thread_id: self._acquired_by[thread_id]})
                if blocking:
                    # NOTE We could use self._acquired_by != this_thread_count in
                    # Python >= 3.10
                    while self._acquired_by - this_thread_count:
                        self._condition.wait()
                else:
                    # NOTE We could use self._acquired_by != this_thread_count in
                    # Python >= 3.10
                    if self._acquired_by - this_thread_count:
                        raise AcquiringThreadLevelLockWouldBlockError()

                # NOTE The following check also works for Python < 3.10 because we
                # systematically get rid of zero-count entries.
                if self._acquired_by:
                    # NOTE The following two checks can be replaced by
                    # assert self._acquired_by == this_thread_count
                    # in Python >= 3.10
                    assert len(self._acquired_by) == 1
                    assert self._acquired_by[thread_id] == this_thread_count[thread_id]
                    if not reentrant:
                        raise RecursiveDeadlockError()

                acquired = True
                self._acquired_by[thread_id] += 1

                yield

            finally:
                if acquired:
                    self._acquired_by[thread_id] -= 1
                    if not self._acquired_by[thread_id]:
                        del self._acquired_by[thread_id]  # NOTE GC
                self._condition.release()
        else:
            raise AcquiringThreadLevelLockWouldBlockError()


class ThreadSafeKeyedRefPool(Generic[T]):
    def __init__(self, lock, refs: dict, factory, destructor=None):
        self._lock = lock
        self._refs = refs
        self._factory = factory
        self._destructor = destructor

    @contextmanager
    def __call__(self, key: T):
        with self._lock:
            entry = self._refs.get(key)
            if entry is None:
                # NOTE We create a new object if none exists
                obj = self._factory(key)
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
                    if self._destructor is not None:
                        self._destructor(obj)
                else:
                    # NOTE Otherwise we count one ref less
                    self._refs[key] = (obj, refcount - 1)


_thread_level_lock_ref = ThreadSafeKeyedRefPool(Lock(), {}, lambda _key: ShareableThreadLock())
_process_level_lock_ref = ThreadSafeKeyedRefPool(Lock(), {}, ShareableProcessLock)
_fd_ref = ThreadSafeKeyedRefPool(
    Lock(), {}, lambda normalized_path: os.open(normalized_path, os.O_RDWR), lambda fd: os.close(fd)
)


@contextmanager
def process_level_lock(
    fd: int, shared: bool = False, blocking: bool = True, reentrant: bool = False
):
    with _process_level_lock_ref(fd) as ref:
        with ref.lock(shared, blocking, reentrant):
            yield


@contextmanager
def thread_level_lock(
    key: str, shared: bool = False, blocking: bool = True, reentrant: bool = False
):
    with _thread_level_lock_ref(key) as ref:
        with ref.lock(shared, blocking, reentrant):
            yield


@contextmanager
def process_level_path_lock(
    normalized_path: str, shared: bool = False, blocking: bool = True, reentrant: bool = False
):
    with _fd_ref(normalized_path) as fd:
        with process_level_lock(fd, shared, blocking, reentrant):
            yield fd


@contextmanager
def path_lock(path: str, shared: bool = False, blocking: bool = True, reentrant: bool = False):
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
    reentrant : bool
        Whether lock acquisition is reentrant. If True, allows to lock
        recursively. Otherwise, recursively locking dead-locks.
    """
    key = os.path.normpath(path)
    with thread_level_lock(key, shared, blocking, reentrant):
        with process_level_path_lock(key, shared, blocking, reentrant) as fd:
            yield fd
