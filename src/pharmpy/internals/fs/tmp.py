import os as _os
import shutil as _shutil
import warnings as _warnings
import weakref as _weakref
from tempfile import mkdtemp


# NOTE This is copied from the Python 3.10 implementation to expose the
# ignore_cleanup_errors flag and patch PermissionError handling for Windows
# lock files. Here is the original source link:
#
# https://github.com/python/cpython/blob/b494f5935c92951e75597bfe1c8b1f3112fec270/Lib/tempfile.py#L778-L848
#
# The whole thing can be replaced by `from tempfile import TemporaryDirectory`
# once Python fixes their implementation AND once we drop support for Python
# Python 3.9.
class TemporaryDirectory:
    """Create and return a temporary directory.  This has the same
    behavior as mkdtemp but can be used as a context manager.  For
    example:

        with TemporaryDirectory() as tmpdir:
            ...

    Upon exiting the context, the directory and everything contained
    in it are removed.
    """

    def __init__(self, suffix=None, prefix=None, dir=None, ignore_cleanup_errors=False):
        self.name = mkdtemp(suffix, prefix, dir)
        self._ignore_cleanup_errors = ignore_cleanup_errors
        self._finalizer = _weakref.finalize(
            self,
            self._cleanup,
            self.name,
            warn_message="Implicitly cleaning up {!r}".format(self),
            ignore_errors=self._ignore_cleanup_errors,
        )

    @classmethod
    def _rmtree(cls, name, ignore_errors=False):
        def onerror(func, path, exc_info):
            if issubclass(exc_info[0], PermissionError):

                def resetperms(path):
                    try:
                        _os.chflags(path, 0)
                    except AttributeError:
                        pass
                    _os.chmod(path, 0o700)

                try:
                    if path != name:
                        resetperms(_os.path.dirname(path))
                    resetperms(path)

                    try:
                        _os.unlink(path)
                    except IsADirectoryError:
                        cls._rmtree(path, ignore_errors=ignore_errors)
                    except PermissionError as pe:
                        # PermissionError is raised on FreeBSD for directories
                        # and by Windows on lock files used by other processes
                        try:
                            cls._rmtree(path, ignore_errors=ignore_errors)
                        except NotADirectoryError:
                            # NOTE This is raised if PermissionError did not
                            # correspond to a IsADirectoryError, e.g. on
                            # Windows.
                            if not ignore_errors:
                                raise pe
                except FileNotFoundError:
                    pass
            elif issubclass(exc_info[0], FileNotFoundError):
                pass
            else:
                if not ignore_errors:
                    raise

        _shutil.rmtree(name, onerror=onerror)

    @classmethod
    def _cleanup(cls, name, warn_message, ignore_errors=False):
        cls._rmtree(name, ignore_errors=ignore_errors)
        _warnings.warn(warn_message, ResourceWarning)

    def __repr__(self):
        return "<{} {!r}>".format(self.__class__.__name__, self.name)

    def __enter__(self):
        return self.name

    def __exit__(self, exc, value, tb):
        self.cleanup()

    def cleanup(self):
        if self._finalizer.detach() or _os.path.exists(self.name):
            self._rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)
