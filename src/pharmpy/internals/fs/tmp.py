import os
import shutil
import time
import warnings
import weakref
from tempfile import mkdtemp

if os.name == 'nt':
    # This is copied and modified from the python 3.9 implementation
    # The aim is to be able to handle Permission issues in Windows
    class TemporaryDirectory:
        """Create and return a temporary directory.  This has the same
        behavior as mkdtemp but can be used as a context manager.  For
        example:

            with TemporaryDirectory() as tmpdir:
                ...

        Upon exiting the context, the directory and everything contained
        in it are removed.
        """

        def __init__(self, suffix=None, prefix=None, dir=None):
            self.name = mkdtemp(suffix, prefix, dir)
            self._finalizer = weakref.finalize(
                self,
                self._cleanup,
                self.name,
                warn_message="Implicitly cleaning up {!r}".format(self),
            )

        @classmethod
        def _rmtree(cls, name):
            def onerror(func, path, exc_info):
                if issubclass(exc_info[0], PermissionError):

                    def resetperms(path):
                        try:
                            os.chflags(path, 0)
                        except AttributeError:
                            pass
                        os.chmod(path, 0o700)

                    try:
                        if path != name:
                            resetperms(os.path.dirname(path))
                        resetperms(path)

                        try:
                            os.unlink(path)
                        # PermissionError is raised on FreeBSD for directories
                        except (IsADirectoryError, PermissionError):
                            time.sleep(0.1)
                            cls._rmtree(path)
                    except FileNotFoundError:
                        pass
                elif issubclass(exc_info[0], FileNotFoundError):
                    pass
                else:
                    raise

            shutil.rmtree(name, onerror=onerror)

        @classmethod
        def _cleanup(cls, name, warn_message):
            cls._rmtree(name)
            warnings.warn(warn_message, ResourceWarning)

        def __repr__(self):
            return "<{} {!r}>".format(self.__class__.__name__, self.name)

        def __enter__(self):
            return self.name

        def __exit__(self, exc, value, tb):
            self.cleanup()

        def cleanup(self):
            if self._finalizer.detach():
                self._rmtree(self.name)

else:
    from tempfile import TemporaryDirectory  # noqa
