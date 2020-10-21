import os
from glob import glob
from pathlib import Path

from pharmpy.workflows import ModelExecutionEngine


class NONMEM7(ModelExecutionEngine):
    """NONMEM7 execution engine.

    An implementation of the generic :class:`~pharmpy.workflows.ModelExecutionEngine` (see for more
    information).

    Will automatically scan for installed NONMEM versions (see :func:`~self.scan_installed`).
    """

    def __init__(self, model, version=None):
        self.installed = self.scan_installed()
        if not version and self.installed:
            version = max(self.installed.keys())
        if version not in self.installed:
            raise RuntimeError(
                'Requested NONMEM7 version %r not available (found: %s)',
                version,
                ', '.join(self.installed.values()),
            )
        else:
            self.info = (version, *self.installed[version])

    def commandline(self, model):
        mod = str(self.model.source.path)
        lst = mod.with_suffix('.lst')
        if mod == lst:
            lst += '.lst'
        return (self.bin, mod, lst)

    @property
    def bin(self):
        return self.info[2] if self.info else None

    @property
    def version(self):
        return self.info[0] if self.info else None

    @staticmethod
    def scan_installed():
        if os.name == 'nt':
            globpaths = ['C:/nm7*', 'C:/nm_7*', 'C:/NONMEM/nm7*', 'C:/NONMEM/nm_7*']
            globpaths = [str(Path(globpath).resolve().expanduser()) for globpath in globpaths]
            script = 'nmfe7%s.bat'
        else:
            globpaths = []
            for root in [Path(x) for x in ('/opt', '~')]:
                tails = ['nm7*', 'nm_7*', 'nm7*', 'nm_7*', 'nonmem/nm7*', 'nonmem/nm_7*']
                globpaths += [root / x for x in tails]
            globpaths = [str(Path(globpath).expanduser()) for globpath in globpaths]
            script = 'nmfe7%s'
        candidates = [path for globpath in globpaths for path in glob(globpath)]

        nm_info = dict()
        for nm_dir in candidates:
            path = Path(nm_dir)

            if not path.is_dir():
                continue
            subdirs = ['run', 'util']
            if not any((path / subdir).is_dir() for subdir in subdirs):
                continue

            ver_suffix = {7.1: '', 7.2: '2', 7.3: '3', 7.4: '4', 7.5: '5'}
            for version, suffix in ver_suffix.items():
                nm_bin = [path / subdir / (script % suffix) for subdir in subdirs]
                nm_bin = list(filter(lambda x: x.is_file(), nm_bin))
                if nm_bin:
                    nm_info[version] = (nm_dir, str(nm_bin[0]))
                    break

        return nm_info
