import logging
import os
import re
from glob import glob
from pathlib import Path

from pharmpy.execute import Engine

# Warning: This is experimental and perhaps the architecture needs to change


class NONMEM7(Engine):
    """NONMEM7 execution engine.

    An implementation of the generic :class:`~pharmpy.execute.engine.Engine` (see for more
    information).

    Will automatically scan for installed NONMEM versions (see :func:`~self.scan_installed`).
    """

    def __init__(self, model, version=None):
        log = logging.getLogger(__name__)

        self.installed = self.scan_installed()
        if not version and self.installed:
            version = max(self.installed.keys())
        if version not in self.installed:
            log.error('Requested NONMEM7 version %r not available (found: %s)', version,
                      ', '.join(self.installed.values()))
            self.info = None
        else:
            self.info = (version, *self.installed[version])

        super().__init__(model)

    def get_commandline(self, task):
        if task in {'evaluate', 'estimate'}:
            mod = str(self.model.path)
            lst = re.sub(r'\.(mod|ctl)$', '.lst', mod)
            if mod == lst:
                lst += '.lst'
            return (self.bin, mod, lst)
        else:
            raise ValueError('Engine (%s) does not support task %r.' % (self.__class__.__name__,
                                                                        task))

    @property
    def bin(self):
        return self.info[2] if self.info else None

    @property
    def version(self):
        return self.info[0] if self.info else None

    def __bool__(self):
        return bool(self.installed)

    @classmethod
    def scan_installed(cls):
        log = logging.getLogger(__name__)
        log.info("Initiating NM version scan in filesystem")

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
            if not any((path/subdir).is_dir() for subdir in subdirs):
                continue

            ver_suffix = {7.1: '', 7.2: '2', 7.3: '3', 7.4: '4', 7.5: '5'}
            for version, suffix in ver_suffix.items():
                nm_bin = [path/subdir/(script % suffix) for subdir in subdirs]
                nm_bin = list(filter(lambda x: x.is_file(), nm_bin))
                if nm_bin:
                    nm_info[version] = (nm_dir, str(nm_bin[0]))
                    break

        return nm_info
