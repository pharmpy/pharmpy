# -*- encoding: utf-8 -*-

import logging
import re
import os
from glob import glob
from pathlib import Path

from pysn.execute import Engine


class NONMEM7(Engine):
    """NONMEM7 implementation of generic engine.

    Scans for installed NM version on initialization."""

    def __init__(self, version=None):
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

        super().__init__(envir=None)

    def create_command(self, model):
        mod = str(model.path)
        lst = re.sub(r'\.(mod|ctl)$', '.lst', mod)
        if mod == lst:
            lst += '.lst'
        return (self.bin, mod, lst)

    @property
    def bin(self):
        """Path to main binary."""
        return self.info[2] if self.info else None

    @property
    def version(self):
        """Version (of main binary)."""
        return self.info[0] if self.info else None

    def __bool__(self):
        """Should only eval True if NONMEM7 is capable of estimation at any time."""
        return bool(self.installed)

    @classmethod
    def scan_installed(cls):
        log = logging.getLogger(__name__)
        log.info("Initiating NM version scan in filesystem")

        if os.name == 'nt':
            globs = ['C:/nm7*', 'C:/nm_7*', 'C:/NONMEM/nm7*', 'C:/NONMEM/nm_7*']
            globs = [str(Path(globpath).resolve().expanduser()) for globpath in globs]
            script = 'nmfe7%s.bat'
        else:
            globs = []
            for root in [Path(x) for x in ('/opt', '~')]:
                tails = ['nm7*', 'nm_7*', 'nm7*', 'nm_7*', 'nonmem/nm7*', 'nonmem/nm_7*']
                globs += [root / x for x in tails]
            globs = [str(Path(globpath).expanduser().resolve()) for globpath in globs]
            script = 'nmfe7%s'
        candidates = [path for globpattern in globs for path in glob(globpattern)]

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
