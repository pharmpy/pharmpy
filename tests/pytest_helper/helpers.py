
from collections import namedtuple
from enum import Enum
import os
from pathlib import Path
import sys



class PathType(Enum):
    DIR = 0,
    FILE = 1,


class PathManifest:
    """Keep manifests over files and dirs"""

    def __init__(self):
        self._paths = {k: set() for k in [PathType.DIR, PathType.FILE]}

    def add_path(self, path):
        """Add path to correct manifest (and asserts path existence)"""
        (res_path, res_type) = pathtype_resolve(path)
        self._add(res_type, path, check_exist=False)

    def add_file(self, path):
        """Add path to file manifest (and asserts path type + existence)"""
        self._add(PathType.FILE, path, check_exist=True)

    def add_dir(self, path):
        """Add path to dir manifest (and asserts path type + existence)"""
        self._add(PathType.DIR, path, check_exist=True)

    def has_path(self, path):
        """Checks if path in file/dir manifest (asserts path existence)"""
        (res_path, res_type) = pathtype_resolve(path)
        return str(res_path) in self._paths[res_type]

    def has_file(self, path):
        """Returns true if file manifest has path (NO path existence assert)"""
        return str(Path(path).resolve()) in self._paths[PathType.FILE]

    def has_dir(self, path):
        """Returns true if dir manifest has path (NO path existence assert)"""
        return str(Path(path).resolve()) in self._paths[PathType.DIR]

    def get_paths(self, is_type):
        """Get all paths if type 'is_type'"""
        return self._paths[is_type]

    def _add(self, is_type, path, check_exist):
        """Add path to 'is_type' manifest

        check_exist=False is dangerous (NO path resolve)"""
        if not check_exist:
            self._paths[is_type].add(str(path))
        else:
            res = pathtype_assert(is_type, path)
            self._paths[is_type].add(str(res))


def pathtype_assert(is_type, path):
    """Asserts path exists, is 'is_type' and returns: resolved path"""
    (res_path, res_type) = pathtype_resolve(path)
    assert res_type == is_type, \
        'path (%s) is %s but caller asserted %s' % (res_path, res_type, is_type)
    return res_path


def pathtype_resolve(path):
    """Asserts path exists and returns tuple: (resolved path, PathType)"""
    res_path = Path(path).resolve()
    assert res_path.exists(), 'path not found (when resolving type): %s' % (res_path,)
    if res_path.is_file():
        return (res_path, PathType.FILE)
    elif res_path.is_dir():
        return (res_path, PathType.DIR)
    else:
        raise AssertionError('resolved path not file or dir: %s' % str(res_path))


class _TestData:
    def __init__(self):
        self._rootpath = None
        self._manifest = PathManifest()

    @property
    def root(self):
        """Path to testdata root directory"""
        return self._rootpath

    @root.setter
    def root(self, path):
        self._init(path)

    def register(self, relpath, name=None):
        """Register file/dir (and assert its existence)"""
        assert self._rootpath, 'tried to add file without testdata.root set'
        (res_path, res_type) = pathtype_resolve(self._rootpath / relpath)
        if res_type == PathType.FILE:
            assert not self._manifest.has_file(res_path), \
                'tried to add file already in manifest: %s' % (res_path,)
            if not name:
                name = res_path.stem
        elif not name:
            name = res_path.name
        if hasattr(self, name):
            msg = 'autogen attr (%s) taken' % (name,)
            attr = getattr(self, name)
            if isinstance(attr, Path):
                rel = [self.root_relative(x) for x in [res_path, attr]]
                msg += ", path '%s' and '%s' seems to collide" % (rel[0], rel[1])
            raise AssertionError(msg + ': set explicit name!')
        setattr(self, name, res_path)
        self._manifest.add_path(res_path)

    @property
    def files(self):
        return self._manifest.get_paths(PathType.FILE)

    def root_relative(self, path):
        root, res = self._rootpath, Path(path).resolve()
        try:
            relpath = res.relative_to(root)
        except ValueError:
            raise AssertionError('path not in root (%s): %s' % (root, res))
        return relpath

    def _init(self, rootpath):
        assert self._rootpath is None, \
            'testdata.root already set' % (name, res_path,)
        self._rootpath = pathtype_assert(PathType.DIR, rootpath)

    def assert_clean(self):
        """Assert testdata root be clean with respect to registered paths"""
        root = self._rootpath
        for root, dirs, files in os.walk(str(self.root)):
            rpath = Path(root)
            for path in dirs + files:
                if not self._manifest.has_path(rpath / path):
                    rel = self.root_relative(rpath / path)
                    raise AssertionError(
                        "root (%s) not clean ('%s' not in manifest)" % (root, rel,)
                    )

testdata = _TestData()
