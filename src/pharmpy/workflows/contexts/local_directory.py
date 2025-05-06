from __future__ import annotations

import json
import os.path
from functools import partial
from pathlib import Path
from typing import Any, Literal, Optional

from pharmpy.deps import pandas as pd
from pharmpy.internals.fs.lock import path_lock
from pharmpy.internals.fs.path import path_absolute
from pharmpy.internals.fs.symlink import create_directory_symlink
from pharmpy.internals.sort import sort_alphanum
from pharmpy.model import Model
from pharmpy.tools.mfl.parse import ModelFeatures
from pharmpy.workflows.hashing import ModelHash
from pharmpy.workflows.results import ModelfitResults, Results

from ..model_database import LocalModelDirectoryDatabase
from ..results import read_results
from .baseclass import Context


class LocalDirectoryContext(Context):
    """Context in a local directory

    Parameters
    ----------
    name : str
        Name of the context
    ref : str
        Path to directory. Will be created if it does not exist.
    """

    def __init__(
        self,
        name: str,
        ref: Optional[str] = None,
    ):
        if ref is None:
            ref = str(Path.cwd())
        path = Path(ref) / name

        isnew = self._init_path(path)
        if isnew:
            self._init_subcontexts()
        self._init_top_path()
        self._init_model_database()
        if isnew:
            self._init_annotations()
            self._init_model_name_map()
            self._init_log()
        super().__init__(name, ref)

    def __repr__(self) -> str:
        return f"<Local directory context at {self.path}>"

    def _init_path(self, path):
        self.path = path_absolute(path)
        if not self.path.is_dir():
            self.path.mkdir(parents=True)
            return True
        else:
            return False

    def _init_subcontexts(self):
        if not (self.path / 'subcontexts').is_dir():
            (self.path / 'subcontexts').mkdir()

    def _init_top_path(self):
        path = self.path
        while True:
            parent = path.parent
            if path == parent:
                raise FileNotFoundError("Cannot find top level of context.")
            if not (parent.name == "subcontexts"):
                self._top_path = path
                break
            path = parent.parent

    def _init_model_database(self):
        self._model_database = LocalModelDirectoryDatabase(self._top_path / '.modeldb')

    def _init_annotations(self):
        path = self._annotations_path
        if not path.is_file():
            path.touch()

    def _init_model_name_map(self):
        self._models_path.mkdir(exist_ok=True)

    def _init_log(self):
        log_path = self._log_path
        if not log_path.is_file():
            with open(log_path, 'w') as fh:
                fh.write("path,time,severity,message\n")

    def _read_lock(self, path: Path):
        # NOTE: Obtain shared (blocking) lock on one file
        path = path.with_suffix('.lock')
        path.touch(exist_ok=True)
        return path_lock(str(path), shared=True)

    def _write_lock(self, path: Path):
        # NOTE: Obtain exclusive (blocking) lock on one file
        path = path.with_suffix('.lock')
        path.touch(exist_ok=True)
        return path_lock(str(path), shared=False)

    def _delete_lock(self, path: Path):
        # Delete a lock
        path = path.with_suffix('.lock')
        if path.is_file():
            path.unlink()

    @staticmethod
    def exists(name: str, ref: Optional[str] = None):
        if ref is None:
            ref = Path.cwd()
        path = Path(ref) / name
        return (
            path.is_dir() and (path / 'subcontexts').is_dir() and (path / 'annotations').is_file()
        )

    def store_results(self, res: Results):
        res.to_json(path=self.path / 'results.json')
        res.to_csv(path=self.path / 'results.csv')

    def retrieve_results(self) -> Results:
        func = partial(_deserialize_model_from_hash, modeldb=self.model_database)
        res = read_results(self.path / 'results.json', model_deserialization_func=func)
        return res

    @property
    def _log_path(self) -> Path:
        return self._top_path / 'log.csv'

    @property
    def _metadata_path(self) -> Path:
        return self.path / 'metadata.json'

    @property
    def _models_path(self) -> Path:
        return self.path / 'models'

    @property
    def _annotations_path(self) -> Path:
        return self.path / 'annotations'

    @property
    def context_path(self) -> str:
        relpath = self.path.relative_to(self._top_path.parent)
        posixpath = str(relpath.as_posix())
        a = posixpath.split('/')[0::2]  # Remove subcontexts/
        ctxpath = '/'.join(a)
        return ctxpath

    def store_metadata(self, metadata: dict):
        with open(self._metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, cls=MetadataJSONEncoder)

    def retrieve_metadata(self) -> dict:
        if not self._metadata_path.is_file():
            return {}
        with open(self._metadata_path, 'r') as f:
            return json.load(f, cls=MetadataJSONDecoder)

    def store_key(self, name: str, key: ModelHash):
        from_path = self._models_path / name
        if not from_path.exists():
            absolute_to_path = self.model_database.path / str(key)
            if absolute_to_path.exists():
                if os.name != 'nt':
                    relative_to_path = Path(os.path.relpath(absolute_to_path, from_path.parent))
                else:
                    relative_to_path = absolute_to_path
                create_directory_symlink(from_path, relative_to_path)

    def retrieve_key(self, name: str) -> ModelHash:
        symlink_path = self._models_path / name
        resolved_path = symlink_path.resolve()
        if symlink_path == resolved_path:
            raise KeyError(f'There is no model with the name "{name}"')
        digest = resolved_path.name
        db = self.model_database
        with db.snapshot(ModelHash(digest)) as txn:
            key = txn.key
        return key

    def list_all_names(self) -> list(str):
        return sort_alphanum([f.name for f in Path(self._models_path).iterdir()])

    def list_all_subcontexts(self) -> list(str):
        path = self.path / 'subcontexts'
        return sort_alphanum([f.name for f in path.iterdir()])

    def retrieve_name(self, key: ModelHash) -> str:
        path = self._models_path
        mydigest = str(key)
        for link_path in path.iterdir():
            resolved = link_path.resolve()
            digest = resolved.name
            if digest == mydigest:
                return link_path.name
        raise KeyError(f"Model with key {mydigest} could not be found.")

    def store_annotation(self, name: str, annotation: str):
        path = self._annotations_path
        with self._write_lock(path):
            with open(path, 'r') as fh:
                lines = []
                found = False
                for line in fh.readlines():
                    a = line.split(" ", 1)
                    if a[0] == name:
                        lines.append(f'{name} {annotation}\n')
                        found = True
                    else:
                        lines.append(line)
                if not found:
                    lines.append(f'{name} {annotation}\n')
            with open(path, 'w') as fh:
                fh.writelines(lines)

    def retrieve_annotation(self, name: str) -> str:
        path = self._annotations_path
        with self._read_lock(path):
            with open(path, 'r') as fh:
                for line in fh.readlines():
                    a = line.split(" ", 1)
                    if a[0] == name:
                        return a[1][:-1]
        raise KeyError(f"No annotation for {name} available")

    def store_message(self, severity, ctxpath: str, date, message: str):
        log_path = self._log_path
        with self._write_lock(log_path):
            with open(log_path, 'a') as fh:

                def mangle_message(message):
                    return '"' + message.replace('"', '""') + '"'

                fh.write(f'{ctxpath},{date},{severity},{mangle_message(message)}\n')

    def retrieve_log(self, level: Literal['all', 'current', 'lower'] = 'all') -> pd.DataFrame:
        log_path = self._log_path
        with self._read_lock(log_path):
            df = pd.read_csv(log_path, parse_dates=['time'])
        count = df['path'].str.count('/')
        curlevel = self.context_path.count('/')
        if level == 'lower':
            df = df.loc[count >= curlevel]
        elif level == 'current':
            df = df.loc[count == curlevel]
        df = df.reset_index(drop=True)
        return df

    def retrieve_common_options(self) -> dict[str, Any]:
        ctx_top = self.get_top_level_context()
        meta = ctx_top.retrieve_metadata()
        return meta['common_options']

    def retrieve_dispatching_options(self) -> dict[str, Any]:
        if hasattr(self, '_dispatching_options'):
            return self._dispatching_options
        ctx_top = self.get_top_level_context()
        meta = ctx_top.retrieve_metadata()
        if 'dispatching_options' in meta:
            options = meta['dispatching_options']
        else:
            from pharmpy.workflows.args import (
                canonicalize_dispatching_options,
                get_default_dispatching_options,
            )

            options = get_default_dispatching_options()
            canonicalize_dispatching_options(options)
        self._dispatching_options = options
        return options

    def get_parent_context(self) -> LocalDirectoryContext:
        if self.path == self._top_path:
            raise ValueError("Already at the top level context")
        parent_path = self.path.parent.parent
        parent = LocalDirectoryContext(name=parent_path.name, ref=parent_path.parent)
        return parent

    def get_top_level_context(self) -> LocalDirectoryContext:
        ctx_top = LocalDirectoryContext(name=self._top_path.name, ref=self._top_path.parent)
        return ctx_top

    def get_subcontext(self, name: str) -> LocalDirectoryContext:
        subcontexts_path = self.path / 'subcontexts'
        if subcontexts_path.is_dir():
            path = subcontexts_path / name
        else:
            path = self.path / name
        if path.is_dir():
            return LocalDirectoryContext(name=name, ref=path.parent)
        else:
            raise ValueError(f'No subcontext with the name "{name}"')

    def create_subcontext(self, name: str) -> LocalDirectoryContext:
        subcontexts_path = self.path / 'subcontexts'
        if subcontexts_path.is_dir():
            path = subcontexts_path
        else:
            path = self.path
        ctx = LocalDirectoryContext(name=name, ref=path)
        return ctx

    def finalize(self):
        self._delete_lock(self._annotations_path)
        self._delete_lock(self._log_path)


class MetadataJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Model):
            # NOTE: This is only used by modelfit at the moment since we encode
            # models for other tools upstream.
            return obj.name
        elif isinstance(obj, ModelfitResults):
            return obj.to_json()
        elif isinstance(obj, ModelFeatures):
            return str(obj)
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


class MetadataJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        return obj


def _deserialize_model_from_hash(obj, key, modeldb):
    model = modeldb.retrieve_model(ModelHash(key))
    return model
