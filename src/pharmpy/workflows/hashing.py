from __future__ import annotations

import base64
import hashlib
import json
from typing import Union

from pharmpy.deps import pandas as pd
from pharmpy.model import Model
from pharmpy.modeling import load_dataset

from .model_entry import ModelEntry


def _encode(obj):
    # Encode a model object into a bytes string
    d = obj.to_dict()
    js = json.dumps(d)
    enc = js.encode('utf-8')
    return enc


def _update_hash_with_dataset(df, h):
    hash_series = pd.util.hash_pandas_object(
        df, index=False, encoding='utf8', hash_key='0123456789123456', categorize=True
    )
    for val in hash_series:
        h.update(int(val).to_bytes(8, byteorder='big'))

    columns = repr(list(df.columns)).encode('utf-8')
    index = repr(df.index).encode('utf-8')
    dtypes = repr(list(df.dtypes)).encode('utf-8')
    h.update(columns)
    h.update(index)
    h.update(dtypes)


def _hash_to_string(h):
    digest = h.digest()
    b64 = base64.urlsafe_b64encode(digest).decode()
    return b64.replace('=', '')  # Remove padding characters


class Hash:
    def __str__(self):
        return self._hash


class DatasetHash(Hash):
    def __init__(self, df):
        h = hashlib.sha256()
        _update_hash_with_dataset(df, h)
        self._hash = _hash_to_string(h)


class ModelHash(Hash):
    def __init__(self, obj: Union[str, Model, ModelEntry, ModelHash]):
        if isinstance(obj, str):
            if len(obj) != 43:  # SHA-256 in base-64 encoding
                raise ValueError(f"Invalid digest to construct hash: {obj}")
            self._hash = obj
            # FIXME: No dataset hash here
            self.dataset_hash = None
        elif isinstance(obj, ModelHash):
            self.dataset_hash = obj.dataset_hash
            self._hash = obj._hash
        else:
            if isinstance(obj, ModelEntry):
                model = obj.model
            else:
                model = obj
            di = model.datainfo
            if di is not None:
                if model.dataset is None:
                    model = load_dataset(model)
                di = di.replace(path=None)
                model = model.replace(datainfo=di)
            model = model.replace(name='', description='')
            model_bytes = _encode(model)
            h = hashlib.sha256()
            _update_hash_with_dataset(model.dataset, h)
            self.dataset_hash = _hash_to_string(h)
            h.update(model_bytes)
            self._hash = _hash_to_string(h)
