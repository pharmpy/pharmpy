import base64
import hashlib
import json

from pharmpy.deps import numpy as np
from pharmpy.modeling import load_dataset


def _encode(obj):
    # Encode a model subobject into a bytes string
    d = obj.to_dict()
    js = json.dumps(d)
    enc = js.encode('utf-8')
    return enc


def _update_hash_with_dataset(df, h):
    values = df.values
    if not values.flags['C_CONTIGUOUS']:
        values = np.ascontiguousarray(values)
    columns = repr(df.columns).encode('utf-8')
    index = repr(df.index).encode('utf-8')
    dtypes = repr(list(df.dtypes)).encode('utf-8')
    h.update(values)
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
    def __init__(self, model_or_hash):
        if isinstance(model_or_hash, ModelHash):
            self.dataset_hash = model_or_hash.dataset_hash
            self._hash = model_or_hash._hash
        else:
            model = model_or_hash
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
