import importlib
from json import JSONDecoder, JSONEncoder

from pharmpy.internals.dataclass import as_shallow_dict, is_dataclass_instance


class MetadataJSONEncoder(JSONEncoder):
    def default(self, obj):
        if is_dataclass_instance(obj):
            return _json_encode_dataclass_instance(obj)

        # NOTE Default behavior
        return super().default(obj)


class MetadataJSONDecoder(JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if _is_json_encoded_dataclass_instance(obj):
            return _json_decode_dataclass_instance(obj)

        # NOTE Default behavior
        return obj


def _json_encode_dataclass_instance(obj):
    cls = obj.__class__
    return {
        '__type__': 'dataclass',
        '__module__': cls.__module__,
        '__class__': cls.__qualname__,
        '__dict__': as_shallow_dict(obj),
    }


def _is_json_encoded_dataclass_instance(obj: dict):
    return obj.get('__type__') == 'dataclass'


def _json_decode_dataclass_instance(obj: dict):
    qualname = obj['__class__']
    module_path = obj['__module__']
    module = importlib.import_module(module_path)
    cls = getattr(module, qualname)
    return cls(**obj['__dict__'])
