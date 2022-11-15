from dataclasses import fields, is_dataclass


def is_dataclass_instance(obj):
    return is_dataclass(obj) and not isinstance(obj, type)


def as_shallow_dict(obj):
    return {field.name: getattr(obj, field.name) for field in fields(obj)}
