"""Pharmpy configuration
"""

# import configparser


class ConfigItem:
    def __init__(self, default, description):
        self.default = default
        self.__doc__ = description

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner=None):
        return instance.__dict__.get(self.name, self.default)

    def __set__(self, instance, value):
        if type(self.default) != type(value):
            raise TypeError(f'Trying to set configuration item {self.name} using object of wrong '
                            f'type: {type(value)} is not {type(self.default)}')
        instance.__dict__[self.name] = value


class Configuration:
    pass


class ConfigurationContext:
    """Context to temporarily set configuration options
    """
    def __init__(self, config, **kwargs):
        self.config = config
        self.options = kwargs

    def __enter__(self):
        old = dict()
        for key in self.options.keys():
            old[key] = getattr(self.config, key)
        for key, val in self.options.items():
            setattr(self.config, key, val)
        self.old = old
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, val in self.old.items():
            setattr(self.config, key, val)
