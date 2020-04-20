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
