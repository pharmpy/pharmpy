"""Pharmpy configuration
"""

import configparser
import sys
from pathlib import Path

import appdirs


def read_configuration():
    appname = 'Pharmpy'
    filename = 'pharmpy.conf'
    config = configparser.ConfigParser()
    user_path = Path(appdirs.user_config_dir(appname)) / filename
    if user_path.is_file():
        config.read(user_path)
    else:
        site_path = Path(appdirs.site_config_dir(appname)) / filename
        if site_path.is_file():
            config.read(user_path)
    return config


config = read_configuration()


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
    def __init__(self):
        config.sections()
        for module_name in sys.modules.keys():
            if module_name.startswith('pharmpy.'):
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
