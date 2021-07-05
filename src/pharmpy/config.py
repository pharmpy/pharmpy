"""Pharmpy configuration
"""

import configparser
import os
from pathlib import Path

import appdirs

appname = 'Pharmpy'
configuration_filename = 'pharmpy.conf'


def user_config_dir():
    user_path = Path(appdirs.user_config_dir(appname)) / configuration_filename
    return user_path


def site_config_dir():
    site_path = Path(appdirs.site_config_dir(appname)) / configuration_filename
    return site_path


def read_configuration():
    config = configparser.ConfigParser()
    env_path = os.getenv('PHARMPYCONFIGPATH')
    if env_path is not None:
        env_path = Path(env_path) / configuration_filename
        if not env_path.is_file():
            raise ValueError(
                'Environment variable PHARMPYCONFIGPATH is set but directory does '
                'not contain a configuration file'
            )
        config.read(env_path)
    else:
        user_path = user_config_dir()
        if user_path.is_file():
            config.read(user_path)
        else:
            site_path = site_config_dir()
            if site_path.is_file():
                config.read(site_path)
    return config


config_file = read_configuration()


class ConfigItem:
    def __init__(self, default, description, cls=None):
        self.default = default
        self.__doc__ = description
        if cls is None:
            self.cls = type(self.default)
        else:
            self.cls = cls

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner=None):
        return instance.__dict__.get(self.name, self.default)

    def __set__(self, instance, value):
        # if self.cls != type(value):
        try:
            if isinstance(self.default, list) and isinstance(value, str):
                value = [v.strip() for v in value.split(',')]
            if isinstance(self.default, bool) and isinstance(value, str):
                if value.lower() == 'true':
                    value = True
                else:
                    value = False
            instance.__dict__[self.name] = self.cls(value)
        except ValueError as exc:
            raise TypeError(
                f'Trying to set configuration item {self.name} using object of wrong '
                f'type: {type(value)} is not {self.cls}'
            ) from exc


class Configuration:
    def __init__(self):
        noconfigfile = bool(int(os.getenv('PHARMPYNOCONFIGFILE', 0)))
        if noconfigfile:
            return
        if self.module in config_file.keys():
            for key, value in config_file[self.module].items():
                setattr(self, key, value)

    def __str__(self):
        settings = ''
        for key, value in vars(self).items():
            settings += f"{key}:\t{value}\n"
        return settings


class ConfigurationContext:
    """Context to temporarily set configuration options"""

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
