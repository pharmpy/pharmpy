import warnings
from importlib import import_module
from types import ModuleType


class LazyImport(ModuleType):
    """Class that masquerades as a module and lazily loads it when accessed the first time

    The code for the class is taken from TensorFlow and is under the Apache 2.0 license

    This is needed for the CLI --version to be fast and in general for the
    library to load progressively.
    """

    def __init__(self, local_name, parent_module_globals, name, attr=None):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        self._attr = attr

        super(LazyImport, self).__init__(name)

    def _load(self):
        # Import the target module and insert it into the parent's namespace
        name = self.__name__
        if name == 'pandas':
            # Because of https://github.com/pandas-dev/pandas/issues/54466
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                module = import_module(name)
            # To avoid warnings about deprecating silent downcasting
            version = tuple(int(i) for i in module.__version__.split('.'))
            if version >= (2, 2, 0):
                module.set_option('future.no_silent_downcasting', True)
        else:
            module = import_module(name)
        resolved = module if self._attr is None else getattr(module, self._attr)
        self._parent_module_globals[self._local_name] = resolved

        # Update this object's dict so that if someone keeps a reference to the
        #   LazyImport, lookups are efficient (__getattr__ is only called on lookups
        #   that fail).
        self.__dict__.update(resolved.__dict__)
        return resolved

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)
