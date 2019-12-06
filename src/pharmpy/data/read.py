# Read dataset from file
import importlib


def _plugin_module(plugin_name):
    module_name = f'pharmpy.plugins.{plugin_name}.data'        # Crude way of importing for now.
    module = importlib.import_module(module_name)
    return module


def read_raw_dataset(path_or_io, file_format='nonmem', **kwargs):
    """Read a dataset from file with minimal processing
       data will be kept in string format.
    """
    module = _plugin_module(file_format)
    return module.read_raw_dataset(path, **kwargs)


def read_dataset(path_or_io, file_format='nonmem', **kwargs):
    """Read a dataset from file
    """
    module = _plugin_module(file_format)
    return module.read_dataset(path, **kwargs)
