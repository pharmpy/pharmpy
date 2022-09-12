from threading import Lock

__all__ = (
    'create_results',
    'fit',
    'read_results',
    'retrieve_models',
    'run_allometry',
    'run_amd',
    'run_covsearch',
    'run_iivsearch',
    'run_iovsearch',
    'run_modelsearch',
    'run_ruvsearch',
    'run_tool',
)


_allowed = set(__all__)

_run_keys = {'create_results', 'fit', 'read_results', 'retrieve_models', 'run_tool'}

_tool_cache = {}
_tool_lock = Lock()


def __getattr__(key):
    if key not in _allowed:
        raise AttributeError(key)

    import importlib

    if key == 'run_amd':
        module = importlib.import_module('.amd.run', __name__)
        return getattr(module, 'run_amd')

    if key in _run_keys:
        module = importlib.import_module('.run', __name__)
        return getattr(module, key)

    assert key[:4] == 'run_'

    with _tool_lock:
        if key not in _tool_cache:
            module = importlib.import_module('.wrap', __name__)
            wrap = getattr(module, 'wrap')
            _tool_cache[key] = wrap(key[4:])

    return _tool_cache[key]


def __dir__():
    return __all__
