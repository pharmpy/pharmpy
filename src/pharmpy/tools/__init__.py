from threading import Lock

__all__ = (
    'broadcast_log',  # pyright: ignore [reportUnsupportedDunderAll]
    'create_report',  # pyright: ignore [reportUnsupportedDunderAll]
    'fit',  # pyright: ignore [reportUnsupportedDunderAll]
    'open_context',  # pyright: ignore [reportUnsupportedDunderAll]
    'is_strictness_fulfilled',  # pyright: ignore [reportUnsupportedDunderAll]
    'load_example_modelfit_results',  # pyright: ignore [reportUnsupportedDunderAll]
    'list_models',  # pyright: ignore [reportUnsupportedDunderAll]
    'predict_influential_individuals',  # pyright: ignore [reportUnsupportedDunderAll]
    'predict_influential_outliers',  # pyright: ignore [reportUnsupportedDunderAll]
    'predict_outliers',  # pyright: ignore [reportUnsupportedDunderAll]
    'print_fit_summary',  # pyright: ignore [reportUnsupportedDunderAll]
    'print_log',  # pyright: ignore [reportUnsupportedDunderAll]
    'read_results',  # pyright: ignore [reportUnsupportedDunderAll]
    'read_modelfit_results',  # pyright: ignore [reportUnsupportedDunderAll]
    'retrieve_model',  # pyright: ignore [reportUnsupportedDunderAll]
    'retrieve_modelfit_results',  # pyright: ignore [reportUnsupportedDunderAll]
    'retrieve_models',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_allometry',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_amd',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_bootstrap',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_covsearch',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_estmethod',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_iivsearch',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_iovsearch',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_linearize',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_modelfit',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_modelsearch',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_retries',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_ruvsearch',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_structsearch',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_tool',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_simulation',  # pyright: ignore [reportUnsupportedDunderAll]
    'summarize_modelfit_results',  # pyright: ignore [reportUnsupportedDunderAll]
    'write_results',  # pyright: ignore [reportUnsupportedDunderAll]
)


_allowed = set(__all__)

_not_wrapped = {
    '.reporting': ('create_report',),
    '.run': (
        'fit',
        'is_strictness_fulfilled',
        'load_example_modelfit_results',
        'print_fit_summary',
        'read_modelfit_results',
        'read_results',
        'retrieve_models',
        'run_tool',
        'summarize_modelfit_results',
        'write_results',
    ),
    '.funcs': (
        'predict_outliers',
        'predict_influential_individuals',
        'predict_influential_outliers',
    ),
    '.context': (
        'open_context',
        'print_log',
        'broadcast_log',
        'retrieve_model',
        'retrieve_modelfit_results',
        'list_models',
    ),
}

_not_wrapped_module_name_index = {
    key: module for module, keys in _not_wrapped.items() for key in keys
}

_tool_cache = {}
_tool_lock = Lock()


def __getattr__(key):
    if key not in _allowed:
        raise AttributeError(key)

    import importlib

    if key in _not_wrapped_module_name_index:
        module = importlib.import_module(_not_wrapped_module_name_index[key], __name__)
        return getattr(module, key)

    assert key.startswith('run_')

    with _tool_lock:
        if key not in _tool_cache:
            module = importlib.import_module('.wrap', __name__)
            wrap = getattr(module, 'wrap')
            tool_name = key[4:]  # NOTE: This removes the run_ prefix
            _tool_cache[key] = wrap(tool_name)

    return _tool_cache[key]


def __dir__():
    return __all__
