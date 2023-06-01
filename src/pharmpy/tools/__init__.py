from threading import Lock

__all__ = (
    'create_report',  # pyright: ignore [reportUnsupportedDunderAll]
    'create_results',  # pyright: ignore [reportUnsupportedDunderAll]
    'fit',  # pyright: ignore [reportUnsupportedDunderAll]
    'load_example_modelfit_results',  # pyright: ignore [reportUnsupportedDunderAll]
    'predict_influential_individuals',  # pyright: ignore [reportUnsupportedDunderAll]
    'predict_influential_outliers',  # pyright: ignore [reportUnsupportedDunderAll]
    'predict_outliers',  # pyright: ignore [reportUnsupportedDunderAll]
    'print_fit_summary',  # pyright: ignore [reportUnsupportedDunderAll]
    'rank_models',  # pyright: ignore [reportUnsupportedDunderAll]
    'read_results',  # pyright: ignore [reportUnsupportedDunderAll]
    'read_modelfit_results',  # pyright: ignore [reportUnsupportedDunderAll]
    'resume_tool',  # pyright: ignore [reportUnsupportedDunderAll]
    'retrieve_final_model',  # pyright: ignore [reportUnsupportedDunderAll]
    'retrieve_models',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_allometry',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_amd',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_bootstrap',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_covsearch',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_estmethod',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_iivsearch',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_iovsearch',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_modelfit',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_modelsearch',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_ruvsearch',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_structsearch',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_tool',  # pyright: ignore [reportUnsupportedDunderAll]
    'summarize_errors',  # pyright: ignore [reportUnsupportedDunderAll]
    'summarize_individuals',  # pyright: ignore [reportUnsupportedDunderAll]
    'summarize_individuals_count_table',  # pyright: ignore [reportUnsupportedDunderAll]
    'summarize_modelfit_results',  # pyright: ignore [reportUnsupportedDunderAll]
    'write_results',  # pyright: ignore [reportUnsupportedDunderAll]
)


_allowed = set(__all__)

_not_wrapped = {
    '.amd.run': ('run_amd',),
    '.reporting': ('create_report',),
    '.run': (
        'create_results',
        'fit',
        'load_example_modelfit_results',
        'print_fit_summary',
        'rank_models',
        'read_modelfit_results',
        'read_results',
        'resume_tool',
        'retrieve_final_model',
        'retrieve_models',
        'run_tool',
        'summarize_errors',
        'summarize_modelfit_results',
        'write_results',
    ),
    '.funcs': (
        'predict_outliers',
        'predict_influential_individuals',
        'predict_influential_outliers',
        'summarize_individuals',
        'summarize_individuals_count_table',
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
            tool_name = key[4:]  # NOTE This removes the run_ prefix
            _tool_cache[key] = wrap(tool_name)

    return _tool_cache[key]


def __dir__():
    return __all__
