from threading import Lock

__all__ = (
    'create_results',  # pyright: ignore [reportUnsupportedDunderAll]
    'fit',  # pyright: ignore [reportUnsupportedDunderAll]
    'predict_influential_individuals',  # pyright: ignore [reportUnsupportedDunderAll]
    'predict_influential_outliers',  # pyright: ignore [reportUnsupportedDunderAll]
    'predict_outliers',  # pyright: ignore [reportUnsupportedDunderAll]
    'print_fit_summary',  # pyright: ignore [reportUnsupportedDunderAll]
    'rank_models',  # pyright: ignore [reportUnsupportedDunderAll]
    'read_results',  # pyright: ignore [reportUnsupportedDunderAll]
    'read_modelfit_results',  # pyright: ignore [reportUnsupportedDunderAll]
    'retrieve_final_model',  # pyright: ignore [reportUnsupportedDunderAll]
    'retrieve_models',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_allometry',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_amd',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_covsearch',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_iivsearch',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_iovsearch',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_modelfit',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_modelsearch',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_ruvsearch',  # pyright: ignore [reportUnsupportedDunderAll]
    'run_tool',  # pyright: ignore [reportUnsupportedDunderAll]
    'summarize_errors',  # pyright: ignore [reportUnsupportedDunderAll]
    'summarize_individuals',  # pyright: ignore [reportUnsupportedDunderAll]
    'summarize_individuals_count_table',  # pyright: ignore [reportUnsupportedDunderAll]
    'summarize_modelfit_results',  # pyright: ignore [reportUnsupportedDunderAll]
    'write_results',  # pyright: ignore [reportUnsupportedDunderAll]
)


_allowed = set(__all__)

_run_keys = {
    'create_results',
    'fit',
    'print_fit_summary',
    'rank_models',
    'read_modelfit_results',
    'read_results',
    'retrieve_final_model',
    'retrieve_models',
    'run_tool',
    'summarize_errors',
    'summarize_modelfit_results',
    'write_results',
}

_func_keys = {
    'predict_outliers',
    'predict_influential_individuals',
    'predict_influential_outliers',
    'summarize_individuals',
    'summarize_individuals_count_table',
}

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

    if key in _func_keys:
        module = importlib.import_module('.funcs', __name__)
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
