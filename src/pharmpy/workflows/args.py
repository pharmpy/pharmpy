from collections.abc import Mapping
from typing import Any

from pharmpy.internals.fs.path import normalize_user_given_path


def split_common_options(d) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Split the dict into dispatching options, common options and other options

    Dispatching options will be handled before the tool is run to setup the context and dispatching
    system. Common options will be handled by the context so that all tasks in the workflow can get
    them. The tool specific options will be sent directly to the tool.

    Parameters
    ----------
    d : dict
        Dictionary of all options

    Returns
    -------
    Tuple of dispatching options, common options and other option dictionaries
    """
    all_dispatching_options = ('context', 'path')
    all_common_options = ('resume', 'esttool')
    dispatching_options = {}
    common_options = {}
    other_options = {}
    for key, value in d.items():
        if key in all_dispatching_options:
            dispatching_options[key] = value
        elif key in all_common_options:
            if key == 'path':
                if value is not None:
                    value = normalize_user_given_path(value)
            common_options[key] = value
        else:
            other_options[key] = value
    return dispatching_options, common_options, other_options
