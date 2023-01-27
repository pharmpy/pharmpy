from typing import Any, Mapping, Tuple

from pharmpy.internals.fs.path import normalize_user_given_path


def split_common_options(d) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Split the dict into common options and other options

    Parameters
    ----------
    d : dict
        Dictionary of all options

    Returns
    -------
    Tuple of common options and other option dictionaries
    """
    execute_options = ['path', 'resume']
    common_options = {}
    other_options = {}
    for key, value in d.items():
        if key in execute_options:
            if key == 'path':
                if value is not None:
                    value = normalize_user_given_path(value)
            common_options[key] = value
        else:
            other_options[key] = value
    return common_options, other_options
