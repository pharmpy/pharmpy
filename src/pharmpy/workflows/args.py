from collections.abc import Mapping
from typing import Any

from pharmpy import DEFAULT_SEED
from pharmpy.internals.fs.path import normalize_user_given_path
from pharmpy.workflows.broadcasters import Broadcaster
from pharmpy.workflows.dispatchers import Dispatcher

ALLOWED_ESTTOOLS = (None, 'dummy', 'nonmem', 'nlmixr')


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
    all_dispatching_options = ('context', 'path', 'broadcaster', 'dispatcher')
    all_common_options = ('resume', 'esttool', 'seed')
    # The defaults below will be overwritten by the user given options
    dispatching_options = {'broadcaster': None, 'dispatcher': None}
    common_options = {'seed': DEFAULT_SEED}
    other_options = {}
    for key, value in d.items():
        if key in all_dispatching_options:
            dispatching_options[key] = value
        elif key in all_common_options:
            if key == 'path':
                if value is not None:
                    value = normalize_user_given_path(value)
            elif key == 'esttool':
                if value not in ALLOWED_ESTTOOLS:
                    raise ValueError(
                        f"Invalid estimation tool {value}, must be one of {ALLOWED_ESTTOOLS}"
                    )
            common_options[key] = value
        else:
            other_options[key] = value
    dispatching_options['broadcaster'] = Broadcaster.canonicalize_broadcaster_name(
        dispatching_options['broadcaster']
    )
    dispatching_options['dispatcher'] = Dispatcher.canonicalize_dispatcher_name(
        dispatching_options['dispatcher']
    )
    return dispatching_options, common_options, other_options
