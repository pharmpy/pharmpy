from pathlib import Path
from typing import Union

from pharmpy.internals.fs.path import normalize_user_given_path
from pharmpy.workflows.contexts import Context, LocalDirectoryContext
from pharmpy.workflows.contexts.broadcasters.terminal import broadcast_message


def create_context(name: str, path: Union[str, Path, None] = None):
    """Create a new context

    Currently a local filesystem context (i.e. a directory)

    Parameters
    ----------
    name : str
        Name of the context
    path : str or Path
        Path to where to put the context

    Examples
    --------
    >>> from pharmpy.tools import create_context
    >>> ctx = create_context("myproject")  # doctest: +SKIP

    """
    ref = str(normalize_user_given_path(path)) if path is not None else None
    ctx = LocalDirectoryContext(name=name, ref=ref)
    return ctx


def print_log(context: Context):
    """Print the log of a context

    Parameters
    ----------
    context : Context
        Print the log of this context

    """
    df = context.retrieve_log()
    for _, row in df.iterrows():
        broadcast_message(row['severity'], row['path'], row['time'], row['message'])
