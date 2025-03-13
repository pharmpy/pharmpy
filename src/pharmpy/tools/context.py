from pathlib import Path
from typing import Optional, Union

from pharmpy.internals.fs.path import normalize_user_given_path
from pharmpy.model import Model
from pharmpy.workflows import ModelfitResults
from pharmpy.workflows.broadcasters import Broadcaster
from pharmpy.workflows.contexts import Context, LocalDirectoryContext


def open_context(name: str, ref: Union[str, Path, None] = None):
    """Open a context from a tool run

    Parameters
    ----------
    name : str
        Name of the context
    ref : str or Path
        Parent path of the context

    Examples
    --------
    >>> from pharmpy.tools import open_context
    >>> ctx = open_context("myrun")  # doctest: +SKIP

    """
    ref = str(normalize_user_given_path(ref)) if ref is not None else None
    ctx = LocalDirectoryContext(name=name, ref=ref)
    return ctx


def print_log(context: Context) -> None:
    """Print the log of a context

    Parameters
    ----------
    context : Context
        Print the log of this context

    """
    broadcast_log(context, "terminal")


def broadcast_log(context: Context, broadcaster: Optional[str] = None) -> None:
    """Broadcast the log of a context

    Default is to use the same broadcaster, but optionally another broadcaster could
    be used.

    Parameters
    ----------
    context : Context
        Broadcast the log of this context
    broadcaster : str
        Name of the broadcaster to use. Default is to use the same as was original used.

    """
    if broadcaster is not None:
        bcst = context.broadcaster
    else:
        bcst = Broadcaster.select_broadcaster(broadcaster)
    df = context.retrieve_log()
    for _, row in df.iterrows():
        bcst.broadcast_message(row['severity'], row['path'], row['time'], row['message'])


def retrieve_model(context: Context, name: str) -> Model:
    """Retrieve a model from a context

    Any models created and run by the tool can be
    retrieved.

    Parameters
    ----------
    context: Context
        A previously opened context
    name : str
        Name of the model or a qualified name with a subcontext path, e.g.
        :code:`"iivsearch/@final"`.

    Return
    ------
    Model
        The model object

    Examples
    --------
    >>> from pharmpy.tools import open_context, retrieve_model
    >>> context = open_context(ref='path/to/', name='modelsearch1')
    >>> model = retrieve_model(context, 'run1')      # doctest: +SKIP

    """
    subctx, model_name = _split_subcontext_and_model(context, name)
    return subctx.retrieve_model_entry(model_name).model


def _split_subcontext_and_model(context: Context, ctx_path: str) -> tuple[Context, str]:
    """Given a context and a context path get the lowest subcontext
    in the path and the model name. If no model is in the path
    return None instead.
    """
    a = ctx_path.split('/')
    if a[-1].startswith('@'):
        model_name = a[-1][1:]
        a = a[:-1]
    elif len(a) == 1:
        # In this case what we have is the model_name
        # and the context is the same
        model_name = a[0]
        a = []
    else:
        model_name = None
    for subname in a:
        try:
            context = context.get_subcontext(subname)
        except ValueError as e:
            all_subs = context.list_all_subcontexts()
            raise ValueError(f"{e}. Did you mean {_present_string_array(all_subs)}?")

    return context, model_name


def _present_string_array(a):
    """Create a user presentable string from array of strings"""
    a = [f'"{e}"' for e in a]
    if len(a) == 1:
        return f'{a[0]}'
    else:
        a = a[:-2] + [f'{a[-2]} or a[-1]']
        return 'one of: ' + ', '.join(a)


def retrieve_modelfit_results(context: Context, name: str) -> ModelfitResults:
    """Retrieve the modelfit results of a model

    Parameters
    ----------
    context : Context
        A previously opened context
    name : str
        Name of the model or a qualified name with a subcontext path, e.g.
        :code:`"iivsearch/@final"`.

    Return
    ------
    ModelfitResults
        The results object

    Examples
    --------
    >>> from pharmpy.tools import open_context, retrieve_modelfit_results
    >>> context = open_context("iivsearch1")   # doctest: +SKIP
    >>> results = retrieve_modelfit_results(context, 'input')      # doctest: +SKIP

    """
    subctx, model_name = _split_subcontext_and_model(context, name)
    return subctx.retrieve_model_entry(model_name).modelfit_results


def list_models(context: Context, recursive: bool = False) -> list[str]:
    """List names of all models in a context

    Will by default list only models in the top level, but can list
    all recursively using the recursive option. This will add the context
    path to each model name as a qualifier.

    Parameters
    ----------
    context : Context
        The context
    recursive : bool
        Only top level or all levels recursively down.

    Return
    ------
    list[str]
        A list of the model names
    """
    if not recursive:
        names = context.list_all_names()
    else:
        names = _get_model_names_recursive(context)
    return names


def _get_model_names_recursive(context):
    ctx_path = context.context_path
    names = [ctx_path + "/@" + name for name in context.list_all_names()]
    for subctx_name in context.list_all_subcontexts():
        subctx = context.get_subcontext(subctx_name)
        subctx_names = _get_model_names_recursive(subctx)
        names += subctx_names
    return names
