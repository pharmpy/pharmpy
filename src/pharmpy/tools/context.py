from pathlib import Path
from typing import Union

from pharmpy.internals.fs.path import normalize_user_given_path
from pharmpy.model import Model
from pharmpy.workflows import ModelfitResults
from pharmpy.workflows.broadcasters import Broadcaster
from pharmpy.workflows.contexts import Context, LocalDirectoryContext


def _open_context(source):
    if isinstance(source, Path) or isinstance(source, str):
        path = Path(source)
        context = LocalDirectoryContext(path)
    elif isinstance(source, Context):
        context = source
    else:
        raise NotImplementedError(f'Not implemented for type \'{type(source)}\'')
    return context


def init_context(name: str, path: Union[str, Path, None] = None):
    """Initialize an already created context or create a new context

    Parameters
    ----------
    name : str
        Name of the context
    path : str or Path
        Path to where to put the context

    Examples
    --------
    >>> from pharmpy.tools import init_context
    >>> ctx = init_context("myproject")  # doctest: +SKIP

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
    broadcaster = Broadcaster.select_broadcaster("terminal")
    df = context.retrieve_log()
    for _, row in df.iterrows():
        broadcaster.broadcast_message(row['severity'], row['path'], row['time'], row['message'])


def retrieve_model(
    source: Union[str, Path, Context],
    name: str,
) -> Model:
    """Retrieve a model from a context/tool run

    Any models created and run by the tool can be
    retrieved.

    Parameters
    ----------
    source : str, Path, Context
        Source where to find models. Can be a path (as str or Path), or a
        Context
    name : str
        Name of the model or a qualified name with a subcontext path, e.g.
        :code:`"iivsearch/@final"`.

    Return
    ------
    Model
        The model object

    Examples
    --------
    >>> from pharmpy.tools import retrieve_model
    >>> tooldir_path = 'path/to/tool/directory'
    >>> model = retrieve_model(tooldir_path, 'run1')      # doctest: +SKIP

    """
    context = _open_context(source)
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


def retrieve_modelfit_results(
    source: Union[str, Path, Context],
    name: str,
) -> ModelfitResults:
    """Retrieve the modelfit results of a model

    Parameters
    ----------
    source : str, Path, Context
        Source where to find models. Can be a path (as str or Path), or a
        Context
    name : str
        Name of the model or a qualified name with a subcontext path, e.g.
        :code:`"iivsearch/@final"`.

    Return
    ------
    ModelfitResults
        The results object

    Examples
    --------
    >>> from pharmpy.tools import init_context, retrieve_modelfit_results
    >>> tooldir_path = 'path/to/tool/directory'
    >>> context = init_context("iivsearch1")   # doctest: +SKIP
    >>> results = retrieve_modelfit_results(context, 'input')      # doctest: +SKIP

    """
    context = _open_context(source)
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
