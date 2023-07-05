from typing import Any, Dict, Optional

from pharmpy.model import EstimationStep, Model
from pharmpy.modeling.help_functions import _as_integer


def set_estimation_step(model: Model, method: str, idx: int = 0, **kwargs):
    """Set estimation step

    Sets estimation step for a model. Methods currently supported are:
        FO, FOCE, ITS, LAPLACE, IMPMAP, IMP, SAEM, BAYES

    Parameters
    ----------
    model : Model
        Pharmpy model
    method : str
        estimation method to change to
    idx : int
        index of estimation step, default is 0 (first estimation step)
    kwargs
        Arguments to pass to EstimationStep (such as interaction, evaluation)

    Returns
    -------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> opts = {'NITER': 1000, 'ISAMPLE': 100}
    >>> model = set_estimation_step(model, 'IMP', evaluation=True, tool_options=opts)
    >>> model.estimation_steps[0]   # doctest: +ELLIPSIS
    EstimationStep('IMP', interaction=True, cov='SANDWICH', evaluation=True, ..., tool_options=...

    See also
    --------
    add_estimation_step
    remove_estimation_step
    append_estimation_step_options
    add_covariance_step
    remove_covariance_step
    set_evaluation_step

    """
    try:
        idx = _as_integer(idx)
    except TypeError:
        raise TypeError(f'Index must be integer: {idx}')

    d = kwargs
    d['method'] = method
    steps = model.estimation_steps
    newstep = steps[idx].replace(**d)
    newsteps = steps[0:idx] + newstep + steps[idx + 1 :]
    model = model.replace(estimation_steps=newsteps)
    return model.update_source()


def add_estimation_step(model: Model, method: str, idx: Optional[int] = None, **kwargs):
    """Add estimation step

    Adds estimation step for a model in a given index. Methods currently supported are:
        FO, FOCE, ITS, LAPLACE, IMPMAP, IMP, SAEM

    Parameters
    ----------
    model : Model
        Pharmpy model
    method : str
        estimation method to change to
    idx : int
        index of estimation step (starting from 0), default is None (adds step at the end)
    kwargs
        Arguments to pass to EstimationStep (such as interaction, evaluation)

    Returns
    -------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> opts = {'NITER': 1000, 'ISAMPLE': 100}
    >>> model = add_estimation_step(model, 'IMP', tool_options=opts)
    >>> ests = model.estimation_steps
    >>> len(ests)
    2
    >>> ests[1]   # doctest: +ELLIPSIS
    EstimationStep('IMP', interaction=False, cov=None, ..., tool_options={'NITER': 1000,...

    See also
    --------
    set_estimation_step
    remove_estimation_step
    append_estimation_step_options
    add_covariance_step
    remove_covariance_step
    set_evaluation_step

    """
    meth = EstimationStep.create(method, **kwargs)

    if idx is not None:
        try:
            idx = _as_integer(idx)
        except TypeError:
            raise TypeError(f'Index must be integer: {idx}')
        steps = model.estimation_steps
        newsteps = steps[0:idx] + meth + steps[idx:]
    else:
        newsteps = model.estimation_steps + meth
    model = model.replace(estimation_steps=newsteps)

    return model.update_source()


def remove_estimation_step(model: Model, idx: int):
    """Remove estimation step

    Parameters
    ----------
    model : Model
        Pharmpy model
    idx : int
        index of estimation step to remove (starting from 0)

    Returns
    -------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = remove_estimation_step(model, 0)
    >>> ests = model.estimation_steps
    >>> len(ests)
    0

    See also
    --------
    add_estimation_step
    set_estimation_step
    append_estimation_step_options
    add_covariance_step
    remove_covariance_step
    set_evaluation_step

    """
    try:
        idx = _as_integer(idx)
    except TypeError:
        raise TypeError(f'Index must be integer: {idx}')

    steps = model.estimation_steps
    newsteps = steps[0:idx] + steps[idx + 1 :]
    model = model.replace(estimation_steps=newsteps)
    return model.update_source()


def append_estimation_step_options(model: Model, tool_options: Dict[str, Any], idx: int):
    """Append estimation step options

    Appends options to an existing estimation step.

    Parameters
    ----------
    model : Model
        Pharmpy model
    tool_options : dict
        any additional tool specific options
    idx : int
        index of estimation step (starting from 0)

    Returns
    -------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> opts = {'NITER': 1000, 'ISAMPLE': 100}
    >>> model = append_estimation_step_options(model, tool_options=opts, idx=0)
    >>> est = model.estimation_steps[0]
    >>> len(est.tool_options)
    2

    See also
    --------
    add_estimation_step
    set_estimation_step
    remove_estimation_step
    add_covariance_step
    remove_covariance_step
    set_evaluation_step

    """
    try:
        idx = _as_integer(idx)
    except TypeError:
        raise TypeError(f'Index must be integer: {idx}')

    steps = model.estimation_steps
    toolopts = dict(steps[idx].tool_options)
    toolopts.update(tool_options)
    newstep = steps[idx].replace(tool_options=toolopts)
    newsteps = steps[0:idx] + newstep + steps[idx + 1 :]
    model = model.replace(estimation_steps=newsteps)
    return model.update_source()


def add_covariance_step(model: Model, cov: str):
    """Adds covariance step to the final estimation step

    Parameters
    ----------
    model : Model
        Pharmpy model
    cov : str
        covariance method to use

    Returns
    -------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_estimation_step(model, 'FOCE', cov=None)
    >>> model = add_covariance_step(model, 'SANDWICH')
    >>> ests = model.estimation_steps
    >>> ests[0]   # doctest: +ELLIPSIS
    EstimationStep('FOCE', interaction=True, cov='SANDWICH', ...)

    See also
    --------
    add_estimation_step
    set_estimation_step
    remove_estimation_step
    append_estimation_step_options
    remove_covariance_step
    set_evaluation_step

    """
    steps = model.estimation_steps
    newstep = steps[-1].replace(cov=f'{cov}')
    newsteps = steps[0:-1] + newstep
    model = model.replace(estimation_steps=newsteps)
    return model.update_source()


def remove_covariance_step(model: Model):
    """Removes covariance step to the final estimation step

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = remove_covariance_step(model)
    >>> ests = model.estimation_steps
    >>> ests[0]   # doctest: +ELLIPSIS
    EstimationStep('FOCE', interaction=True, cov=None, ...)

    See also
    --------
    add_estimation_step
    set_estimation_step
    remove_estimation_step
    append_estimation_step_options
    add_covariance_step
    set_evaluation_step

    """
    steps = model.estimation_steps
    newstep = steps[-1].replace(cov=None)
    newsteps = steps[:-1] + newstep
    model = model.replace(estimation_steps=newsteps)
    return model.update_source()


def set_evaluation_step(model: Model, idx: int = -1):
    """Set estimation step

    Sets estimation step for a model. Methods currently supported are:
        FO, FOCE, ITS, LAPLACE, IMPMAP, IMP, SAEM, BAYES

    Parameters
    ----------
    model : Model
        Pharmpy model
    idx : int
        index of estimation step, default is -1 (last estimation step)

    Returns
    -------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_evaluation_step(model)
    >>> model.estimation_steps[0]   # doctest: +ELLIPSIS
    EstimationStep('FOCE', interaction=True, cov='SANDWICH', evaluation=True, ...

    See also
    --------
    set_estimation_step
    add_estimation_step
    remove_estimation_step
    append_estimation_step_options
    add_covariance_step
    remove_covariance_step

    """
    try:
        idx = _as_integer(idx)
    except TypeError:
        raise TypeError(f'Index must be integer: {idx}')

    steps = model.estimation_steps
    newstep = steps[idx].replace(evaluation=True)
    if idx != -1:
        newsteps = steps[0:idx] + newstep + steps[idx + 1 :]
    else:
        newsteps = steps[0:-1] + newstep
    model = model.replace(estimation_steps=newsteps)
    return model.update_source()
