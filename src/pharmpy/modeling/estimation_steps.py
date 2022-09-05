from pharmpy.model import EstimationStep
from pharmpy.modeling.help_functions import _as_integer


def set_estimation_step(model, method, idx=0, **kwargs):
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
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> opts = {'NITER': 1000, 'ISAMPLE': 100}
    >>> set_estimation_step(model, "IMP", evaluation=True, tool_options=opts)   # doctest: +ELLIPSIS
    <...>
    >>> model.estimation_steps[0]   # doctest: +ELLIPSIS
    EstimationStep("IMP", interaction=True, cov=True, evaluation=True, ..., tool_options=...

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
    newstep = steps[idx].derive(**d)
    model.estimation_steps = steps[0:idx] + newstep + steps[idx + 1 :]
    return model


def add_estimation_step(model, method, idx=None, **kwargs):
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
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> opts = {'NITER': 1000, 'ISAMPLE': 100}
    >>> add_estimation_step(model, "IMP", tool_options=opts)   # doctest: +ELLIPSIS
    <...>
    >>> ests = model.estimation_steps
    >>> len(ests)
    2
    >>> ests[1]   # doctest: +ELLIPSIS
    EstimationStep("IMP", interaction=False, cov=False, ..., tool_options={'NITER': 1000,...

    See also
    --------
    set_estimation_step
    remove_estimation_step
    append_estimation_step_options
    add_covariance_step
    remove_covariance_step
    set_evaluation_step

    """
    meth = EstimationStep(method, **kwargs)

    if idx is not None:
        try:
            idx = _as_integer(idx)
        except TypeError:
            raise TypeError(f'Index must be integer: {idx}')
        steps = model.estimation_steps
        model.estimation_steps = steps[0:idx] + meth + steps[idx:]
    else:
        model.estimation_steps = model.estimation_steps + meth

    return model


def remove_estimation_step(model, idx):
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
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> remove_estimation_step(model, 0)      # doctest: +ELLIPSIS
    <...>
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
    model.estimation_steps = steps[0:idx] + steps[idx + 1 :]
    return model


def append_estimation_step_options(model, tool_options, idx):
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
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> opts = {'NITER': 1000, 'ISAMPLE': 100}
    >>> append_estimation_step_options(model, tool_options=opts, idx=0)   # doctest: +ELLIPSIS
    <...>
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
    toolopts = steps[idx].tool_options.copy()
    toolopts.update(tool_options)
    newstep = steps[idx].derive(tool_options=toolopts)
    model.estimation_steps = steps[0:idx] + newstep + steps[idx + 1 :]
    return model


def add_covariance_step(model):
    """Adds covariance step to the final estimation step

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    Model
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> set_estimation_step(model, 'FOCE', cov=False)      # doctest: +ELLIPSIS
    <...>
    >>> add_covariance_step(model)      # doctest: +ELLIPSIS
    <...>
    >>> ests = model.estimation_steps
    >>> ests[0]   # doctest: +ELLIPSIS
    EstimationStep("FOCE", interaction=True, cov=True, ...)

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
    newstep = steps[-1].derive(cov=True)
    model.estimation_steps = steps[0:-1] + newstep
    return model


def remove_covariance_step(model):
    """Removes covariance step to the final estimation step

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    Model
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> remove_covariance_step(model)      # doctest: +ELLIPSIS
    <...>
    >>> ests = model.estimation_steps
    >>> ests[0]   # doctest: +ELLIPSIS
    EstimationStep("FOCE", interaction=True, cov=False, ...)

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
    newstep = steps[-1].derive(cov=False)
    model.estimation_steps = steps[:-1] + newstep
    return model


def set_evaluation_step(model, idx=-1):
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
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> set_evaluation_step(model)   # doctest: +ELLIPSIS
    <...>
    >>> model.estimation_steps[0]   # doctest: +ELLIPSIS
    EstimationStep("FOCE", interaction=True, cov=True, evaluation=True, ...

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
    newstep = steps[idx].derive(evaluation=True)
    if idx != -1:
        model.estimation_steps = steps[0:idx] + newstep + steps[idx + 1 :]
    else:
        model.estimation_steps = steps[0:-1] + newstep
    return model
