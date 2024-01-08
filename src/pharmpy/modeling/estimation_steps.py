from typing import Any, Dict, Literal, Optional

from pharmpy.model import EstimationStep, EstimationSteps, Model, SimulationStep
from pharmpy.modeling.help_functions import _as_integer

ESTIMATION_METHODS = ('FO', 'FOCE', 'ITS', 'LAPLACE', 'IMPMAP', 'IMP', 'SAEM', 'BAYES')


def set_estimation_step(model: Model, method: Literal[ESTIMATION_METHODS], idx: int = 0, **kwargs):
    """Set estimation step

    Sets estimation step for a model. Methods currently supported are:
        FO, FOCE, ITS, LAPLACE, IMPMAP, IMP, SAEM, BAYES

    Parameters
    ----------
    model : Model
        Pharmpy model
    method : {'FO', 'FOCE', 'ITS', 'LAPLACE', 'IMPMAP', 'IMP', 'SAEM', 'BAYES'}
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
    EstimationStep('IMP', interaction=True, parameter_uncertainty_method='SANDWICH', evaluation=True, ...,
    tool_options=...

    See also
    --------
    add_estimation_step
    remove_estimation_step
    append_estimation_step_options
    add_parameter_uncertainty_step
    remove_parameter_uncertainty_step
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


def add_estimation_step(
    model: Model, method: Literal[ESTIMATION_METHODS], idx: Optional[int] = None, **kwargs
):
    """Add estimation step

    Adds estimation step for a model in a given index. Methods currently supported are:
        FO, FOCE, ITS, LAPLACE, IMPMAP, IMP, SAEM

    Parameters
    ----------
    model : Model
        Pharmpy model
    method : {'FO', 'FOCE', 'ITS', 'LAPLACE', 'IMPMAP', 'IMP', 'SAEM', 'BAYES'}
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
    EstimationStep('IMP', interaction=False, parameter_uncertainty_method=None, ..., tool_options={'NITER': 1000,...

    See also
    --------
    set_estimation_step
    remove_estimation_step
    append_estimation_step_options
    add_parameter_uncertainty_step
    remove_parameter_uncertainty_step
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


def set_simulation(model: Model, n: int = 1, seed: int = 64206):
    """Change model into simulation model

    Parameters
    ----------
    model : Model
        Pharmpy model
    n : int
        Number of replicates
    seed : int
        Random seed for the simulation

    Returns
    -------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_simulation(model, n=10, seed=1234)
    >>> steps = model.estimation_steps
    >>> steps[0]
    SimulationStep(n=10, seed=1234, solver=None, solver_rtol=None, solver_atol=None, tool_options=None)

    """
    final_est = None
    for step in model.estimation_steps:
        if isinstance(step, EstimationStep):
            final_est = step
    if final_est is not None:
        step = SimulationStep.create(
            n=n,
            seed=seed,
            solver=final_est.solver,
            solver_atol=final_est.solver_atol,
            solver_rtol=final_est.solver_rtol,
        )
    else:
        step = SimulationStep.create(n=n, seed=seed)
    steps = EstimationSteps((step,))
    model = model.replace(estimation_steps=steps)
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
    add_parameter_uncertainty_step
    remove_parameter_uncertainty_step
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
    add_parameter_uncertainty_step
    remove_parameter_uncertainty_step
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


def add_parameter_uncertainty_step(
    model: Model, parameter_uncertainty_method: Literal['SANDWICH', 'CPG', 'OFIM', 'EFIM']
):
    """Adds parameter uncertainty step to the final estimation step

    Parameters
    ----------
    model : Model
        Pharmpy model
    parameter_uncertainty_method : str
        Parameter uncertainty method to use

    Returns
    -------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_estimation_step(model, 'FOCE', parameter_uncertainty_method=None)
    >>> model = add_parameter_uncertainty_step(model, 'SANDWICH')
    >>> ests = model.estimation_steps
    >>> ests[0]   # doctest: +ELLIPSIS
    EstimationStep('FOCE', interaction=True, parameter_uncertainty_method='SANDWICH', ...)

    See also
    --------
    add_estimation_step
    set_estimation_step
    remove_estimation_step
    append_estimation_step_options
    remove_parameter_uncertainty_step
    set_evaluation_step

    """
    steps = model.estimation_steps
    newstep = steps[-1].replace(parameter_uncertainty_method=f'{parameter_uncertainty_method}')
    newsteps = steps[0:-1] + newstep
    model = model.replace(estimation_steps=newsteps)
    return model.update_source()


def remove_parameter_uncertainty_step(model: Model):
    """Removes parameter uncertainty step from the final estimation step

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
    >>> model = remove_parameter_uncertainty_step(model)
    >>> ests = model.estimation_steps
    >>> ests[0]   # doctest: +ELLIPSIS
    EstimationStep('FOCE', interaction=True, parameter_uncertainty_method=None, ...)

    See also
    --------
    add_estimation_step
    set_estimation_step
    remove_estimation_step
    append_estimation_step_options
    add_parameter_uncertainty_step
    set_evaluation_step

    """
    steps = model.estimation_steps
    newstep = steps[-1].replace(parameter_uncertainty_method=None)
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
        Index of estimation step, default is -1 (last estimation step)

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
    EstimationStep('FOCE', interaction=True, parameter_uncertainty_method='SANDWICH', evaluation=True, ...

    See also
    --------
    set_estimation_step
    add_estimation_step
    remove_estimation_step
    append_estimation_step_options
    add_parameter_uncertainty_step
    remove_parameter_uncertainty_step

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
