from __future__ import annotations

from itertools import product
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence, Union

from pharmpy import DEFAULT_SEED
from pharmpy.basic import Expr
from pharmpy.internals.immutable import frozenmapping
from pharmpy.model import EstimationStep, ExecutionSteps, Model, SimulationStep
from pharmpy.modeling.help_functions import _as_integer

MethodType = Literal['FO', 'FOCE', 'ITS', 'LAPLACE', 'IMPMAP', 'IMP', 'SAEM', 'BAYES']


def set_estimation_step(model: Model, method: MethodType, idx: int = 0, **kwargs):
    """Set estimation step

    Sets estimation step for a model. Methods currently supported are:
        FO, FOCE, ITS, LAPLACE, IMPMAP, IMP, SAEM, BAYES

    Parameters
    ----------
    model : Model
        Pharmpy model
    method : MethodType
        estimation method to change to
    idx : int
        index of estimation step, default is 0 (first estimation step)
    kwargs
        Arguments to pass to EstimationStep (such as interaction, evaluation)

    Returns
    -------
    Model
        Updated Pharmpy model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> opts = {'NITER': 1000, 'ISAMPLE': 100}
    >>> model = set_estimation_step(model, 'IMP', evaluation=True, tool_options=opts)
    >>> model.execution_steps[0]   # doctest: +ELLIPSIS
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
    if 'evaluation' in d:
        d['maximum_evaluations'] = None
    steps = model.execution_steps
    newstep = steps[idx].replace(**d)
    newsteps = steps[0:idx] + newstep + steps[idx + 1 :]
    model = model.replace(execution_steps=newsteps)
    return model.update_source()


def add_estimation_step(
    model: Model,
    method: MethodType,
    idx: Optional[int] = None,
    interaction: bool = False,
    parameter_uncertainty_method: Optional[str] = None,
    evaluation: bool = False,
    maximum_evaluations: Optional[int] = None,
    laplace: bool = False,
    isample: Optional[int] = None,
    niter: Optional[int] = None,
    auto: Optional[bool] = None,
    keep_every_nth_iter: Optional[int] = None,
    residuals: Sequence[str] = (),
    predictions: Sequence[str] = (),
    solver: Optional[str] = None,
    solver_rtol: Optional[int] = None,
    solver_atol: Optional[int] = None,
    tool_options: Mapping[str, Any] = frozenmapping({}),
    derivatives: Sequence[Sequence[Expr]] = (),
    individual_eta_samples: bool = False,
):
    """Add estimation step

    Adds estimation step for a model in a given index. Methods currently supported are:
        FO, FOCE, ITS, LAPLACE, IMPMAP, IMP, SAEM

    Parameters
    ----------
    model : Model
        Pharmpy model
    method : MethodType
        estimation method to change to
    idx : int
        index of estimation step (starting from 0), default is None (adds step at the end)
    interaction : bool
        See :class:`~pharmpy.model.EstimationStep` for more information on options
    parameter_uncertainty_method : Optional[str]
        See above
    evaluation : bool
        See above
    maximum_evaluations: Optional[int]
        See above
    laplace : bool
        See above
    isample : Optional[int]
        See above
    niter : Optional[int]
        See above
    auto : Optional[bool]
        See above
    keep_every_nth_iter : Optional[int]
        See above
    residuals : Sequence[str]
        See above
    predictions : Sequence[str]
        See above
    solver : Optional[str]
        See above
    solver_rtol : Optional[int]
        See above
    solver_atol : Optional[int]
        See above
    tool_options : Mapping[str, Any]
        See above
    derivatives : Sequence[Sequence[Expr]]
        See above
    individual_eta_samples : bool
        See above

    Returns
    -------
    Model
        Updated Pharmpy model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> opts = {'NITER': 1000, 'ISAMPLE': 100}
    >>> model = add_estimation_step(model, 'IMP', tool_options=opts)
    >>> ests = model.execution_steps
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
    meth = EstimationStep.create(
        method,
        interaction=interaction,
        parameter_uncertainty_method=parameter_uncertainty_method,
        evaluation=evaluation,
        maximum_evaluations=maximum_evaluations,
        laplace=laplace,
        isample=isample,
        niter=niter,
        auto=auto,
        keep_every_nth_iter=keep_every_nth_iter,
        residuals=residuals,
        predictions=predictions,
        solver=solver,
        solver_rtol=solver_rtol,
        solver_atol=solver_atol,
        tool_options=tool_options,
        derivatives=derivatives,
        individual_eta_samples=individual_eta_samples,
    )

    if idx is not None:
        try:
            idx = _as_integer(idx)
        except TypeError:
            raise TypeError(f'Index must be integer: {idx}')
        steps = model.execution_steps
        newsteps = steps[0:idx] + meth + steps[idx:]
    else:
        newsteps = model.execution_steps + meth
    model = model.replace(execution_steps=newsteps)

    return model.update_source()


def set_simulation(model: Model, n: int = 1, seed: int = DEFAULT_SEED):
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
        Update Pharmpy model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_simulation(model, n=10, seed=1234)
    >>> steps = model.execution_steps
    >>> steps[0]
    SimulationStep(n=10, seed=1234, solver=None, solver_rtol=None, solver_atol=None, tool_options={})

    """
    final_est = None
    for step in model.execution_steps:
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
    steps = ExecutionSteps((step,))
    model = model.replace(execution_steps=steps)
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
        Update Pharmpy model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = remove_estimation_step(model, 0)
    >>> ests = model.execution_steps
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

    steps = model.execution_steps
    newsteps = steps[0:idx] + steps[idx + 1 :]
    model = model.replace(execution_steps=newsteps)
    return model.update_source()


def append_estimation_step_options(model: Model, tool_options: Mapping[str, Any], idx: int = -1):
    """Append estimation step options

    Appends options to an existing estimation step.

    Parameters
    ----------
    model : Model
        Pharmpy model
    tool_options : dict
        any additional tool specific options
    idx : int
        index of estimation step (starting from 0). Default is the last step

    Returns
    -------
    Model
        Updated Pharmpy model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> opts = {'NITER': 1000, 'ISAMPLE': 100}
    >>> model = append_estimation_step_options(model, tool_options=opts, idx=0)
    >>> est = model.execution_steps[0]
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

    steps = model.execution_steps
    if len(steps) == 0:
        raise ValueError("No execution steps in the model")
    toolopts = dict(steps[idx].tool_options)
    toolopts.update(tool_options)
    newstep = steps[idx].replace(tool_options=toolopts)
    if idx == -1:
        # Special case so that the tail doesn't become steps[0:]
        idx = idx + len(steps)
    newsteps = steps[0:idx] + newstep + steps[idx + 1 :]
    model = model.replace(execution_steps=newsteps)
    return model.update_source()


def add_parameter_uncertainty_step(
    model: Model, parameter_uncertainty_method: Literal['SANDWICH', 'SMAT', 'RMAT', 'EFIM']
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
        Update Pharmpy model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_estimation_step(model, 'FOCE', parameter_uncertainty_method=None)
    >>> model = add_parameter_uncertainty_step(model, 'SANDWICH')
    >>> ests = model.execution_steps
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
    steps = model.execution_steps
    newstep = steps[-1].replace(parameter_uncertainty_method=f'{parameter_uncertainty_method}')
    newsteps = steps[0:-1] + newstep
    model = model.replace(execution_steps=newsteps)
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
    >>> ests = model.execution_steps
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
    steps = model.execution_steps
    newstep = steps[-1].replace(parameter_uncertainty_method=None)
    newsteps = steps[:-1] + newstep
    model = model.replace(execution_steps=newsteps)
    return model.update_source()


def set_evaluation_step(model: Model, idx: int = -1):
    """Set evaluation step

    Change the final or the estimation step with a specific index to do evaulation.

    Parameters
    ----------
    model : Model
        Pharmpy model
    idx : int
        Index of estimation step, default is -1 (last estimation step)

    Returns
    -------
    Model
        Update Pharmpy model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_evaluation_step(model)
    >>> model.execution_steps[0]   # doctest: +ELLIPSIS
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

    steps = model.execution_steps
    newstep = steps[idx].replace(evaluation=True, maximum_evaluations=None)
    if idx != -1:
        newsteps = steps[0:idx] + newstep + steps[idx + 1 :]
    else:
        newsteps = steps[0:-1] + newstep
    model = model.replace(execution_steps=newsteps)
    return model.update_source()


def add_derivative(
    model: Model, with_respect_to: Optional[Union[Sequence[Union[Sequence[str], str]], str]] = None
):
    """
    Add a derivative to be calculcated when running the model. Currently, only
    derivatives with respect to the prediction is supported. Default is to add all possible
    ETA and EPS derivatives.
    First order derivates are specied either by single string or single-element tuple.
    For instance with_respect_to = "ETA_1" or with_respect_to = ("ETA_1",)

    Second order derivatives are specified by giving the two independent varibles in a tuple
    of tuples. For instance with_respect_to ((ETA_1, EPS_1),)

    Multiple derivatives can be specified within a tuple. For instance ((ETA_1, EPS_1), "ETA_1")

    Currently, only ETAs and EPSILONs are supported

    Parameters
    ----------
    model : Model
        Pharmpy model
    with_respect_to : Union[str, Sequence[str, Sequence[str]]]
        Parameter name(s) to use as independent variables. Default is None.

    Returns
    -------
    Model
        Updated Pharmpy model

    """

    if with_respect_to is None:
        etas = model.random_variables.etas.symbols
        eps = model.random_variables.epsilons.symbols
        wrt = tuple(
            tuple((e,) for e in etas) + tuple((e,) for e in eps) + tuple(product(eps, etas))
        )

    elif isinstance(with_respect_to, str):
        wrt = ((Expr.symbol(with_respect_to),),)
    elif isinstance(with_respect_to, Sequence):
        wrt = tuple(
            (Expr.symbol(w),) if isinstance(w, str) else tuple(map(Expr.symbol, w))
            for w in with_respect_to
        )
    else:
        raise ValueError(
            f"with_respect_to argument need to be tuple, str or None. Recieved {type(with_respect_to)}"
        )

    for derivative in wrt:
        for parameter in derivative:
            if parameter not in model.random_variables.symbols:
                raise ValueError(f"Parameter '{parameter}' not part of ETAs or EPS")

    steps = model.execution_steps
    new_step = steps[-1]

    for new_derivative in wrt:
        derivatives = tuple(set(d) for d in new_step.derivatives)

        if set(new_derivative) not in derivatives:
            new_step = new_step.replace(derivatives=derivatives + (new_derivative,))
    new_steps = steps[:-1] + new_step
    model = model.replace(execution_steps=new_steps)

    return model.update_source()


def remove_derivative(
    model: Model, with_respect_to: Optional[Union[Sequence[Union[Sequence[str], str]], str]] = None
):
    """
    Remove a derivative currently being calculcate when running model. Currently, only
    derivatives with respect to the prediction is supported. Default is to remove all
    that are present,
    First order derivates are specied either by single string or single-element tuple.
    For instance with_respect_to = "ETA_1" or with_respect_to = ("ETA_1",)

    Second order derivatives are specified by giving the two independent varibles in a tuple
    of tuples. For instance with_respect_to ((ETA_1, EPS_1),)

    Multiple derivatives can be specified within a tuple. For instance ((ETA_1, EPS_1), "ETA_1")

    Currently, only ETAs and EPSILONs are supported

    Parameters
    ----------
    model : Model
        Pharmpy model
    with_respect_to : Union[str, Sequence[str, Sequence[str]]]
        Parameter name(s) to use as independent variables. Default is None.

    Returns
    -------
    Model
        Updated Pharmpy model

    """

    if with_respect_to is None:
        etas = model.random_variables.etas.symbols
        eps = model.random_variables.epsilons.symbols
        wrt = tuple(
            tuple((e,) for e in etas) + tuple((e,) for e in eps) + tuple(product(eps, etas))
        )

    elif isinstance(with_respect_to, str):
        wrt = ((Expr.symbol(with_respect_to),),)
    elif isinstance(with_respect_to, Sequence):
        wrt = tuple(
            (Expr.symbol(w),) if isinstance(w, str) else tuple(map(Expr.symbol, w))
            for w in with_respect_to
        )
    else:
        raise ValueError(
            f"with_respect_to argument need to be tuple, str or None. Recieved {type(with_respect_to)}"
        )

    wrt = tuple(set(d) for d in wrt)
    new_steps = model.execution_steps
    derivatives = new_steps[-1].derivatives

    derivatives = tuple(d for d in derivatives if set(d) not in wrt)
    new_step = new_steps[-1].replace(derivatives=derivatives)
    new_steps = new_steps[:-1] + new_step

    model = model.replace(execution_steps=new_steps)

    return model.update_source()


def add_predictions(model: Model, pred: list[str]):
    """Add predictions and/or residuals

    Add predictions to estimation step.

    Parameters
    ----------
    model : Model
        Pharmpy model
    pred : list
        List of predictions (e.g. ['IPRED', 'PRED'])

    Returns
    -------
    Model
        Updated Pharmpy model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model.execution_steps[-1].predictions
    ('CIPREDI', 'PRED')
    >>> model = add_predictions(model, ['IPRED'])
    >>> model.execution_steps[-1].predictions
    ('CIPREDI', 'IPRED', 'PRED')

    See also
    --------
    remove_predictions
    remove_residuals
    set_estimation_step
    add_estimation_step
    remove_estimation_step
    append_estimation_step_options
    add_parameter_uncertainty_step
    remove_parameter_uncertainty_step
    """
    allowed_prediction_variables = ['PRED', 'IPRED', 'CIPREDI']

    if any(p not in allowed_prediction_variables for p in pred):
        raise ValueError(
            f'Prediction variables need to be one of the following:'
            f' {allowed_prediction_variables}'
        )

    steps = model.execution_steps
    old_predictions = steps[-1].predictions
    new_predictions = tuple(sorted(set(old_predictions) | set(pred)))
    newstep = steps[-1].replace(predictions=new_predictions)
    newsteps = steps[0:-1] + newstep
    model = model.replace(execution_steps=newsteps)
    return model.update_source()


def add_residuals(model: Model, res: list[str]):
    """Add predictions and/or residuals

    Add residuals to estimation step.

    Added redidual variable(s) need to be one of the following :
        ['RES', 'IRES', 'WRES', 'IWRES', 'CWRES']

    Parameters
    ----------
    model : Model
        Pharmpy model
    res : list
        List of residuals (e.g. ['CWRES'])

    Returns
    -------
    Model
        Updated Pharmpy model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model.execution_steps[-1].residuals
    ('CWRES', 'RES')
    >>> model = add_residuals(model, ['WRES'])
    >>> model.execution_steps[-1].residuals
    ('CWRES', 'RES', 'WRES')

    See also
    --------
    remove_predictions
    remove_residuals
    set_estimation_step
    add_estimation_step
    remove_estimation_step
    append_estimation_step_options
    add_parameter_uncertainty_step
    remove_parameter_uncertainty_step
    """
    allowed_residual_variables = ['RES', 'IRES', 'WRES', 'IWRES', 'CWRES']

    if any(p not in allowed_residual_variables for p in res):
        raise ValueError(
            f'Residual variables need to be one of the following:' f' {allowed_residual_variables}'
        )

    steps = model.execution_steps
    old_residuals = steps[-1].residuals
    new_residuals = tuple(sorted(set(old_residuals) | set(res)))
    newstep = steps[-1].replace(residuals=new_residuals)
    newsteps = steps[0:-1] + newstep
    model = model.replace(execution_steps=newsteps)
    return model.update_source()


def remove_predictions(model: Model, to_remove: Optional[Iterable[str]] = None):
    """Remove predictions and/or residuals

    Remove predictions from estimation step.

    Parameters
    ----------
    model : Model
        Pharmpy model
    to_remove : Optional[Iterable[str]]
        Predictions to remove

    Returns
    -------
    Model
        Updated Pharmpy model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = remove_predictions(model)
    >>> model.execution_steps[-1].predictions
    ()

    See also
    --------
    add_predictions
    add_residuals
    set_estimation_step
    add_estimation_step
    remove_estimation_step
    append_estimation_step_options
    add_parameter_uncertainty_step
    remove_parameter_uncertainty_step
    """
    steps = model.execution_steps
    if to_remove is not None:
        old_predictions = steps[-1].predictions
        for value in to_remove:
            if value not in old_predictions:
                raise ValueError(f"{value} not in predictions")
        new_predictions = tuple(sorted(set(old_predictions) - set(to_remove)))
        newstep = steps[-1].replace(predictions=new_predictions)
    else:
        newstep = steps[-1].replace(predictions=())
    newsteps = steps[0:-1] + newstep
    model = model.replace(execution_steps=newsteps)
    model = model.update_source()
    return model


def remove_residuals(model: Model, to_remove: Optional[Iterable[str]] = None):
    """Remove residuals

    Remove residuals from estimation step.

    Parameters
    ----------
    model : Model
        Pharmpy model
    to_remove : Optional[Iterable[str]]
        Residuals to remove

    Returns
    -------
    Model
        Updated Pharmpy model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = remove_residuals(model)
    >>> model.execution_steps[-1].residuals
    ()

    See also
    --------
    add_predictions
    add_residuals
    set_estimation_step
    add_estimation_step
    remove_estimation_step
    append_estimation_step_options
    add_parameter_uncertainty_step
    remove_parameter_uncertainty_step
    """
    steps = model.execution_steps
    if to_remove is not None:
        old_residuals = steps[-1].residuals
        for value in to_remove:
            if value not in old_residuals:
                raise ValueError(f"{value} not in residuals")
        new_residuals = tuple(sorted(set(old_residuals) - set(to_remove)))
        newstep = steps[-1].replace(residuals=new_residuals)
        newsteps = steps[0:-1] + newstep
    else:
        newstep = steps[-1].replace(residuals=())
    newsteps = steps[0:-1] + newstep
    model = model.replace(execution_steps=newsteps)
    model = model.update_source()
    return model


def is_simulation_model(model: Model) -> bool:
    """Check if a model is a pure simulation model

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    bool
        True if it is a simulation model

    Examples
    --------
    >>> from pharmpy.modeling import is_simulation_model, load_example_model
    >>> model = load_example_model("pheno")
    >>> is_simulation_model(model)
    False

    """
    for step in model.execution_steps:
        if not isinstance(step, SimulationStep):
            return False
    return True
