from itertools import chain, combinations
from typing import Callable, Iterable, List, Tuple, TypeVar, Union

from sympy.core.add import Add
from sympy.core.symbol import Symbol
from sympy.functions.elementary.piecewise import Piecewise

import pharmpy.tools.modelsearch.tool
from pharmpy import Model
from pharmpy.modeling import add_iov, copy_model, rank_models, remove_iiv, remove_iov
from pharmpy.statements import Assignment
from pharmpy.tools.common import create_results, update_initial_estimates
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import Task, Workflow, call_workflow

NAME_WF = 'iovsearch'

T = TypeVar('T')


def create_workflow(
    column='OCC',
    list_of_parameters=None,
    rank_type='bic',
    cutoff=None,
    distribution='same-as-iiv',
    model=None,
):
    """Run IOVsearch tool. For more details, see :ref:`iovsearch`.

    Parameters
    ----------
    column : str
        Name of column in dataset to use as occasion column (default is 'OCC')
    list_of_parameters : list
        List of parameters to test IOV on, if none all parameters with IIV will be tested (default)
    rank_type : str
        Which ranking type should be used (OFV, AIC, BIC). Default is BIC
    cutoff : None or float
        Cutoff for which value of the ranking type that is considered significant. Default
        is None (all models will be ranked)
    distribution : str
        Which distribution added IOVs should have (default is same-as-iiv)
    model : Model
        Pharmpy model

    Returns
    -------
    IOVSearchResults
        IOVSearch tool result object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> run_iovsearch('OCC', model=model)      # doctest: +SKIP
    """
    wf = Workflow()
    wf.name = NAME_WF

    init_task = init(model)
    wf.add_task(init_task)

    bic_type = 'random'
    search_task = Task(
        'search',
        task_brute_force_search,
        column,
        list_of_parameters,
        rank_type,
        cutoff,
        bic_type,
        distribution,
    )

    wf.add_task(search_task, predecessors=init_task)
    search_output = wf.output_tasks

    results_task = Task(
        'results',
        task_results,
        rank_type,
        cutoff,
        bic_type,
    )

    wf.add_task(results_task, predecessors=search_output)

    return wf


def init(model):
    return (
        Task('init', lambda model: model)
        if model is None
        else Task('init', lambda model: model, model)
    )


def task_brute_force_search(
    occ: str,
    list_of_parameters: Union[None, list],
    rank_type: str,
    cutoff: Union[None, float],
    bic_type: Union[None, str],
    distribution: str,
    model: Model,
):

    # NOTE Default is to try all IIV ETAs.
    if list_of_parameters is None:
        iiv = model.random_variables.iiv
        all_iiv_parameters = list(map(lambda rv: rv.name, iiv))
        list_of_parameters = all_iiv_parameters

    # NOTE Check that model has at least one IIV.
    if not list_of_parameters:
        return [model]

    # NOTE Add IOVs on given parameters or all parameters with IIVs.
    model_with_iov = copy_model(model, name='iovsearch_candidate1')
    model_with_iov.parent_model = model.name
    names = [name for name in list_of_parameters]
    model_with_iov.description = f'add_iov({",".join(names)})'
    update_initial_estimates(model_with_iov)
    # TODO should we exclude already present IOVs?
    add_iov(model_with_iov, occ, list_of_parameters, distribution=distribution)
    # NOTE Fit the new model.
    wf = create_fit_workflow(models=[model_with_iov])
    model_with_iov = call_workflow(wf, f'{NAME_WF}-fit-with-matching-IOVs')

    # NOTE Remove IOVs. Test all subsets (~2^n).
    # TODO should we exclude already present IOVs?
    iov = model_with_iov.random_variables.iov
    # NOTE We only need to remove the IOV ETA corresponding to the first
    # category in order to remove all IOV ETAs of the other categories
    all_iov_parameters = list(
        filter(lambda name: name.endswith('_1'), map(lambda rv: rv.name, iov))
    )
    no_of_models = 1
    wf = wf_etas_removal(
        remove_iov, model_with_iov, non_empty_proper_subsets(all_iov_parameters), no_of_models + 1
    )
    iov_candidates = call_workflow(wf, f'{NAME_WF}-fit-with-removed-IOVs')

    # NOTE Keep best candidate.
    best_model_so_far = best_model(
        model,
        [model_with_iov, *iov_candidates],
        rank_type=rank_type,
        cutoff=cutoff,
        bic_type=bic_type,
    )

    # NOTE If no improvement with respect to input model, STOP.
    if best_model_so_far is model:
        return [model, model_with_iov, *iov_candidates]

    # NOTE Remove IIV with corresponding IOVs. Test all subsets (~2^n).
    iiv_parameters_with_associated_iov = list(
        map(
            lambda s: s.name,
            _get_iiv_etas_with_corresponding_iov(best_model_so_far),
        )
    )
    # TODO should we exclude already present IOVs?
    no_of_models = len(iov_candidates) + 1
    wf = wf_etas_removal(
        remove_iiv,
        best_model_so_far,
        non_empty_subsets(iiv_parameters_with_associated_iov),
        no_of_models + 1,
    )
    iiv_candidates = call_workflow(wf, f'{NAME_WF}-fit-with-removed-IIVs')

    return [model, model_with_iov, *iov_candidates, *iiv_candidates]


def task_remove_etas_subset(
    remove: Callable[[Model, List[str]], None], model: Model, subset: List[str], n: int
):
    model_with_some_etas_removed = copy_model(model, name=f'iovsearch_candidate{n}')
    model_with_some_etas_removed.parent_model = model.name
    names = [name for name in subset]
    model_with_some_etas_removed.description = f'{remove.__name__}({",".join(names)})'
    update_initial_estimates(model_with_some_etas_removed)
    remove(model_with_some_etas_removed, subset)
    return model_with_some_etas_removed


def wf_etas_removal(
    remove: Callable[[Model, List[str]], None],
    model: Model,
    etas_subsets: Iterable[Tuple[str]],
    n: float,
):
    wf = Workflow()
    for i, subset_of_iiv_parameters in enumerate(etas_subsets):
        task = Task(
            repr(subset_of_iiv_parameters),
            task_remove_etas_subset,
            remove,
            model,
            list(subset_of_iiv_parameters),
            n,
        )
        wf.add_task(task)
        n += 1

    wf_fit = create_fit_workflow(n=i + 1)
    wf.insert_workflow(wf_fit)

    task_gather = Task('gather', lambda *models: models)
    wf.add_task(task_gather, predecessors=wf.output_tasks)
    return wf


def best_model(
    base: Model,
    models: List[Model],
    rank_type: str,
    cutoff: Union[None, float],
    bic_type: Union[None, str],
):
    candidates = [base, *models]
    _, srtd = rank_models(base, candidates, rank_type=rank_type, cutoff=cutoff, bic_type=bic_type)
    if srtd:
        return srtd[0]
    else:
        return base


def subsets(iterable: Iterable[T], min_size: int = 0, max_size: int = -1) -> Iterable[Tuple[T]]:
    """Returns an iterable over all the subsets of the input iterable with
    minimum and maximum size constraints. Allows maximum_size to be given
    relatively to iterable "length" by specifying a negative value.

    Adapted from powerset function defined in
    https://docs.python.org/3/library/itertools.html#itertools-recipes

    subsets([1,2,3], min_size=1, max_size=2) --> (1,) (2,) (3,) (1,2) (1,3) (2,3)"
    """
    s = list(iterable)
    max_size = len(s) + max_size + 1 if max_size < 0 else max_size
    return chain.from_iterable(combinations(s, r) for r in range(min_size, max_size + 1))


def non_empty_proper_subsets(iterable: Iterable[T]) -> Iterable[Tuple[T]]:
    """Returns an iterable over all the non-empty proper subsets of the input
    iterable.

    non_empty_proper_subsets([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3)"
    """
    return subsets(iterable, min_size=1, max_size=-2)


def non_empty_subsets(iterable: Iterable[T]) -> Iterable[Tuple[T]]:
    """Returns an iterable over all the non-empty subsets of the input
    iterable.

    non_empty_subsets([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    """
    return subsets(iterable, min_size=1, max_size=-1)


def task_results(rank_type, cutoff, bic_type, models):
    base_model, *res_models = models

    res = create_results(
        IOVSearchResults, base_model, base_model, res_models, rank_type, cutoff, bic_type=bic_type
    )

    return res


class IOVSearchResults(pharmpy.results.Results):
    def __init__(
        self,
        summary_tool=None,
        summary_models=None,
        summary_individuals=None,
        summary_individuals_count=None,
        summary_errors=None,
        best_model=None,
        input_model=None,
        models=None,
    ):
        self.summary_tool = summary_tool
        self.summary_models = summary_models
        self.summary_individuals = summary_individuals
        self.summary_individuals_count = summary_individuals_count
        self.summary_errors = summary_errors
        self.best_model = best_model
        self.input_model = input_model
        self.models = models


def _get_iov_piecewise_assignment_symbols(model: Model):
    iovs = set(rv.symbol for rv in model.random_variables.iov)
    for statement in model.statements:
        if isinstance(statement, Assignment) and isinstance(statement.expression, Piecewise):
            expression_symbols = [p[0] for p in statement.expression.as_expr_set_pairs()]
            if all(s in iovs for s in expression_symbols):
                yield statement.symbol


def _get_iiv_etas_with_corresponding_iov(model: Model):
    iovs = set(_get_iov_piecewise_assignment_symbols(model))
    iiv = model.random_variables.iiv

    for statement in model.statements:
        if isinstance(statement, Assignment) and isinstance(statement.expression, Add):
            for symbol in statement.expression.free_symbols:
                if symbol in iovs:
                    rest = statement.expression - symbol
                    if isinstance(rest, Symbol) and rest in iiv:
                        yield rest
                    break
