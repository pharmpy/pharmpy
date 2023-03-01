from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Set, Tuple, TypeVar, Union

from pharmpy.deps import sympy
from pharmpy.internals.expr.subs import subs
from pharmpy.internals.expr.tree import prune
from pharmpy.internals.graph.directed.reachability import reachable_from
from pharmpy.model import Assignment, Model, ODESystem

T = TypeVar('T')


def _extract_minus(expr):
    if expr.is_Mul and expr.args[0] == -1:
        return sympy.Mul(*expr.args[1:])
    else:
        return expr


def get_unit_of(model: Model, variable: Union[str, sympy.Symbol]):
    """Derive the physical unit of a variable in the model

    Unit information for the dataset needs to be available.
    The variable can be defined in the code, a dataset olumn, a parameter
    or a random variable.

    Parameters
    ----------
    model : Model
        Pharmpy model object
    variable : str or Symbol
        Find physical unit of this variable

    Returns
    -------
    unit expression
        A sympy physics.units expression

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, get_unit_of
    >>> model = load_example_model("pheno")
    >>> get_unit_of(model, "Y")
    milligram/liter
    >>> get_unit_of(model, "V")
    liter
    >>> get_unit_of(model, "WGT")
    kilogram
    """
    if isinstance(variable, str):
        symbol = sympy.Symbol(variable)
    else:
        symbol = variable
        variable = variable.name

    di = model.datainfo
    if variable in di.names:
        return di[variable].unit

    # FIXME: handle other DVs?
    y = list(model.dependent_variables.keys())[0]
    input_units = {sympy.Symbol(col.name): col.unit for col in di}
    pruned_nodes = {sympy.exp}

    def pruning_predicate(e: sympy.Expr) -> bool:
        return e.func in pruned_nodes

    unit_eqs = []
    unit_eqs.append(y - di[di.dv_column.name].unit)

    for s in model.statements:
        if isinstance(s, Assignment):
            expr = sympy.expand(
                subs(prune(pruning_predicate, s.expression), input_units, simultaneous=True)
            )
            if expr.is_Add:
                for term in expr.args:
                    unit_eqs.append(s.symbol - _extract_minus(term))
            else:
                unit_eqs.append(s.symbol - _extract_minus(expr))
        elif isinstance(s, ODESystem):
            amt_unit = di[di.typeix['dose'][0].name].unit
            time_unit = di[di.idv_column.name].unit
            for e in s.compartmental_matrix.diagonal():
                if e.is_Add:
                    for term in e.args:
                        unit_eqs.append(amt_unit / time_unit - _extract_minus(term))
                elif e == 0:
                    pass
                else:
                    unit_eqs.append(amt_unit / time_unit - _extract_minus(e))
            for a in s.amounts:
                unit_eqs.append(amt_unit - a)

    # NOTE This keeps only the equations required to solve for "symbol"
    filtered_unit_eqs = _filter_equations(unit_eqs, symbol)
    # NOTE For some reason telling sympy to solve for "symbol" does not work
    sol = sympy.solve(filtered_unit_eqs, dict=True)
    return sol[0][symbol]


def _filter_equations(
    equations: Iterable[sympy.Expr], symbol: sympy.Symbol
) -> Iterable[sympy.Expr]:
    # NOTE This has the side-effect of deduplicating equations
    fs = {eq: eq.free_symbols for eq in equations}

    # NOTE We could first contract clique edges but I have not found a way to
    # make it as elegant as the current implementation
    edges = _cliques_spanning_forest_edges_linear_superset(fs.values())

    graph = _adjacency_list(edges)

    dependent_symbols = reachable_from(
        {symbol},
        graph.__getitem__,
    )

    # NOTE All symbols are in the same connected component so we only need to
    # test one symbol for each equation
    return (
        eq for eq, symbols in fs.items() if symbols and next(iter(symbols)) in dependent_symbols
    )


def _adjacency_list(edges: Iterable[Tuple[T, T]]) -> Dict[T, Set[T]]:
    graph = defaultdict(set)
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)

    return graph


def _cliques_spanning_forest_edges_linear_superset(
    cliques: Iterable[Iterable[T]],
) -> Iterable[Tuple[T, T]]:
    # NOTE This is not a forest but it has a linear number of edges in the
    # input size. Building a spanning tree would require a union-find data
    # structure and superlinear time, which is unnecessary here since we are
    # only interested in connected components of the graph.
    for clique in cliques:
        yield from _clique_spanning_tree_edges(clique)


def _clique_spanning_tree_edges(clique: Iterable[T]) -> Iterable[Tuple[T, T]]:
    it = iter(clique)
    try:
        u = next(it)
    except StopIteration:
        return
    for v in it:
        yield (u, v)
