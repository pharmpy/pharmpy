from __future__ import annotations

from typing import Any, Optional, Union, overload

from pharmpy.basic import BooleanExpr, Expr, Quantity, Unit
from pharmpy.basic.expr import solve
from pharmpy.model import (
    Add,
    Assignment,
    CompartmentalSystem,
    CompartmentalSystemBuilder,
    Drop,
    Model,
    Statements,
    get_and_check_dataset,
    get_and_check_odes,
)


@overload
def get_unit_of(model: Model, variable: None) -> dict[str, Unit]: ...


@overload
def get_unit_of(model: Model, variable: Union[str, Expr]) -> Unit: ...


def get_unit_of(model: Model, variable: Union[str, Expr, None] = None) -> Unit | dict[str, Unit]:
    """Derive the physical unit of a variable in the model

    Unit information for the dataset needs to be available.
    The variable can be defined in the code, a dataset column, a parameter
    or a random variable. Optionally units could be derived for all variables
    in the model.

    Parameters
    ----------
    model : Model
        Pharmpy model
    variable : str | Expr | None
        Find physical unit of this variable. For None get a dict with units for
        all variables defined by the model.

    Returns
    -------
    Unit
        A unit expression

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, get_unit_of
    >>> model = load_example_model("pheno")
    >>> get_unit_of(model, "Y")
    mg/L
    >>> get_unit_of(model, "VC")
    L
    >>> get_unit_of(model, "WGT")
    kg
    """
    if isinstance(variable, str):
        variable = Expr.symbol(variable)

    di = model.datainfo

    # FIXME: No multiple DV-support for now

    # Map from Symbol -> known Unit
    known = {
        col.symbol: col.variable.properties.get("unit", None)
        for col in di
        if not col.variable.properties.get("unit", None) is None
    }
    y = list(model.dependent_variables.keys())[0]
    known[y] = known[di.dv_column.symbol]
    if model.statements.ode_system is not None:
        amount_unit = di.typeix['dose'][0].variable.properties.get("unit", None)
        if amount_unit is not None:
            for amt in model.statements.ode_system.amounts:
                known[amt] = amount_unit
        idv_unit = di.typeix['idv'][0].variable.properties.get("unit", None)
        if idv_unit is not None:
            known[model.statements.ode_system.t] = idv_unit
    else:
        amount_unit = None
        idv_unit = None

    # Set of tuples symbol, expression where units cannot yet be deduced
    unknown = set()

    for s in reversed(model.statements):
        if variable is not None and variable in known:
            return known[variable]

        if isinstance(s, Assignment):
            handle_assignment(s.symbol, s.expression, known, unknown, model)
        elif isinstance(s, CompartmentalSystem):
            eqs = s.eqs
            for eq in eqs:
                func = eq.lhs.args[0]
                assert isinstance(func, Expr)
                funcname = func.name
                # FIXME: Could collide
                derivative_symbol = Expr.symbol(f"d{funcname}_dt")
                if amount_unit is not None and idv_unit is not None:
                    known[derivative_symbol] = amount_unit / idv_unit
                handle_assignment(derivative_symbol, eq.rhs, known, unknown, model)

        unknown = recheck_unknowns(unknown, known, model)

    if variable is not None and variable in known:
        return known[variable]

    if variable is not None:
        raise RuntimeError(f"Couldn't deduct unit for {variable}")
    else:
        all_units = {}
        for symbol in get_all_symbols(model):
            all_units[str(symbol)] = known.get(symbol, None)
        return all_units


def get_all_symbols(model):
    symbols = (
        set(model.parameters.symbols)
        | set(model.random_variables.symbols)
        | model.statements.lhs_symbols
        | set(model.datainfo.symbols)
    )
    if model.statements.ode_system is not None:
        symbols.add(model.statements.ode_system.t)
    return symbols


def product(a, start: Any = 1):
    prod = start
    for e in a:
        prod *= e
    return prod


def simplify_for_units(expr: Expr) -> Expr:
    # Remove known unitless cases: exp, log
    # FIXME: Remove constants other than -1, 1 and 0
    if expr.is_add():
        return sum((simplify_for_units(term) for term in expr.expr_args), start=Expr(0))
    elif expr.is_mul():
        return product((simplify_for_units(factor) for factor in expr.expr_args), start=Expr(1))
    elif expr.is_exp() or (expr.is_function() and expr.name == "log"):
        return Expr(1)
    elif expr.is_function() and expr.name in {"forward", "first"}:
        return simplify_for_units(expr.expr_args[0])
    elif expr.is_function() and expr.name in {"newind", "count_if"}:
        return Expr(1)
    else:
        return expr


def deduct_equal_units(symbol: Expr, expr: Expr) -> list[BooleanExpr]:
    # FIXME: we could also recurse down to additions inside exp and log or parentheses
    eqs = []
    expr = expr.expand()
    if expr.is_add():
        for term in expr.expr_args:
            eqs.append(BooleanExpr.eq(symbol, simplify_for_units(term)))
    elif expr.is_piecewise():
        for piece, _ in expr.piecewise_args:
            if piece != symbol and not piece.is_number():
                eqs.append(BooleanExpr.eq(symbol, simplify_for_units(piece)))
    else:
        eqs.append(BooleanExpr.eq(symbol, simplify_for_units(expr)))
    return eqs


def derive_unit(expr, known):
    if expr.is_number():
        unit = Unit(1)
    elif expr.is_symbol():
        unit = known[expr]
    elif expr.is_mul():
        unit = product([derive_unit(factor, known) for factor in expr.args], start=Unit(1))
    elif expr.is_pow():
        base, exp = expr.args
        if not exp.is_integer():
            raise NotImplementedError("Non integer exponent not implemented for unit deduction")
        unit = derive_unit(base, known) ** int(exp)
    else:
        raise NotImplementedError("Expression not implemented for unit deduction")
    return unit


def used_symbols(expr, model):
    # This is a workardound for free_symbols which doesn't give A_...(t)
    assignment = Assignment(Expr("DUMMY"), expr)
    symbols = assignment.rhs_symbols
    # Handle case where t is only found in amounts. Not needed for deduction
    odes = model.statements.ode_system
    if odes is not None:
        d = {amt: Expr(f"__DUMMY___{i}") for i, amt in enumerate(odes.amounts)}
        if model.statements.ode_system.t not in expr.subs(d).free_symbols:
            symbols -= {model.statements.ode_system.t}
    return symbols


def handle_assignment(symbol, expression, known, unknown, model):
    eqs = deduct_equal_units(symbol, expression)
    sol = solve(eqs, exclude=known.keys())
    for lhs, rhs in sol.items():
        # FIXME: Could also learn the unit of the whole assignment
        # FIXME: Unit of covariance is product of unit of both rvs
        # FIXME: Can get information from conditions in piecewises
        if rhs == 1:
            unit = Unit(1)
        elif used_symbols(rhs, model).issubset(known.keys()):
            unit = derive_unit(rhs, known)
        else:
            unknown.add((lhs, rhs))
            continue

        known[lhs] = unit
        if lhs in model.random_variables:
            rv = model.random_variables[lhs]
            var = rv.variance
            known[var] = unit**2


def recheck_unknowns(unknown, known, model):
    still_unknown = set()
    for symbol, expression in unknown:
        # We need to attempt solving the equation again since
        # the unknown might not be on the lhs
        eq = BooleanExpr.eq(symbol, expression)
        sol = solve([eq], exclude=known.keys())
        sol_symbol, sol_expression = sol.popitem()
        if used_symbols(sol_expression, model).issubset(known.keys()):
            unit = derive_unit(sol_expression, known)
            known[sol_symbol] = unit
        else:
            still_unknown.add((symbol, expression))
    return still_unknown


def convert_unit(
    model: Model,
    variable: str,
    unit: Union[str, Unit],
    original_unit: Optional[Union[str, Unit]] = None,
    in_dataset: bool = False,
) -> Model:
    """Convert between units for a data variable

    The conversion could either be handled in the model code or optionally in the dataset (if applicable).

    Parameters
    ----------
    model : Model
        Pharmpy model
    variable : str
        Which variable in the dataset or the model code to convert
    unit : str
        The new unit
    original_unit : str
        If no original unit is available in the datainfo this will be used
    in_dataset : bool
        Set to True if the conversion should be done in the dataset instead of in model code

    Returns
    -------
    Model
        Updated Pharmpy model

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, convert_unit
    >>> model = load_example_model("pheno")
    >>> model = convert_unit(model, "WGT", "g")
    """

    unit = Unit(unit)
    if original_unit is None and variable in model.datainfo:
        original_unit = model.datainfo[variable].variable.properties.get("unit", None)
    if original_unit is None:
        raise ValueError("Cannot find the original unit of {variable}")
    original_unit = Unit(original_unit)

    if original_unit == unit:
        return model

    if not original_unit.is_compatible_with(unit):
        raise ValueError(f"Unable to convert from {original_unit} to {unit}: different dimensions.")

    conversion_factor = Quantity(1.0, original_unit).convert_to(unit).value
    conversion_factor = (
        int(conversion_factor) if int(conversion_factor) == conversion_factor else conversion_factor
    )

    if not in_dataset:
        column = model.datainfo[variable]
        if column.type in {'dose', 'dv'}:
            odes = get_and_check_odes(model)
            dosing_cmts = odes.dosing_compartments
            cb = CompartmentalSystemBuilder(odes)
            cb.set_bioavailability(
                dosing_cmts[0], dosing_cmts[0].bioavailability * conversion_factor
            )
            new_statements = (
                model.statements.before_odes + CompartmentalSystem(cb) + model.statements.after_odes
            )
        else:
            original_symbol = Expr.symbol(variable)
            scaled_symbol = Expr.symbol(f"SCALED_{variable}")
            expr = conversion_factor * original_symbol
            assignment = Assignment.create(scaled_symbol, expr)
            new_statements = assignment + Statements.create(
                [s.subs({original_symbol: scaled_symbol}) for s in model.statements]
            )
        model = model.replace(statements=new_statements)
    else:
        df = get_and_check_dataset(model)
        scaled_column = conversion_factor * df[variable]
        df = df.assign(**{variable: scaled_column})
        new_var = model.datainfo[variable].variable.set_property("unit", unit)
        new_col = model.datainfo[variable].replace(variable_mapping=new_var)
        new_di = model.datainfo.set_column(new_col)
        prov_new = [Drop.create(variable), Add.create(variable)]
        new_di = new_di.replace(provenance=new_di.provenance + prov_new)
        model = model.replace(dataset=df, datainfo=new_di)
    return model.update_source()
