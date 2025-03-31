from abc import ABC, abstractmethod

from pharmpy.basic import Expr
from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.deps import sympy


def get_variables_before_odes(model) -> dict[Expr, Expr]:
    # Get a dict of symbol to full expression
    # for all variables needed to solve the ODE system
    statements = model.statements.before_odes
    defined_variables = {s.symbol for s in statements}
    required_symbols = (
        model.statements.ode_system.free_symbols | model.statements.after_odes.free_symbols
    )
    required_variables = defined_variables.intersection(required_symbols)
    d = {}
    statements = statements.subs(model.parameters.inits)
    for var in required_variables:
        expr = statements.full_expression(var)
        d[var] = expr
    return d


def get_functions_to_solve_for(model) -> set[Expr]:
    # Get the set of functions that we need to solve for in the ode system
    all_funcs = set(model.statements.ode_system.amounts)
    all_deps = set()
    for s in model.statements.after_odes:
        all_deps |= s.rhs_symbols
    needed_funcs = all_funcs.intersection(all_deps)
    return needed_funcs


def evaluate_variables_before_odes(model):
    # Evaluate all parameters needed to solve the
    # odesystem. Return a dataframe with one variable
    # per column for all datarecords
    exprs = get_variables_before_odes(model)
    dependencies = set()
    for expr in exprs.values():
        dependencies |= expr.free_symbols
    random_symbols = set(model.random_variables.symbols)
    data_deps = dependencies - random_symbols
    random_deps = dependencies.intersection(random_symbols)

    # For now zero out etas
    subs = {rv: 0 for rv in random_deps}
    exprs = {symb: expr.subs(subs) for symb, expr in exprs.items()}

    df = model.dataset.loc[:, [str(var) for var in data_deps]]

    evaluator = SymengineSubsEvaluator()

    res = evaluator.evaluate_expressions(exprs, df)
    return res


def prepare_data_for_odes(model, data):
    idname = model.datainfo.id_column.name
    idvname = model.datainfo.idv_column.name
    amtname = model.datainfo.typeix['dose'][0].name
    dt = model.dataset.groupby('ID')[idvname].diff().fillna(0.0)
    df = data.assign(ID=model.dataset[idname], AMT=model.dataset[amtname], t=dt)
    df.columns = [col if isinstance(col, Expr) else Expr.symbol(col) for col in df.columns]
    return df


def evaluate_variables_after_odes(model, data):
    evaluator = SymengineSubsEvaluator()

    # FIXME: Name of DV and multiple DVs
    expr = model.statements.after_odes.full_expression('Y')
    # FIXME: Could be etas here
    subs = {rv: 0 for rv in model.random_variables.epsilons.symbols}
    expr = expr.subs(subs)
    res = evaluator.evaluate_expressions({Expr.symbol('Y'): expr}, data)
    return res


def evaluate_model(model):
    df = evaluate_variables_before_odes(model)
    df = prepare_data_for_odes(model, df)
    solvefor = get_functions_to_solve_for(model)

    ode_solver = SymbolicSolver()
    sol = ode_solver.solve_ode_system(model.statements.ode_system.eqs, df, solvefor)
    df = pd.concat((df, sol), axis=1)
    res = evaluate_variables_after_odes(model, df)
    res.columns = [str(col) for col in res.columns]
    return res


class ExpressionEvaluator(ABC):
    @abstractmethod
    def evaluate_expressions(self, expressions, data):
        pass


class SymengineSubsEvaluator(ExpressionEvaluator):
    def evaluate_expressions(self, expressions, data):
        def fn(row):
            d = {symb: float(expr.subs(dict(row))) for symb, expr in expressions.items()}
            return pd.Series(d)

        res = data.apply(fn, axis=1)
        return res


class ODESolver(ABC):
    @abstractmethod
    def solve_ode_system(self, eqs, data, solvefor):
        pass


class SymbolicSolver(ODESolver):
    def solve_ode_system(self, eqs, data, solvefor):
        # FIXME: No initial conditions. Also remember that 0 cannot be used
        # FIXME: Should set assumptions on symbols before solving
        # FIXME: Need a way to find and handle systems with no explicit solutions
        solvefor = {sympy.sympify(s) for s in solvefor}
        sol = sympy.dsolve(sympy.sympify(eqs))
        sol = [eq for eq in sol if eq.args[0] in solvefor]
        # FIXME: This assumes C*exp(..) and splits out the exp part
        exprs = {eq.args[0]: eq.args[1].args[1] for eq in sol}

        evaluator = SymengineSubsEvaluator()
        df = evaluator.evaluate_expressions(exprs, data)

        amounts = data[Expr.symbol('AMT')]
        new = np.empty(len(amounts))
        curind = None
        for i, (ind, amt, target) in enumerate(
            zip(data[Expr.symbol('ID')], amounts, df.iloc[:, 0])
        ):
            if ind != curind:
                curind = ind
                new[i] = amt
            else:
                new[i] = amt + new[i - 1] * target
        sol = pd.DataFrame({solvefor.pop(): new})
        return sol
