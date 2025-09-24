from pharmpy.basic import Expr
from pharmpy.model import Assignment, Model, Statements
from pharmpy.modeling import (
    cholesky_decompose,
    remove_parameter_uncertainty_step,
    remove_unused_parameters_and_rvs,
    set_evaluation_step,
    set_initial_estimates,
)
from pharmpy.workflows import ModelEntry


def prepare_evaluation_model(me: ModelEntry) -> ModelEntry:
    model = me.model
    df = model.dataset
    df = df[df['FREMTYPE'] == 0]
    df = df.reset_index()
    model = model.replace(dataset=df)
    model = set_evaluation_step(model)
    model = remove_parameter_uncertainty_step(model)
    results = me.modelfit_results
    model = set_initial_estimates(model, results.parameter_estimates)
    return ModelEntry.create(model=model)


def _remove_frem_code(model: Model) -> Statements:
    kept_statements = [
        s
        for s in model.statements
        if not isinstance(s, Assignment)
        or (
            not s.symbol.name.startswith("SDC")
            and Expr.symbol("FREMTYPE") not in s.expression.free_symbols
        )
    ]
    return Statements.create(kept_statements)


def _remove_frem_epsilon(model):
    rvs = model.random_variables
    rvs = rvs[:-1]
    return rvs


def prepare_frem_model(me: ModelEntry) -> ModelEntry:
    model = me.model
    kept_statements = _remove_frem_code(model)
    print(kept_statements)
    kept_rvs = _remove_frem_epsilon(model)

    model = model.replace(statements=kept_statements, random_variables=kept_rvs)
    model = remove_unused_parameters_and_rvs(model)
    frem_etas = model.random_variables[-1].names
    model = cholesky_decompose(model, frem_etas)

    return ModelEntry.create(model=model)
