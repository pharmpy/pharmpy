from pharmpy.basic import Expr
from pharmpy.model import Assignment, Model
from pharmpy.modeling import (
    cholesky_decompose,
    remove_unused_parameters_and_rvs,
    set_evaluation_step,
)


def _prepare_evaluation_model(model: Model) -> Model:
    df = model.dataset
    df = df[df['FREMTYPE'] == 0]
    model = model.replace(dataset=df)
    model = set_evaluation_step(model)
    return model


def _remove_frem_code(model):
    statements = model.statements
    kept_statements = [
        s
        for s in statements
        if not isinstance(s, Assignment)
        or not s.symbol.name.startswith("SDC")
        or Expr.symbol("FREMTYPE") not in s.expression.free_symbols
    ]
    return kept_statements


def _remove_frem_epsilon(model):
    rvs = model.random_variables
    rvs = rvs[:-1]
    return rvs


def _prepare_frem_model(model: Model) -> Model:
    kept_statements = _remove_frem_code(model)
    kept_rvs = _remove_frem_epsilon(model)

    model = model.replace(statements=kept_statements, random_variables=kept_rvs)
    model = remove_unused_parameters_and_rvs(model)
    frem_etas = model.random_variables[-1].names
    model = cholesky_decompose(model, frem_etas)

    return model
