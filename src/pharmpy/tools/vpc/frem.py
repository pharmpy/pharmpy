from pharmpy.basic import Expr
from pharmpy.internals.math import triangular
from pharmpy.model import Assignment, Model, Statements, get_and_check_dataset
from pharmpy.modeling import (
    cholesky_decompose,
    get_thetas,
    insert_ebes_into_dataset,
    remove_parameter_uncertainty_step,
    remove_unused_parameters_and_rvs,
    set_evaluation_step,
    set_initial_estimates,
)
from pharmpy.workflows import ModelEntry


def prepare_evaluation_model(me: ModelEntry) -> ModelEntry:
    model = me.model
    df = get_and_check_dataset(model)
    df = df[df['FREMTYPE'] == 0]
    df = df.reset_index(drop=True)
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
    kept_rvs = _remove_frem_epsilon(model)

    model = model.replace(statements=kept_statements, random_variables=kept_rvs)
    model = remove_unused_parameters_and_rvs(model)
    frem_etas = list(model.random_variables.etas[-1].names)
    model = cholesky_decompose(model, frem_etas)

    all_indests = me.modelfit_results.individual_estimates
    indests = all_indests[frem_etas]

    def trimcov(x):
        return x.loc[frem_etas, frem_etas]

    etcs = me.modelfit_results.individual_estimates_covariance.apply(trimcov)
    model = insert_ebes_into_dataset(model, indests, etcs)

    n = len(frem_etas)
    nthetas = triangular(n)
    thetas = get_thetas(model)
    sdcorr_thetas = thetas[len(thetas) - nthetas :]
    new_params = model.parameters[: len(thetas) - nthetas] + model.parameters[len(thetas) :]

    assignments = []
    i = 0
    for row in range(n):
        for col in range(row + 1):
            name = sdcorr_thetas[i].name
            if row == col:
                assignment = Assignment(
                    Expr.symbol(name), Expr.symbol(f"ETC_{row+1}_{col+1}").sqrt()
                )
            else:
                assignment = Assignment(
                    Expr.symbol(name),
                    Expr.symbol(f"ETC_{row+1}_{col+1}")
                    / (
                        Expr.symbol(f"ETC_{col+1}_{col+1}") * Expr.symbol(f"ETC_{row+1}_{row+1}")
                    ).sqrt(),
                )
            assignments.append(assignment)
            i += 1

    i = 1
    new_statements = []
    for s in model.statements:
        if (
            isinstance(s, Assignment)
            and s.symbol.name.endswith("_C")
            and s.symbol.name[:-2] in frem_etas
        ):
            expr = s.expression + Expr.symbol(f"ET_{i}")
            assignment = Assignment(s.symbol, expr)
            new_statements.append(assignment)
            i += 1
        else:
            new_statements.append(s)

    model = model.replace(
        parameters=new_params, statements=Statements.create(assignments + new_statements)
    )
    model = model.update_source()
    return ModelEntry.create(model=model)
