import pharmpy.model
from pharmpy.basic import Expr
from pharmpy.model import Assignment, EstimationStep, ExecutionSteps, Statements
from pharmpy.workflows import Task, Workflow, WorkflowBuilder


def create_workflow(model=None):
    wb = WorkflowBuilder(name="linearize")

    if model is not None:
        start_task = Task('start_linearize', start_linearize, model)
    else:
        start_task = Task('start_linearize', start_linearize)

    wb.add_task(start_task)
    return Workflow(wb)


def start_linearize(model):
    return model


def create_linearized_model(model):
    linbase = pharmpy.model.Model(
        parameters=model.parameters,
        random_variables=model.random_variables,
        datainfo=model.datainfo,
    )

    ms = []
    base_terms_sum = 0
    for i, eta in enumerate(model.random_variables.etas.names, start=1):
        deta = Expr.symbol("D_ETA1")
        oeta = Expr.symbol("OETA")
        base = Assignment(Expr.symbol(f'BASE{i}'), deta * (Expr.symbol(eta) - oeta))
        ms.append(base)
        base_terms_sum += base.symbol

    base_terms = Assignment(Expr.symbol('BASE_TERMS'), base_terms_sum)
    ms.append(base_terms)
    ipred = Assignment(Expr.symbol('IPRED'), Expr.symbol('OPRED') + base_terms.symbol)
    ms.append(ipred)

    i = 1
    err_terms_sum = 0
    for epsno, eps in enumerate(model.random_variables.epsilons, start=1):
        err = Assignment(Expr.symbol(f'ERR{epsno}'), Expr.symbol(f'D_EPS{epsno}'))
        err_terms_sum += err.symbol
        ms.append(err)
        i += 1
        for etano, eta in enumerate(model.random_variables.etas.names, start=1):
            inter = Assignment(
                Expr.symbol(f'ERR{i}'),
                Expr.symbol(f'D_EPSETA{epsno}_{etano}')
                * (Expr.symbol(eta) - Expr.symbol(f'OETA{etano}')),
            )
            err_terms_sum += inter.symbol
            ms.append(inter)
            i += 1
    error_terms = Assignment(Expr.symbol('ERROR_TERMS'), err_terms_sum)
    ms.append(error_terms)

    # FIXME: Handle other DVs?
    y = list(model.dependent_variables.keys())[0]
    Assignment.create(y, ipred.symbol + error_terms.symbol)

    est = EstimationStep.create('foce', interaction=True)
    linbase = linbase.replace(
        name='linbase', statements=Statements(ms), execution_steps=ExecutionSteps.create([est])
    )
    return linbase
