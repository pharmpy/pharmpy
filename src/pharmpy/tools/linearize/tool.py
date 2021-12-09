import sympy

import pharmpy.model
from pharmpy import Assignment, ModelStatements
from pharmpy.estimation import EstimationStep, EstimationSteps
from pharmpy.workflows import Task, Workflow


def create_workflow(model=None):
    wf = Workflow()
    wf.name = "linearize"

    if model is not None:
        start_task = Task('start_linearize', start_linearize, model)
    else:
        start_task = Task('start_linearize', start_linearize)

    wf.add_task(start_task)
    return wf


def start_linearize(model):
    return model


def create_linearized_model(model):
    linbase = pharmpy.model.Model()
    linbase.parameters = model.parameters.copy()
    linbase.random_variables = model.random_variables.copy()

    ms = ModelStatements()
    base_terms_sum = 0
    for i, eta in enumerate(model.random_variables.etas, start=1):
        deta = sympy.Symbol("D_ETA1")
        oeta = sympy.Symbol("OETA")
        base = Assignment(f'BASE{i}', deta * (eta.symbol - oeta))
        ms.append(base)
        base_terms_sum += base.symbol

    base_terms = Assignment('BASE_TERMS', base_terms_sum)
    ms.append(base_terms)
    ipred = Assignment('IPRED', sympy.Symbol('OPRED') + base_terms.symbol)
    ms.append(ipred)

    i = 1
    err_terms_sum = 0
    for epsno, eps in enumerate(model.random_variables.epsilons, start=1):
        err = Assignment(f'ERR{epsno}', f'D_EPS{epsno}')
        err_terms_sum += err.symbol
        ms.append(err)
        i += 1
        for etano, eta in enumerate(model.random_variables.etas, start=1):
            inter = Assignment(
                f'ERR{i}',
                sympy.Symbol(f'D_EPSETA{epsno}_{etano}')
                * (eta.symbol - sympy.Symbol(f'OETA{etano}')),
            )
            err_terms_sum += inter.symbol
            ms.append(inter)
            i += 1
    error_terms = Assignment('ERROR_TERMS', err_terms_sum)
    ms.append(error_terms)

    y = model.dependent_variable
    Assignment(y, ipred.symbol + error_terms.symbol)
    linbase.statements = ms

    linbase.name = 'linbase'

    est = EstimationStep('foce', interaction=True)
    linbase.estimation_steps = EstimationSteps([est])
    return linbase
