import pharmpy.model
from pharmpy.deps import sympy
from pharmpy.model import Assignment, EstimationStep, EstimationSteps, Statements
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
    linbase = pharmpy.model.Model(
        parameters=model.parameters,
        random_variables=model.random_variables,
        datainfo=model.datainfo,
    )

    ms = []
    base_terms_sum = 0
    for i, eta in enumerate(model.random_variables.etas.names, start=1):
        deta = sympy.Symbol("D_ETA1")
        oeta = sympy.Symbol("OETA")
        base = Assignment(sympy.Symbol(f'BASE{i}'), deta * (sympy.Symbol(eta) - oeta))
        ms.append(base)
        base_terms_sum += base.symbol

    base_terms = Assignment(sympy.Symbol('BASE_TERMS'), base_terms_sum)
    ms.append(base_terms)
    ipred = Assignment(sympy.Symbol('IPRED'), sympy.Symbol('OPRED') + base_terms.symbol)
    ms.append(ipred)

    i = 1
    err_terms_sum = 0
    for epsno, eps in enumerate(model.random_variables.epsilons, start=1):
        err = Assignment(sympy.Symbol(f'ERR{epsno}'), sympy.Symbol(f'D_EPS{epsno}'))
        err_terms_sum += err.symbol
        ms.append(err)
        i += 1
        for etano, eta in enumerate(model.random_variables.etas.names, start=1):
            inter = Assignment(
                sympy.Symbol(f'ERR{i}'),
                sympy.Symbol(f'D_EPSETA{epsno}_{etano}')
                * (sympy.Symbol(eta) - sympy.Symbol(f'OETA{etano}')),
            )
            err_terms_sum += inter.symbol
            ms.append(inter)
            i += 1
    error_terms = Assignment(sympy.Symbol('ERROR_TERMS'), err_terms_sum)
    ms.append(error_terms)

    # FIXME: handle other DVs?
    y = list(model.dependent_variables.keys())[0]
    Assignment(y, ipred.symbol + error_terms.symbol)

    est = EstimationStep.create('foce', interaction=True)
    linbase = linbase.replace(
        name='linbase', statements=Statements(ms), estimation_steps=EstimationSteps.create([est])
    )
    return linbase
