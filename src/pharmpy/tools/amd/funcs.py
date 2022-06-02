from pathlib import Path

import sympy

import pharmpy
from pharmpy import (
    Assignment,
    DataInfo,
    EstimationStep,
    EstimationSteps,
    ModelStatements,
    Parameter,
    Parameters,
    RandomVariable,
    RandomVariables,
)
from pharmpy.modeling import (
    create_joint_distribution,
    set_first_order_absorption,
    set_initial_estimates,
    set_proportional_error_model,
)
from pharmpy.modeling.data import read_dataset_from_datainfo
from pharmpy.workflows import default_model_database


def create_start_model(dataset_path, modeltype='pk_oral', cl_init=0.01, vc_init=1.0, mat_init=0.1):
    dataset_path = Path(dataset_path)

    pop_cl = Parameter('POP_CL', cl_init, lower=0)
    pop_vc = Parameter('POP_VC', vc_init, lower=0)
    iiv_cl = Parameter('IIV_CL', 0.01)
    iiv_vc = Parameter('IIV_VC', 0.01)

    params = Parameters([pop_cl, pop_vc, iiv_cl, iiv_vc])

    eta_cl = RandomVariable.normal('eta_cl', 'iiv', 0, iiv_cl.symbol)
    eta_vc = RandomVariable.normal('eta_vc', 'iiv', 0, iiv_vc.symbol)
    rvs = RandomVariables([eta_cl, eta_vc])

    CL = sympy.Symbol('CL')
    VC = sympy.Symbol('VC')
    cl_ass = Assignment(CL, pop_cl.symbol * sympy.exp(eta_cl.symbol))
    vc_ass = Assignment(VC, pop_vc.symbol * sympy.exp(eta_vc.symbol))

    dose = sympy.Symbol('AMT')
    odes = pharmpy.CompartmentalSystem()
    central = odes.add_compartment('CENTRAL')
    output = odes.add_compartment('OUTPUT')
    odes.add_flow(central, output, CL / VC)
    central.dose = pharmpy.Bolus(dose)

    ipred = Assignment('IPRED', central.amount / VC)
    y_ass = Assignment('Y', ipred.symbol)

    stats = ModelStatements([cl_ass, vc_ass, odes, ipred, y_ass])

    datainfo_path = dataset_path.with_suffix('.datainfo')

    if datainfo_path.is_file():
        di = DataInfo.read_json(dataset_path.with_suffix('.datainfo'))
        di.path = dataset_path
    else:
        # FIXME: Create a default di here?
        di = None
    df = read_dataset_from_datainfo(di)

    est = EstimationStep(
        "FOCE",
        interaction=True,
        maximum_evaluations=99999,
        predictions=['CIPREDI'],
        residuals=['CWRES'],
    )
    eststeps = EstimationSteps([est])

    model = pharmpy.Model()
    model.name = 'start'
    model.parameters = params
    model.random_variables = rvs
    model.statements = stats
    model.dependent_variable = y_ass.symbol
    model.database = default_model_database()
    model.dataset = df
    if di:
        model.datainfo = di
    model.estimation_steps = eststeps
    model.filename_extension = '.mod'  # Should this really be needed?

    set_proportional_error_model(model, zero_protection=True)
    create_joint_distribution(model, [eta_cl.name, eta_vc.name])
    if modeltype == 'pk_oral':
        set_first_order_absorption(model)
        set_initial_estimates(model, {'POP_MAT': mat_init})

    return model
