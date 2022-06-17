from pathlib import Path

import pandas as pd
import sympy

import pharmpy
from pharmpy import (
    Assignment,
    ColumnInfo,
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
    di = _create_default_datainfo(dataset_path)
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
    model.datainfo = di
    model.estimation_steps = eststeps
    model.filename_extension = '.mod'  # Should this really be needed?

    set_proportional_error_model(model, zero_protection=True)
    create_joint_distribution(model, [eta_cl.name, eta_vc.name])
    if modeltype == 'pk_oral':
        set_first_order_absorption(model)
        set_initial_estimates(model, {'POP_MAT': mat_init})
    return model


def _create_default_datainfo(path):
    datainfo_path = path.with_suffix('.datainfo')
    if datainfo_path.is_file():
        di = DataInfo.read_json(path.with_suffix('.datainfo'))
    else:
        colnames = list(pd.read_csv(path, nrows=0))
        column_info = []
        for colname in colnames:
            info = ColumnInfo(colname)
            if colname == 'ID' or colname == 'L1':
                info.type = 'id'
                info.scale = 'nominal'
                info.datatype = 'int32'
            elif colname == 'DV':
                info.type = 'dv'
            elif colname == 'TIME':
                info.type = 'idv'
                info.scale = 'ratio'
                if not set(colnames).isdisjoint({'DATE', 'DAT1', 'DAT2', 'DAT3'}):
                    info.datatype = 'nmtran-time'
            elif colname == 'EVID':
                info.type = 'event'
                info.scale = 'nominal'
            elif colname == 'MDV':
                if 'EVID' in colnames:
                    info.type = 'mdv'
                else:
                    info.type = 'event'
                    info.scale = 'nominal'
                    info.datatype = 'int32'
            elif colname == 'AMT':
                info.type = 'dose'
                info.scale = 'ratio'
            column_info.append(info)
        di = DataInfo(column_info)
    di.path = path
    return di
