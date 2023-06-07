"""
:meta private:
"""
from pathlib import Path
from typing import Optional

from pharmpy.deps import pandas as pd
from pharmpy.deps import sympy
from pharmpy.internals.fs.path import path_absolute
from pharmpy.model import (
    Assignment,
    Bolus,
    ColumnInfo,
    Compartment,
    CompartmentalSystem,
    CompartmentalSystemBuilder,
    DataInfo,
    EstimationStep,
    EstimationSteps,
    Infusion,
    Model,
    NormalDistribution,
    Parameter,
    Parameters,
    RandomVariables,
    Statements,
    output,
)

from .data import read_dataset_from_datainfo
from .error import set_proportional_error_model
from .odes import set_first_order_absorption
from .parameter_variability import add_iiv, create_joint_distribution
from .parameters import set_initial_estimates


def create_basic_pk_model(
    modeltype: str,
    dataset_path: Optional[str] = None,
    cl_init: float = 0.01,
    vc_init: float = 1.0,
    mat_init: float = 0.1,
) -> Model:
    """
    Creates a basic pk model of given type

    Parameters
    ----------
    modeltype : str
        Type of PK model to create. Supported are 'oral' and 'iv'
    dataset_path : str
        Optional path to a dataset
    cl_init : float
        Initial estimate of the clearance parameter
    vc_init : float
        Initial estimate of the central volume parameter
    mat_init : float
        Initial estimate of the mean absorption time parameter (if applicable)

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = create_basic_pk_model('oral')

    """
    if dataset_path is not None:
        dataset_path = Path(dataset_path)
        di = _create_default_datainfo(dataset_path)
        df = read_dataset_from_datainfo(di, datatype='nonmem')
    else:
        di = DataInfo()
        df = None

    pop_cl = Parameter('POP_CL', cl_init, lower=0)
    pop_vc = Parameter('POP_VC', vc_init, lower=0)
    iiv_cl = Parameter('IIV_CL', 0.1)
    iiv_vc = Parameter('IIV_VC', 0.1)

    params = Parameters((pop_cl, pop_vc, iiv_cl, iiv_vc))

    eta_cl_name = 'ETA_CL'
    eta_cl = NormalDistribution.create(eta_cl_name, 'iiv', 0, iiv_cl.symbol)
    eta_vc_name = 'ETA_VC'
    eta_vc = NormalDistribution.create(eta_vc_name, 'iiv', 0, iiv_vc.symbol)
    rvs = RandomVariables.create([eta_cl, eta_vc])

    CL = sympy.Symbol('CL')
    VC = sympy.Symbol('VC')
    cl_ass = Assignment(CL, pop_cl.symbol * sympy.exp(sympy.Symbol(eta_cl_name)))
    vc_ass = Assignment(VC, pop_vc.symbol * sympy.exp(sympy.Symbol(eta_vc_name)))

    cb = CompartmentalSystemBuilder()
    central = Compartment.create('CENTRAL', dose=dosing(di, df, 1))
    cb.add_compartment(central)
    cb.add_flow(central, output, CL / VC)

    ipred = Assignment(sympy.Symbol('IPRED'), central.amount / VC)
    y_ass = Assignment(sympy.Symbol('Y'), ipred.symbol)

    stats = Statements([cl_ass, vc_ass, CompartmentalSystem(cb), ipred, y_ass])

    est = EstimationStep.create(
        "FOCE",
        interaction=True,
        maximum_evaluations=99999,
        predictions=['CIPREDI'],
        residuals=['CWRES'],
    )
    eststeps = EstimationSteps.create([est])

    model = Model(
        name='start',
        statements=stats,
        estimation_steps=eststeps,
        dependent_variables={y_ass.symbol: 1},
        random_variables=rvs,
        parameters=params,
        description='Start model',
        filename_extension='.mod',  # Should this really be needed?
        dataset=df,
        datainfo=di,
    )

    model = set_proportional_error_model(model)
    model = create_joint_distribution(
        model,
        [eta_cl_name, eta_vc_name],
        individual_estimates=model.modelfit_results.individual_estimates
        if model.modelfit_results is not None
        else None,
    )
    if modeltype == 'oral':
        model = set_first_order_absorption(model)
        model = set_initial_estimates(model, {'POP_MAT': mat_init})
        model = add_iiv(model, list_of_parameters='MAT', expression='exp', initial_estimate=0.1)

    return model


def _create_default_datainfo(path):
    path = path_absolute(path)
    datainfo_path = path.with_suffix('.datainfo')
    if datainfo_path.is_file():
        di = DataInfo.read_json(datainfo_path)
        di = di.replace(path=path)
    else:
        colnames = list(pd.read_csv(path, nrows=0))
        column_info = []
        for colname in colnames:
            if colname == 'ID' or colname == 'L1':
                info = ColumnInfo.create(colname, type='id', scale='nominal', datatype='int32')
            elif colname == 'DV':
                info = ColumnInfo.create(colname, type='dv')
            elif colname == 'TIME':
                if not set(colnames).isdisjoint({'DATE', 'DAT1', 'DAT2', 'DAT3'}):
                    datatype = 'nmtran-time'
                else:
                    datatype = 'float64'
                info = ColumnInfo.create(colname, type='idv', scale='ratio', datatype=datatype)
            elif colname == 'EVID':
                info = ColumnInfo.create(colname, type='event', scale='nominal')
            elif colname == 'MDV':
                if 'EVID' in colnames:
                    info = ColumnInfo.create(colname, type='mdv')
                else:
                    info = ColumnInfo.create(
                        colname, type='event', scale='nominal', datatype='int32'
                    )
            elif colname == 'AMT':
                info = ColumnInfo.create(colname, type='dose', scale='ratio')
            else:
                info = ColumnInfo.create(colname)
            column_info.append(info)
        di = DataInfo.create(column_info, path=path, separator=',')
    return di


def dosing(di: DataInfo, dataset, dose_comp: int):
    # FIXME: Copied from plugins.nonmem.advan
    if di is None:
        return Bolus(sympy.Symbol('AMT'))

    if 'RATE' not in di.names or di['RATE'].drop:
        return Bolus(sympy.Symbol('AMT'))

    df = dataset
    if (df['RATE'] == 0).all():
        return Bolus(sympy.Symbol('AMT'))
    elif (df['RATE'] == -1).any():
        return Infusion(sympy.Symbol('AMT'), rate=sympy.Symbol(f'R{dose_comp}'))
    elif (df['RATE'] == -2).any():
        return Infusion(sympy.Symbol('AMT'), duration=sympy.Symbol(f'D{dose_comp}'))
    else:
        return Infusion(sympy.Symbol('AMT'), rate=sympy.Symbol('RATE'))
