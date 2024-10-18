"""
:meta private:
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from pharmpy.basic import Expr
from pharmpy.deps import sympy
from pharmpy.internals.fs.path import normalize_user_given_path
from pharmpy.model import (
    Assignment,
    ColumnInfo,
    Compartment,
    CompartmentalSystem,
    CompartmentalSystemBuilder,
    DataInfo,
    EstimationStep,
    ExecutionSteps,
    Model,
    NormalDistribution,
    Parameter,
    Parameters,
    RandomVariables,
    Statements,
    get_and_check_odes,
    output,
)

from .data import add_admid, create_default_datainfo, read_dataset_from_datainfo
from .error import set_proportional_error_model
from .odes import add_bioavailability, set_first_order_absorption
from .parameter_variability import add_iiv, create_joint_distribution
from .parameters import set_initial_estimates


def create_basic_pk_model(
    administration: str = 'iv',
    dataset_path: Optional[Union[str, Path]] = None,
    cl_init: float = 0.01,
    vc_init: float = 1.0,
    mat_init: float = 0.1,
) -> Model:
    """
    Creates a basic pk model of given type. The model will be a one compartment model, with first
    order elimination and in the case of oral administration first order absorption with no absorption
    delay. The elimination rate will be :math:`CL/V` and the absorption rate will be :math:`1/MAT`

    Parameters
    ----------
    administration : str
        Type of PK model to create. Supported are 'iv', 'oral' and 'ivoral'
    dataset_path : str or Path
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
        dataset_path = normalize_user_given_path(dataset_path)
        di = create_default_datainfo(dataset_path)
        df = read_dataset_from_datainfo(di, datatype='nonmem')
    else:
        di_col_dict = {'ID': 'id', 'TIME': 'idv', 'AMT': 'dose', 'DV': 'dv'}
        di_ci = [ColumnInfo.create(key, type=value) for key, value in di_col_dict.items()]
        di = DataInfo.create(di_ci)
        df = None

    if administration not in [
        'iv',
        'oral',
        'ivoral',
    ]:
        raise ValueError(f'Invalid input: `{administration}` as administration is not supported.')

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

    CL = Expr.symbol('CL')
    VC = Expr.symbol('VC')
    cl_ass = Assignment(CL, pop_cl.symbol * Expr.symbol(eta_cl_name).exp())
    vc_ass = Assignment(VC, pop_vc.symbol * Expr.symbol(eta_vc_name).exp())

    cb = CompartmentalSystemBuilder()
    # FIXME: This shouldn't be used here
    from pharmpy.model.external.nonmem.advan import dosing, find_dose

    doses = dosing(di, df, 1)
    central = Compartment.create('CENTRAL', doses=find_dose(doses, 1))
    cb.add_compartment(central)
    cb.add_flow(central, output, CL / VC)

    ipred = Assignment(Expr.symbol('IPRED'), central.amount / VC)
    y_ass = Assignment(Expr.symbol('Y'), ipred.symbol)

    stats = Statements([cl_ass, vc_ass, CompartmentalSystem(cb), ipred, y_ass])

    est = EstimationStep.create(
        "FOCE",
        interaction=True,
        maximum_evaluations=99999,
        predictions=('PRED', 'CIPREDI'),
        residuals=('CWRES',),
    )
    eststeps = ExecutionSteps.create([est])

    model = Model.create(
        name='start',
        statements=stats,
        execution_steps=eststeps,
        dependent_variables={y_ass.symbol: 1},
        random_variables=rvs,
        parameters=params,
        description='Start model',
        dataset=df,
        datainfo=di,
    )

    model = set_proportional_error_model(model)
    model = create_joint_distribution(model, [eta_cl_name, eta_vc_name])

    if administration == 'oral' or administration == 'ivoral':
        model = set_first_order_absorption(model)
        model = set_initial_estimates(model, {'POP_MAT': mat_init})
        model = add_iiv(model, list_of_parameters='MAT', expression='exp', initial_estimate=0.1)
    if administration == 'ivoral':
        # FIXME: Dependent on CMT column having 1 and 2 as values otherwise
        # compartment structure don't match
        model = add_bioavailability(model, logit_transform=True)
        model = set_initial_estimates(model, {'POP_BIO': 0.5})

        # Add IIV to BIO
        # TODO: ? Should there be another initial estimate?
        model = add_iiv(model, list_of_parameters="BIO", expression='add', initial_estimate=0.1)

        # Set dosing to the CENTRAL compartment as well
        ode = model.statements.ode_system
        cb = CompartmentalSystemBuilder(ode)
        if df is None:
            doses = dosing(di, df, 2)
            central_dose = find_dose(doses, comp_number=2, admid=2)
        else:
            # doses = dosing(di, df, 1)
            central_dose = find_dose(doses, comp_number=2, admid=2)
            if not central_dose:
                raise ValueError(
                    (
                        "Could not determine IV dose from dataset. "
                        "Currently require CMT column with values 1 and 2 "
                    )
                )
        central = cb.find_compartment("CENTRAL")
        assert central is not None
        cb.set_dose(central, dose=central_dose)

        ode = CompartmentalSystem(cb)
        model = model.replace(
            statements=model.statements.before_odes + ode + model.statements.after_odes
        )
        if df is not None:
            model = add_admid(model)

            # Change bioavailability to piecewise for oral/iv doses
            model = model.replace(
                statements=model.statements.reassign(
                    Expr.symbol("F_BIO"),
                    Expr.piecewise(
                        (
                            1 / (1 + ((-Expr.symbol("BIO")).exp())),
                            sympy.Eq(sympy.Symbol('ADMID'), 1),
                        ),
                        (1, sympy.true),
                    ),
                )
            )

        # Add covariate to error model with the following logic
        # RUV = 1 * covariate
        # Y = F + F*EPS*RUV
        ruv_ass = Assignment(Expr.symbol("RUV"), Expr.integer(1))
        model = model.replace(
            statements=model.statements.before_odes
            + ruv_ass
            + get_and_check_odes(model)
            + model.statements.after_odes
        )
        Y_ass = model.statements.find_assignment("Y")
        assert Y_ass is not None
        ipred = Y_ass.expression.make_args(Y_ass.expression)[0]
        error = Y_ass.expression.make_args(Y_ass.expression)[1]
        new_Y_ass = Assignment.create(Y_ass.symbol, ipred + error * ruv_ass.symbol)

        model = model.replace(
            statements=model.statements.reassign(Y_ass.symbol, new_Y_ass.expression)
        )

    return model
