from typing import Literal, Optional

from pharmpy.mfl import ModelFeatures
from pharmpy.mfl.features import Absorption, Covariate, Elimination, LagTime, Peripherals, Transits
from pharmpy.model import Model
from pharmpy.modeling import (
    get_bioavailability,
    get_covariate_effects,
    get_individual_parameters,
    get_number_of_peripheral_compartments,
    get_number_of_transit_compartments,
    get_parameter_rv,
    get_pd_parameters,
    get_pk_parameters,
    has_first_order_absorption,
    has_first_order_elimination,
    has_michaelis_menten_elimination,
    has_mixed_mm_fo_elimination,
    has_seq_zo_fo_absorption,
    has_weibull_absorption,
    has_zero_order_absorption,
    has_zero_order_elimination,
)
from pharmpy.modeling.odes import has_lag_time


def expand_model_features(model: Model, model_features: ModelFeatures) -> ModelFeatures:
    if model_features.is_expanded():
        return model_features

    refs = model_features.refs
    expand_to = {ref: _interpret_ref(model, ref) for ref in refs}

    model_features_expanded = model_features.expand(expand_to)

    return model_features_expanded


def _interpret_ref(model, symbol):
    if symbol.name in ['ABSORPTION', 'ELIMINATION', 'DISTRIBUTION']:
        return tuple(get_pk_parameters(model, kind=symbol.name.lower()))
    elif symbol.name == 'CATEGORICAL':
        return tuple(
            column.name
            for column in model.datainfo
            if column.type == 'covariate' and not column.continuous
        )
    elif symbol.name == 'CONTINUOUS':
        return tuple(
            column.name
            for column in model.datainfo
            if column.type == 'covariate' and column.continuous
        )
    elif symbol.name == 'IIV':
        return tuple(get_individual_parameters(model, level='iiv'))
    elif symbol.name == 'PD':
        try:
            pd_parameters = get_pd_parameters(model)
        except ValueError:
            return tuple()
        return tuple(pd_parameters)
    elif symbol.name == 'PD_IIV':
        try:
            pd_parameters = get_pd_parameters(model)
        except ValueError:
            return tuple()
        return tuple(
            pd_param for pd_param in pd_parameters if len(get_parameter_rv(model, pd_param)) > 0
        )
    elif symbol.name == 'PK':
        return tuple(get_pk_parameters(model))
    elif symbol.name == 'BIOAVAIL':
        return tuple(_get_bioaval_parameters(model))
    elif symbol.name == 'PK_IIV':
        return [
            pk_param
            for pk_param in get_pk_parameters(model)
            if len(get_parameter_rv(model, pk_param)) > 0
        ]
    else:
        raise ValueError(f'Could not find symbol: {symbol}')


def _get_bioaval_parameters(model):
    """Find all bioavail individual parameters
    Handle the case where one statement is only a logistic transformation
    """
    all_bio = get_bioavailability(model)
    pk = model.statements.before_odes
    found = []
    for _, bio in all_bio.items():
        if bio.is_symbol():
            ass = pk.find_assignment(bio)
            rhs_symbs = ass.rhs_symbols
            if len(rhs_symbs) == 1:
                symb = rhs_symbs.pop()
                ass2 = pk.find_assignment(symb)
                found.append(ass2.symbol.name)
            else:
                found += [s.name for s in rhs_symbs]
    return found


def get_model_features(
    model: Model, type: Optional[Literal['pk', 'covariates']] = None
) -> ModelFeatures:
    if type is not None and type not in ['pk', 'covariates']:
        raise ValueError(f'Invalid `type`: {type}')
    features = []

    if type is None or type == 'pk':
        absorption = _get_absorption(model)
        if absorption:
            features.append(absorption)
            transits = _get_transits(model)
            if transits:
                features.append(transits)

            if has_lag_time(model):
                features.append(LagTime.create(on=True))
            else:
                features.append(LagTime.create(on=False))

        elimination = _get_elimination(model)
        if elimination:
            features.append(elimination)

        peripherals = get_number_of_peripheral_compartments(model)
        features.append(Peripherals.create(peripherals))

    if type is None or type == 'covariates':
        covariates = get_covariate_effects(model)
        if len(covariates) > 0:
            for (param, cov), fp_op in covariates.items():
                fp, op = fp_op[0][0], fp_op[0][1]
                features.append(Covariate.create(parameter=param, covariate=cov.name, fp=fp, op=op))
    return ModelFeatures.create(features)


def _get_absorption(model):
    absorption_type = None
    if has_weibull_absorption(model):
        absorption_type = "WEIBULL"
    elif has_seq_zo_fo_absorption(model):
        absorption_type = "SEQ-ZO-FO"
    elif has_zero_order_absorption(model):
        absorption_type = "ZO"
    elif has_first_order_absorption(model):
        absorption_type = "FO"
    if absorption_type:
        return Absorption.create(absorption_type)
    else:
        return None


def _get_elimination(model):
    elimination_type = None
    if has_mixed_mm_fo_elimination(model):
        elimination_type = "MIX-FO-MM"
    elif has_zero_order_elimination(model):
        elimination_type = "ZO"
    elif has_first_order_elimination(model):
        elimination_type = "FO"
    elif has_michaelis_menten_elimination(model):
        elimination_type = "MM"
    if elimination_type:
        return Elimination.create(elimination_type)
    else:
        return None


def _get_transits(model):
    transits = get_number_of_transit_compartments(model)
    if not transits or model.statements.ode_system.find_depot(model.statements):
        with_depot = True
    else:
        with_depot = False
    return Transits.create(transits, with_depot)
