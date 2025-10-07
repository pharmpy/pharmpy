from pharmpy.mfl import ModelFeatures
from pharmpy.model import Model
from pharmpy.modeling import (
    get_bioavailability,
    get_individual_parameters,
    get_parameter_rv,
    get_pd_parameters,
    get_pk_parameters,
)


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
