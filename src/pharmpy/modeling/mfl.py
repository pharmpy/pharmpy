import builtins
import itertools
from collections import defaultdict
from functools import partial
from typing import Callable, Literal, Optional, Sequence, Union

from pharmpy.deps import pandas as pd
from pharmpy.mfl import (
    IIV,
    Absorption,
    Allometry,
    Covariance,
    Covariate,
    DirectEffect,
    EffectComp,
    Elimination,
    IndirectEffect,
    LagTime,
    Metabolite,
    ModelFeature,
    ModelFeatures,
    Peripherals,
    Transits,
)
from pharmpy.model import Model
from pharmpy.modeling import (
    add_allometry,
    add_covariate_effect,
    add_effect_compartment,
    add_iiv,
    add_indirect_effect,
    add_lag_time,
    add_metabolite,
    create_joint_distribution,
    get_bioavailability,
    get_covariate_effects,
    get_individual_parameters,
    get_number_of_peripheral_compartments,
    get_number_of_transit_compartments,
    get_parameter_rv,
    get_pd_parameters,
    get_pk_parameters,
    get_rv_parameters,
    has_first_order_absorption,
    has_first_order_elimination,
    has_michaelis_menten_elimination,
    has_mixed_mm_fo_elimination,
    has_seq_zo_fo_absorption,
    has_weibull_absorption,
    has_zero_order_absorption,
    has_zero_order_elimination,
    remove_covariate_effect,
    remove_iiv,
    remove_lag_time,
    set_direct_effect,
    set_first_order_absorption,
    set_first_order_elimination,
    set_michaelis_menten_elimination,
    set_mixed_mm_fo_elimination,
    set_n_transit_compartments,
    set_peripheral_compartments,
    set_seq_zo_fo_absorption,
    set_transit_compartments,
    set_weibull_absorption,
    set_zero_order_absorption,
    set_zero_order_elimination,
    split_joint_distribution,
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
    model: Model, type: Optional[Literal['pk', 'covariates', 'iiv', 'covariance']] = None
) -> ModelFeatures:
    if type is not None and type not in ['pk', 'covariates', 'iiv', 'covariance']:
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

    if type is None or type == 'iiv':
        features.extend(_get_iiv(model))

    if type is None or type == 'covariance':
        iivs = model.random_variables.iiv
        for iiv1, iiv2 in itertools.combinations(iivs.names, 2):
            if iivs.get_covariance(iiv1, iiv2) != 0:
                param1 = get_rv_parameters(model, iiv1)[0]
                param2 = get_rv_parameters(model, iiv2)[0]
                features.append(Covariance.create('IIV', (param1, param2)))

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


def _get_iiv(model):
    features = []
    individual_params = get_individual_parameters(model, 'iiv')
    rvs = model.random_variables.iiv
    for param in individual_params:
        param_statement = model.statements.before_odes.full_expression(param)
        args = param_statement.args
        fp = None
        for arg in args:
            if not arg.free_symbols.intersection(rvs.symbols):
                continue
            if fp:
                raise NotImplementedError(
                    f'Could not determine eta distribution: {param_statement}'
                )
            if param_statement.is_add():
                if arg in rvs.symbols:
                    fp = 'add'
            elif param_statement.is_mul():
                if arg.is_exp():
                    fp = 'exp'
        if not fp:
            raise NotImplementedError(f'Could not determine eta distribution: {param_statement}')
        iiv = IIV.create(param, fp=fp, optional=False)
        features.append(iiv)
    return features


def generate_transformations(
    model_features: Union[ModelFeatures, Sequence[ModelFeature]],
    include_add: bool = True,
    include_remove: bool = True,
    individual_estimates: Optional[pd.DataFrame] = None,
) -> list[Callable]:
    if isinstance(model_features, Sequence):
        model_features = ModelFeatures.create(model_features)
    if not model_features.is_expanded():
        raise ValueError('Incorrect option `model_features`: must be expanded')

    transformations = []
    covariances = model_features.covariance
    for feature in model_features:
        if feature in covariances:
            continue
        funcs = _get_funcs(feature, include_add, include_remove)
        transformations.extend(funcs)

    if covariances:
        transformations.extend(
            _get_covariance_func(covariances, include_add, include_remove, individual_estimates)
        )

    return transformations


def _get_funcs(feature: ModelFeature, include_add: bool, include_remove: bool) -> list[Callable]:
    if isinstance(feature, (Absorption, Elimination)):
        return _get_absorption_elimination_func(feature)
    elif isinstance(feature, Transits):
        return _get_transit_func(feature)
    elif isinstance(feature, LagTime):
        return _get_lag_time_func(feature)
    elif isinstance(feature, Peripherals):
        return _get_peripherals_func(feature)
    elif isinstance(feature, (DirectEffect, IndirectEffect, EffectComp)):
        return _get_pd_func(feature)
    elif isinstance(feature, Metabolite):
        return _get_metabolite_func(feature)
    elif isinstance(feature, Covariate):
        return _get_covariate_func(feature, include_add, include_remove)
    elif isinstance(feature, Allometry):
        return _get_allometry_func(feature)
    elif isinstance(feature, IIV):
        return _get_iiv_func(feature, include_add, include_remove)
    else:
        raise NotImplementedError


def _get_absorption_elimination_func(feature: Union[Absorption, Elimination]):
    assert type(feature) in FUNC_MAPPING.keys()
    func = FUNC_MAPPING[type(feature)].get(feature.type)
    if func is None:
        raise ValueError
    return [func]


def _get_transit_func(feature: Transits):
    if feature.number == 'N':
        func = partial(set_n_transit_compartments, keep_depot=feature.depot)
    else:
        n = feature.number
        if not feature.depot:
            if n == 0:
                return []
            n += 1
        func = partial(set_transit_compartments, n=n, keep_depot=feature.depot)
    return [func]


def _get_lag_time_func(feature: LagTime):
    if feature.on:
        func = add_lag_time
    else:
        func = remove_lag_time
    return [func]


def _get_peripherals_func(feature: Peripherals):
    if feature.metabolite:
        name = 'METABOLITE'
    else:
        name = None
    func = partial(set_peripheral_compartments, n=feature.number, name=name)
    return [func]


def _get_pd_func(feature: Union[DirectEffect, IndirectEffect, EffectComp]):
    kwargs = {'expr': feature.type.lower()}
    if isinstance(feature, IndirectEffect):
        kwargs['prod'] = feature.production
    assert type(feature) in FUNC_MAPPING.keys()
    func = partial(FUNC_MAPPING[type(feature)], **kwargs)
    return [func]


def _get_metabolite_func(feature: Metabolite):
    presystemic = True if feature.type == 'PSC' else False
    func = partial(add_metabolite, presystemic=presystemic)
    return [func]


def _get_covariate_func(feature: Covariate, include_add: bool, include_remove: bool):
    funcs = []
    if include_add:
        kwargs_add = {
            'parameter': feature.parameter,
            'covariate': feature.covariate,
            'effect': feature.fp.lower(),
            'operation': feature.op,
            'allow_nested': True,
        }
        func_add = partial(add_covariate_effect, **kwargs_add)
        funcs.append(func_add)
    if feature.optional and include_remove:
        kwargs_remove = {'parameter': feature.parameter, 'covariate': feature.covariate}
        func_remove = partial(remove_covariate_effect, **kwargs_remove)
        funcs.append(func_remove)
    return funcs


def _get_allometry_func(feature: Allometry):
    func = partial(
        add_allometry, allometric_variable=feature.covariate, reference_value=feature.reference
    )
    return [func]


def _get_iiv_func(feature: IIV, include_add: bool, include_remove: bool):
    funcs = []
    if include_add:
        func_add = partial(
            add_iiv, list_of_parameters=feature.parameter, expression=feature.fp.lower()
        )
        funcs.append(func_add)
    if include_remove:
        func_remove = partial(remove_iiv, to_remove=feature.parameter)
        funcs.append(func_remove)
    return funcs


def _get_covariance_func(
    features: Sequence[Covariance], include_add, include_remove, ies: Optional[pd.DataFrame]
):
    funcs = []
    if include_add:
        blocks = Covariance.get_covariance_blocks(features)
        for block in blocks:
            if include_add:
                func_join = partial(
                    create_joint_distribution,
                    rvs=block,
                    individual_estimates=ies,
                )
                funcs.append(func_join)
    if include_remove:
        groups = defaultdict(list)
        for feature in features:
            param_1, param_2 = feature.parameters
            groups[param_1].append(param_2)
            groups[param_2].append(param_1)

        for key, values in groups.items():
            if len(values) == len(groups) - 1:
                func_split = partial(
                    split_joint_distribution,
                    rvs=[key],
                )
                funcs.append(func_split)
    return funcs


FUNC_MAPPING = {
    Absorption: {
        'FO': set_first_order_absorption,
        'ZO': set_zero_order_absorption,
        'SEQ-ZO-FO': set_seq_zo_fo_absorption,
        'WEIBULL': set_weibull_absorption,
    },
    Elimination: {
        'FO': set_first_order_elimination,
        'ZO': set_zero_order_elimination,
        'MIX-FO-MM': set_mixed_mm_fo_elimination,
        'MM': set_michaelis_menten_elimination,
    },
    DirectEffect: set_direct_effect,
    IndirectEffect: add_indirect_effect,
    EffectComp: add_effect_compartment,
}


def transform_into_search_space(
    model: Model,
    search_space: Union[ModelFeatures, Sequence[ModelFeature]],
    type: Optional[Literal['pk', 'covariates', 'iiv', 'covariance']] = None,
    individual_estimates: Optional[pd.DataFrame] = None,
) -> Model:
    if isinstance(search_space, Sequence):
        search_space = ModelFeatures.create(search_space)
    search_space = _get_mfl_from_type(search_space, type)
    model_features = get_model_features(model, type=type)

    if model_features == search_space:
        return model

    features_to_add, features_to_remove = _get_feature_diffs(search_space, model_features, type)

    transformations = generate_transformations(
        features_to_remove, include_add=False, individual_estimates=individual_estimates
    )
    transformations += generate_transformations(
        features_to_add, include_remove=False, individual_estimates=individual_estimates
    )
    model_transformed = model
    for func in transformations:
        try:
            model_transformed = func(model_transformed)
        except Exception:
            raise ValueError(f'Could not transform model: {func}')
    return model_transformed


def _get_mfl_from_type(mfl, type):
    if type == 'pk':
        return mfl.filter(filter_on='pk')
    elif type == 'covariates':
        return mfl.covariates
    elif type == 'iiv':
        return mfl.iiv
    elif type == 'covariance':
        return mfl.covariance
    else:
        return mfl


def _get_feature_diffs(search_space, model_features, type):
    features_to_add = []
    features_to_remove = []

    # None of these are optional since all are in model_features
    features_not_in_search_space = model_features - search_space
    for feature in features_not_in_search_space:
        if isinstance(feature, (Covariate, IIV, Covariance)):
            features_to_remove.append(feature)
        else:
            types = search_space.get_feature_type(builtins.type(feature))
            if types:
                features_to_add.append(min(types))

    if type is None or type == 'iiv':
        mutex_dict = defaultdict(list)
        for feature in search_space.iiv:
            feature = feature.replace(optional=False)
            mutex_dict[feature.parameter].append(feature)
    else:
        mutex_dict = dict()

    features_not_in_model = search_space - model_features
    features_not_in_model = features_not_in_model.filter(filter_on='forced')
    for feature in features_not_in_model:
        # Will have been handled in features_not_in_search_space
        if not hasattr(feature, 'optional'):
            continue

        if (type is None or type == 'iiv') and isinstance(feature, IIV):
            other_features = mutex_dict.get(feature.parameter)
            # Handle multiple types of fps for IIVs, e.g. exp cannot be combined with add
            if any(f in model_features for f in other_features):
                continue
            min_feature = sorted(other_features, reverse=True)[0]
            if feature != min_feature:
                continue
        if feature not in model_features:
            features_to_add.append(feature)

    return features_to_add, features_to_remove


def is_in_search_space(
    model: Model,
    search_space: Union[ModelFeatures, Sequence[ModelFeature]],
    type: Optional[Literal['pk', 'covariates', 'iiv', 'covariance']] = None,
):
    if isinstance(search_space, Sequence):
        search_space = ModelFeatures.create(search_space)
    search_space = _get_mfl_from_type(search_space, type)
    model_features = get_model_features(model, type=type)

    if model_features == search_space:
        return True
    if model_features not in search_space:
        return False

    features_to_add, features_to_remove = _get_feature_diffs(search_space, model_features, type)

    if features_to_add or features_to_remove:
        return False
    else:
        return True
