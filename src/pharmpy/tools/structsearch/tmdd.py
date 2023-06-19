import itertools

from pharmpy.modeling import get_observations, set_initial_estimates, set_name, set_tmdd


def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))


def create_qss_models(model):
    # Create qss models with different initial estimates from basic pk model
    qss_base_model = set_tmdd(model, type="QSS")
    cmax = get_observations(model).max()
    all_inits = product_dict(
        POP_KDEG=(0.5623, 17.28), POP_R_0=(0.001 * cmax, 0.01 * cmax, 0.1 * cmax, 1 * cmax)
    )
    qss_candidate_models = [
        set_initial_estimates(set_name(qss_base_model, f"QSS{i}"), inits)
        for i, inits in enumerate(all_inits, start=1)
    ]
    return qss_candidate_models


def create_remaining_models(model, ests):
    models = (
        create_full_models(model, ests)
        + create_cr_models(model, ests)
        + create_ib_models(model, ests)
        + create_crib_models(model, ests)
        + create_wagner_model(model, ests)
        + create_mmapp_model(model, ests)
    )
    return models


def create_cr_models(model, ests):
    # Create cr models with different initial estimates from basic pk model and best qss ests
    cr_base_model = set_tmdd(model, type="CR")
    cr_base_model = set_initial_estimates(
        cr_base_model,
        {"POP_KINT": ests['POP_KINT'], "POP_R0": ests['POP_R0'], "IIV_R0": ests['IIV_R0']},
    )
    cr1 = set_name(cr_base_model, "CR1")
    cr1 = set_initial_estimates(
        cr_base_model, {"POP_KOFF": 0.5623, "POP_KON": 0.5623 / (ests['POP_KDC'] * ests['POP_VC'])}
    )
    cr2 = set_name(cr_base_model, "CR2")
    cr2 = set_initial_estimates(
        cr_base_model, {"POP_KOFF": 17.78, "POP_KON": 17.78 / (ests['POP_KDC'] * ests['POP_VC'])}
    )
    return [cr1, cr2]


def create_ib_models(model, ests):
    # Create ib models with different initial estimates from basic pk model and best qss ests
    ib_base_model = set_tmdd(model, type="IB")
    ib_base_model = set_initial_estimates(
        ib_base_model,
        {
            "POP_KINT": ests['POP_KINT'],
            "POP_R0": ests['POP_R0'],
            "POP_KDEG": ests['POP_KDEG'],
            "IIV_R0": ests['IIV_R0'],
        },
    )
    ib1 = set_name(ib_base_model, "IB1")
    ib1 = set_initial_estimates(
        ib_base_model, {"POP_KON": 0.5623 / (ests['POP_KDC'] * ests['POP_VC'])}
    )
    ib2 = set_name(ib_base_model, "IB2")
    ib2 = set_initial_estimates(
        ib_base_model, {"POP_KON": 17.78 / (ests['POP_KDC'] * ests['POP_VC'])}
    )
    return [ib1, ib2]


def create_crib_models(model, ests):
    # Create crib models with different initial estimates from basic pk model and best qss ests
    crib_base_model = set_tmdd(model, type="IB")
    crib_base_model = set_initial_estimates(
        crib_base_model,
        {"POP_KINT": ests['POP_KINT'], "POP_R0": ests['POP_R0'], "IIV_R0": ests['IIV_R0']},
    )
    crib1 = set_name(crib_base_model, "CRIB1")
    crib1 = set_initial_estimates(
        crib_base_model, {"POP_KON": 0.5623 / (ests['POP_KDC'] * ests['POP_VC'])}
    )
    crib2 = set_name(crib_base_model, "CRIB2")
    crib2 = set_initial_estimates(
        crib_base_model, {"POP_KON": 17.78 / (ests['POP_KDC'] * ests['POP_VC'])}
    )
    return [crib1, crib2]


def create_full_models(model, ests):
    # Create full models with different initial estimates from basic pk model and best qss ests
    full_base_model = set_tmdd(model, type="FULL")
    full_base_model = set_initial_estimates(
        full_base_model,
        {
            "POP_KINT": ests['POP_KINT'],
            "POP_R0": ests['POP_R0'],
            "IIV_R0": ests['IIV_R0'],
            "POP_KDEG": ests['POP_KDEG'],
            "POP_KON": 0.1 / (ests['POP_KDEG'] * ests['POP_VC']),
        },
    )
    candidates = [
        set_initial_estimates(full_base_model, {'POP_KOFF': koff}) for koff in (0.1, 1, 10, 100)
    ]
    return candidates


def create_wagner_model(model, ests):
    wagner = set_tmdd(model, type="WAGNER")
    wagner = set_name(wagner, "WAGNER")
    wagner = set_initial_estimates(
        wagner,
        {
            "POP_KINT": ests['POP_KINT'],
            "POP_R0": ests['POP_R0'],
            "IIV_R0": ests['IIV_R0'],
            "POP_KM": ests['POP_KDC'] * ests['POP_VC'],
        },
    )
    return [wagner]


def create_mmapp_model(model, ests):
    mmapp = set_tmdd(model, type="MMAPP")
    mmapp = set_name(mmapp, "MMAPP")
    mmapp = set_initial_estimates(
        mmapp,
        {
            "POP_KINT": ests['POP_KINT'],
            "POP_R0": ests['POP_R0'],
            "IIV_R0": ests['IIV_R0'],
            "POP_KDEG": ests['POP_KDEG'],
        },
    )
    return [mmapp]
