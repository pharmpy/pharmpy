import itertools

from pharmpy.modeling import (
    get_observations,
    remove_peripheral_compartment,
    set_initial_estimates,
    set_name,
    set_tmdd,
)


def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))


def create_qss_models(model, ests, index=1):
    # Create qss models with different initial estimates from basic pk model
    qss_base_model = set_tmdd(model, type="QSS")
    cmax = get_observations(model).max()
    all_inits = product_dict(
        POP_KDEG=(0.5623, 17.28), POP_R_0=(0.001 * cmax, 0.01 * cmax, 0.1 * cmax, 1 * cmax)
    )
    qss_candidate_models = [
        set_initial_estimates(set_name(qss_base_model, f"QSS{i}"), inits)
        for i, inits in enumerate(all_inits, start=index)
    ]
    if "POP_KM" in ests.keys():
        qss_candidate_models = [
            set_initial_estimates(
                model,
                {
                    "POP_KDC": ests["POP_KM"],
                },
            )
            for model in qss_candidate_models
        ]
    if "POP_KM" in ests.keys() and "POP_CLMM" in ests.keys():
        qss_candidate_models = [
            set_initial_estimates(
                model,
                {
                    "POP_KINT": ests["POP_KM"]
                    * ests["POP_CLMM"]
                    / model.parameters["POP_R_0"].init,
                },
            )
            for model in qss_candidate_models
        ]
    if "IIV_CLMM" in ests.keys():
        qss_candidate_models = [
            set_initial_estimates(model, {"IIV_R_0": ests["IIV_CLMM"]})
            for model in qss_candidate_models
        ]
    else:
        qss_candidate_models = [
            set_initial_estimates(model, {"IIV_R_0": 0.04}) for model in qss_candidate_models
        ]
    return qss_candidate_models


def create_remaining_models(model, ests, parent_model, num_peripherals_qss):
    # if best qss model has fewer compartments than model, remove one compartment
    num_peripherals_model = len(model.statements.ode_system.find_peripheral_compartments())
    if num_peripherals_qss < num_peripherals_model:
        model = remove_peripheral_compartment(model)

    if 'POP_VC' in ests:
        est_vc = ests['POP_VC']
    else:
        est_vc = 0.1

    models = (
        create_full_models(model, ests, parent_model, est_vc)
        + create_cr_models(model, ests, parent_model, est_vc)
        + create_ib_models(model, ests, parent_model, est_vc)
        + create_crib_models(model, ests, parent_model, est_vc)
        + create_wagner_model(model, ests, parent_model, est_vc)
        + create_mmapp_model(model, ests, parent_model, est_vc)
    )
    return models


def create_cr_models(model, ests, parent_model, est_vc):
    # Create cr models with different initial estimates from basic pk model and best qss ests
    cr_base_model = set_tmdd(model, type="CR")
    cr_base_model = set_initial_estimates(
        cr_base_model,
        {"POP_KINT": ests['POP_KINT'], "POP_R_0": ests['POP_R_0'], "IIV_R_0": ests['IIV_R_0']},
    )
    cr1 = set_name(cr_base_model, "structsearch_run5")
    cr1 = cr1.replace(description="CR1", parent_model=parent_model)
    cr1 = set_initial_estimates(
        cr1, {"POP_KOFF": 0.5623, "POP_KON": 0.5623 / (ests['POP_KDC'] * ests['POP_VC'])}
    )
    cr2 = set_name(cr_base_model, "structsearch_run6")
    cr2 = cr2.replace(description="CR2", parent_model=parent_model)
    cr2 = set_initial_estimates(
        cr2, {"POP_KOFF": 17.78, "POP_KON": 17.78 / (ests['POP_KDC'] * ests['POP_VC'])}
    )
    return [cr1, cr2]


def create_ib_models(model, ests, parent_model, est_vc):
    # Create ib models with different initial estimates from basic pk model and best qss ests
    ib_base_model = set_tmdd(model, type="IB")
    ib_base_model = set_initial_estimates(
        ib_base_model,
        {
            "POP_KINT": ests['POP_KINT'],
            "POP_R_0": ests['POP_R_0'],
            "POP_KDEG": ests['POP_KDEG'],
            "IIV_R_0": ests['IIV_R_0'],
        },
    )
    ib1 = set_name(ib_base_model, "structsearch_run7")
    ib1 = ib1.replace(description="IB1", parent_model=parent_model)
    ib1 = set_initial_estimates(ib1, {"POP_KON": 0.5623 / (ests['POP_KDC'] * ests['POP_VC'])})
    ib2 = set_name(ib_base_model, "structsearch_run8")
    ib2 = ib2.replace(description="IB2", parent_model=parent_model)
    ib2 = set_initial_estimates(ib2, {"POP_KON": 17.78 / (ests['POP_KDC'] * ests['POP_VC'])})
    return [ib1, ib2]


def create_crib_models(model, ests, parent_model, est_vc):
    # Create crib models with different initial estimates from basic pk model and best qss ests
    crib_base_model = set_tmdd(model, type="CRIB")
    crib_base_model = set_initial_estimates(
        crib_base_model,
        {
            "POP_KINT": ests['POP_KINT'],
            "POP_R_0": ests['POP_R_0'],
            "IIV_R_0": ests['IIV_R_0'],
            "POP_KDEG": ests['POP_KDEG'],
        },
    )
    crib1 = set_name(crib_base_model, "structsearch_run9")
    crib1 = crib1.replace(description="CR+IB1", parent_model=parent_model)
    crib1 = set_initial_estimates(crib1, {"POP_KON": 0.5623 / (ests['POP_KDC'] * ests['POP_VC'])})
    crib2 = set_name(crib_base_model, "structsearch_run10")
    crib2 = crib2.replace(description="CR+IB2", parent_model=parent_model)
    crib2 = set_initial_estimates(crib2, {"POP_KON": 17.78 / (ests['POP_KDC'] * ests['POP_VC'])})
    return [crib1, crib2]


def create_full_models(model, ests, parent_model, est_vc):
    # Create full models with different initial estimates from basic pk model and best qss ests
    full_base_model = set_tmdd(model, type="FULL")
    full_base_model = set_initial_estimates(
        full_base_model,
        {
            "POP_KINT": ests['POP_KINT'],
            "POP_R_0": ests['POP_R_0'],
            "IIV_R_0": ests['IIV_R_0'],
            "POP_KDEG": ests['POP_KDEG'],
        },
    )
    candidates = [
        set_initial_estimates(full_base_model, {'POP_KOFF': koff}) for koff in (0.1, 1, 10, 100)
    ]
    candidates = [
        set_initial_estimates(
            model, {'POP_KON': model.parameters['POP_KOFF'].init / ests['POP_KDC'] * ests['POP_VC']}
        )
        for model in candidates
    ]
    candidates = [set_name(model, f"structsearch_run{i}") for i, model in enumerate(candidates, 1)]
    candidates = [
        m.replace(parent_model=parent_model, description=f"FULL{i}")
        for i, m in enumerate(candidates, 1)
    ]
    return candidates


def create_wagner_model(model, ests, parent_model, est_vc):
    wagner = set_tmdd(model, type="WAGNER")
    wagner = set_name(wagner, "structsearch_run11")
    wagner = wagner.replace(description="WAGNER", parent_model=parent_model)
    wagner = set_initial_estimates(
        wagner,
        {
            "POP_KINT": ests['POP_KINT'],
            "POP_R_0": ests['POP_R_0'],
            "IIV_R_0": ests['IIV_R_0'],
            "POP_KM": ests['POP_KDC'] * ests['POP_VC'],
        },
    )
    return [wagner]


def create_mmapp_model(model, ests, parent_model, est_vc):
    mmapp = set_tmdd(model, type="MMAPP")
    mmapp = set_name(mmapp, "structsearch_run12")
    mmapp = mmapp.replace(description="MMAPP", parent_model=parent_model)
    mmapp = set_initial_estimates(
        mmapp,
        {
            "POP_KINT": ests['POP_KINT'],
            "POP_R_0": ests['POP_R_0'],
            "IIV_R_0": ests['IIV_R_0'],
            "POP_KDEG": ests['POP_KDEG'],
        },
    )
    return [mmapp]
