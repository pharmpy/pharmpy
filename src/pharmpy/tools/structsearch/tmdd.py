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


def create_qss_models(model, ests, dv_types, index=1):
    # Create qss models with different initial estimates from basic pk model
    qss_base_model = set_tmdd(model, type="QSS", dv_types=dv_types)
    cmax = get_observations(model).max()
    all_inits = product_dict(
        POP_KDEG=(0.5623, 17.28), POP_R_0=(0.001 * cmax, 0.01 * cmax, 0.1 * cmax, 1 * cmax)
    )
    qss_candidate_models = [
        set_initial_estimates(set_name(qss_base_model, f"structsearch_run{i}"), inits)
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
    if 'POP_VC' in ests:
        est_vc = ests['POP_VC']
    else:
        est_vc = 0.1

    if "POP_KM" in ests.keys() and "POP_CLMM" in ests.keys():
        qss_candidate_models = [
            set_initial_estimates(
                model,
                {
                    "POP_KINT": ests["POP_KM"]
                    * ests["POP_CLMM"]
                    / (model.parameters["POP_R_0"].init * est_vc),
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
    qss_candidate_models = [
        model.replace(description=f"QSS{i}")
        for i, model in enumerate(qss_candidate_models, start=index)
    ]
    return qss_candidate_models


def create_remaining_models(model, ests, num_peripherals_qss, dv_types, index_offset=0):
    # if best qss model has fewer compartments than model, remove one compartment
    num_peripherals_model = len(model.statements.ode_system.find_peripheral_compartments())
    if num_peripherals_qss < num_peripherals_model:
        model = remove_peripheral_compartment(model)

    if dv_types is None:
        models = (
            create_full_models(model, ests, dv_types)
            + create_cr_models(model, ests, dv_types)
            + create_ib_models(model, ests, dv_types)
            + create_crib_models(model, ests, dv_types)
            + create_wagner_model(model, ests, dv_types)
            + create_mmapp_model(model, ests, dv_types)
        )
    else:
        if len(dv_types) < 2:
            raise ValueError('`dv_types` must contain more than 1 dv type')
        else:
            if (
                ('drug' in dv_types or 'drug_tot' in dv_types)
                and ('target' in dv_types or 'target_tot' in dv_types)
                and 'complex' not in dv_types
            ):
                models = (
                    create_full_models(model, ests, dv_types)
                    + create_ib_models(model, ests, dv_types)
                    + create_mmapp_model(model, ests, dv_types)
                )
            elif (
                ('drug' in dv_types or 'drug_tot' in dv_types)
                and 'complex' in dv_types
                and not ('target' in dv_types or 'target_tot' in dv_types)
            ):
                models = (
                    create_full_models(model, ests, dv_types)
                    + create_cr_models(model, ests, dv_types)
                    + create_ib_models(model, ests, dv_types)
                    + create_crib_models(model, ests, dv_types)
                    + create_wagner_model(model, ests, dv_types)
                )
            else:
                models = (
                    create_full_models(model, ests, dv_types)
                    + create_cr_models(model, ests, dv_types)
                    + create_ib_models(model, ests, dv_types)
                    + create_crib_models(model, ests, dv_types)
                    + create_wagner_model(model, ests, dv_types)
                    + create_mmapp_model(model, ests, dv_types)
                )
    models = [
        set_name(model, f'structsearch_run{i}') for i, model in enumerate(models, index_offset + 1)
    ]
    return models


def create_cr_models(model, ests, dv_types):
    # Create cr models with different initial estimates from basic pk model and best qss ests
    cr_base_model = set_tmdd(model, type="CR", dv_types=dv_types)
    cr_base_model = set_initial_estimates(
        cr_base_model,
        {"POP_KINT": ests['POP_KINT'], "POP_R_0": ests['POP_R_0'], "IIV_R_0": ests['IIV_R_0']},
    )
    cr1 = cr_base_model.replace(description="CR1")
    cr1 = set_initial_estimates(cr1, ests, strict=False)
    cr1 = set_initial_estimates(cr1, {"POP_KOFF": 0.5623, "POP_KON": 0.5623 / ests['POP_KDC']})
    cr2 = cr_base_model.replace(description="CR2")
    cr2 = set_initial_estimates(cr2, ests, strict=False)
    cr2 = set_initial_estimates(cr2, {"POP_KOFF": 17.78, "POP_KON": 17.78 / ests['POP_KDC']})
    return [cr1, cr2]


def create_ib_models(model, ests, dv_types):
    # Create ib models with different initial estimates from basic pk model and best qss ests
    ib_base_model = set_tmdd(model, type="IB", dv_types=dv_types)
    ib_base_model = set_initial_estimates(
        ib_base_model,
        {
            "POP_KINT": ests['POP_KINT'],
            "POP_R_0": ests['POP_R_0'],
            "POP_KDEG": ests['POP_KDEG'],
            "IIV_R_0": ests['IIV_R_0'],
        },
    )
    ib1 = ib_base_model.replace(description="IB1")
    ib1 = set_initial_estimates(ib1, ests, strict=False)
    ib1 = set_initial_estimates(ib1, {"POP_KON": 0.5623 / ests['POP_KDC']})
    ib2 = ib_base_model.replace(description="IB2")
    ib2 = set_initial_estimates(ib2, ests, strict=False)
    ib2 = set_initial_estimates(ib2, {"POP_KON": 17.78 / ests['POP_KDC']})
    return [ib1, ib2]


def create_crib_models(model, ests, dv_types):
    # Create crib models with different initial estimates from basic pk model and best qss ests
    crib_base_model = set_tmdd(model, type="CRIB", dv_types=dv_types)
    crib_base_model = set_initial_estimates(
        crib_base_model,
        {
            "POP_KINT": ests['POP_KINT'],
            "POP_R_0": ests['POP_R_0'],
            "IIV_R_0": ests['IIV_R_0'],
        },
        strict=False,
    )
    crib1 = crib_base_model.replace(description="CR+IB1")
    crib1 = set_initial_estimates(crib1, ests, strict=False)
    crib1 = set_initial_estimates(crib1, {"POP_KON": 0.5623 / ests['POP_KDC']})
    crib2 = crib_base_model.replace(description="CR+IB2")
    crib2 = set_initial_estimates(crib2, ests, strict=False)
    crib2 = set_initial_estimates(crib2, {"POP_KON": 17.78 / ests['POP_KDC']})
    return [crib1, crib2]


def create_full_models(model, ests, dv_types):
    # Create full models with different initial estimates from basic pk model and best qss ests
    full_base_model = set_tmdd(model, type="FULL", dv_types=dv_types)
    full_base_model = set_initial_estimates(full_base_model, ests, strict=False)
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
            model, {'POP_KON': model.parameters['POP_KOFF'].init / ests['POP_KDC']}
        )
        for model in candidates
    ]
    candidates = [m.replace(description=f"FULL{i}") for i, m in enumerate(candidates, 1)]
    return candidates


def create_wagner_model(model, ests, dv_types):
    wagner = set_tmdd(model, type="WAGNER", dv_types=dv_types)
    wagner = wagner.replace(description="WAGNER")
    wagner = set_initial_estimates(
        wagner,
        {
            "POP_KINT": ests['POP_KINT'],
            "POP_R_0": ests['POP_R_0'],
            "IIV_R_0": ests['IIV_R_0'],
            "POP_KM": ests['POP_KDC'],
        },
    )
    return [wagner]


def create_mmapp_model(model, ests, dv_types):
    mmapp = set_tmdd(model, type="MMAPP", dv_types=dv_types)
    mmapp = mmapp.replace(description="MMAPP")
    mmapp = set_initial_estimates(
        mmapp,
        {
            "POP_KINT": ests['POP_KINT'],
            "POP_R_0": ests['POP_R_0'],
            "IIV_R_0": ests['IIV_R_0'],
            "POP_KDEG": ests['POP_KDEG'],
            "POP_KM": ests['POP_KDC'],
        },
    )
    return [mmapp]
