import pytest

from pharmpy.modeling import (
    calculate_parameters_from_ucp,
    calculate_ucp_scale,
    fix_parameters,
    get_omegas,
    get_sigmas,
    get_thetas,
)


def test_calculate_ucp_scale(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')

    ucp = calculate_ucp_scale(model)

    assert len(get_thetas(model)) == len(ucp.theta)
    for res, expected in zip(ucp.theta, [19.27717888, 13.90639125, 13.82933276]):
        assert abs(res - expected) <= 10**-4

    assert len(get_omegas(model)) == len(ucp.omega)
    for res, expected in zip(ucp.omega.flatten(), [0.15921694, 0.0, 0.0, 0.15964163]):
        assert abs(res - expected) <= 10**-4

    assert len(get_sigmas(model)) == len(ucp.sigma)
    for res, expected in zip(ucp.sigma.flatten(), [0.10411923]):
        assert abs(res - expected) <= 10**-4


def test_calculate_parameters_from_ucp(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')

    ucp_scale = calculate_ucp_scale(model)

    new_param = calculate_parameters_from_ucp(model, ucp_scale, model.parameters.inits)
    assert len(new_param) == 6

    model = fix_parameters(model, model.parameters.names)
    with pytest.raises(
        ValueError, match='Parameter "PTVCL" cannot both be fixed and given in ucps'
    ):
        calculate_parameters_from_ucp(model, ucp_scale, model.parameters.inits)

    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    new_parameters = model.parameters.inits
    new_parameters.pop("PTVCL")
    with pytest.raises(ValueError, match='Parameter "PTVCL" is neither fixed nor given in ucps.'):
        calculate_parameters_from_ucp(model, ucp_scale, new_parameters)
