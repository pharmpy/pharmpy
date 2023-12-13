import pytest

from pharmpy.tools.retries.tool import create_new_parameter_inits


@pytest.mark.parametrize(
    ('scale',),
    (('UCP',), ('normal',)),
)
def test_parameter_inits(pheno, scale):
    fraction = 0.1
    seed = 13
    new_parameters = create_new_parameter_inits(pheno, fraction, scale, seed)

    for parameter, value in new_parameters.items():
        param_init = pheno.parameters[parameter].init
        upper = param_init + fraction * param_init
        lower = param_init - fraction * param_init
        assert lower < value < upper
