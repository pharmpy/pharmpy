import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.modeling import (
    calculate_parameters_from_ucp,
    calculate_ucp_scale,
    set_initial_estimates,
)
from pharmpy.tools import run_retries


@pytest.mark.parametrize(
    ('scale',),
    (('UCP',),),
)
def test_retries(tmp_path, model_count, scale, start_modelres):
    with chdir(tmp_path):
        fraction = 0.1
        number_of_candidates = 2
        res = run_retries(
            number_of_candidates=number_of_candidates,
            fraction=fraction,
            scale=scale,
            results=start_modelres[1],
            model=start_modelres[0],
        )

        # All candidate models + start model
        assert len(res.summary_tool) == 3
        assert len(res.summary_models) == 3
        assert len(res.models) == 3
        for model in res.models:
            if model != start_modelres[0]:
                is_within_fraction(start_modelres[0], start_modelres[1], model, scale, fraction)
        rundir = tmp_path / 'retries1'
        assert rundir.is_dir()
        assert model_count(rundir) == 2 + 2
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'metadata.json').exists()


def is_within_fraction(start_model, start_model_res, candidate_model, scale, fraction):
    parameter_value = list(start_model_res.parameter_estimates.items())

    allowed_dict = {}
    if scale == "normal":
        for parameter, value in parameter_value:
            allowed_dict[parameter] = (
                value - value * fraction,
                value + value * fraction,
            )
    elif scale == "UCP":
        start_model = set_initial_estimates(start_model, start_model_res.parameter_estimates)
        ucp_scale = calculate_ucp_scale(start_model)
        lower = {}
        upper = {}
        for parameter, _ in parameter_value:
            lower[parameter] = 0.1 - (0.1 * fraction)
            upper[parameter] = 0.1 + (0.1 * fraction)
        new_lower_parameters = calculate_parameters_from_ucp(start_model, ucp_scale, lower)
        new_upper_parameters = calculate_parameters_from_ucp(start_model, ucp_scale, upper)
        for parameter, _ in parameter_value:
            allowed_dict[parameter] = (
                new_lower_parameters[parameter],
                new_upper_parameters[parameter],
            )
    for parameter in candidate_model.parameters:
        assert allowed_dict[parameter.name][0] < parameter.init < allowed_dict[parameter.name][1]
