import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.modeling import calculate_parameters_from_ucp, calculate_ucp_scale
from pharmpy.tools import run_retries


@pytest.mark.parametrize(
    ('scale',),
    (
        ('UCP',),
        ('normal',),
    ),
)
def test_retries(tmp_path, model_count, scale, start_model):
    with chdir(tmp_path):
        degree = 0.1
        number_of_candidates = 5
        res = run_retries(
            number_of_candidates=number_of_candidates,
            degree=degree,
            scale=scale,
            results=start_model.modelfit_results,
            model=start_model,
        )

        # All candidate models + start model
        assert len(res.summary_tool) == 6
        assert len(res.summary_models) == 6
        assert len(res.models) == 6
        for model in res.models:
            is_within_degree(start_model, model, scale, degree)
        rundir = tmp_path / 'retries_dir1'
        assert rundir.is_dir()
        assert model_count(rundir) == 5  # Not the start model ?
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'metadata.json').exists()


def is_within_degree(start_model, candidate_model, scale, degree):
    allowed_dict = {}
    if scale == "normal":
        for parameter in start_model.parameters:
            allowed_dict[parameter.name] = (
                parameter.init - parameter.init * degree,
                parameter.init + parameter.init * degree,
            )
    elif scale == "UCP":
        ucp_scale = calculate_ucp_scale(start_model)
        lower = {}
        upper = {}
        for p in start_model.parameters:
            lower[p.name] = 0.1 - (0.1 * degree)
            upper[p.name] = 0.1 + (0.1 * degree)
        new_lower_parameters = calculate_parameters_from_ucp(start_model, ucp_scale, lower)
        new_upper_parameters = calculate_parameters_from_ucp(start_model, ucp_scale, upper)
        for p in start_model.parameters:
            allowed_dict[p.name] = (new_lower_parameters[p.name], new_upper_parameters[p.name])
    for parameter in candidate_model.parameters:
        assert allowed_dict[parameter.name][0] < parameter.init < allowed_dict[parameter.name][1]
