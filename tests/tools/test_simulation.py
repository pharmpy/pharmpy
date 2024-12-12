import pytest

from pharmpy.model import ExecutionSteps
from pharmpy.modeling import set_simulation
from pharmpy.tools.simulation.tool import validate_input


def test_validate_input_raises(load_example_model_for_test):
    model = load_example_model_for_test("pheno")

    with pytest.raises(Exception):
        validate_input(model)

    simmodel = set_simulation(model)
    validate_input(simmodel)

    nostepmodel = model.replace(execution_steps=ExecutionSteps())

    with pytest.raises(Exception):
        validate_input(nostepmodel)
