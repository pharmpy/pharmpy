import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.modeling import set_simulation
from pharmpy.tools import run_simulation
from pharmpy.workflows.results import SimulationResults


@pytest.mark.parametrize(
    'esttool',
    ['nonmem', 'dummy'],
)
def test_sim(tmp_path, load_model_for_test, testdata, esttool):
    with chdir(tmp_path):
        model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
        model = set_simulation(model, n=10)
        res = run_simulation(model, esttool=esttool)
        assert isinstance(res, SimulationResults)
        assert len(res.table) == 10 * len(model.dataset)
