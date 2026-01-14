import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.tools import run_linearize
from pharmpy.tools.context import open_context
from pharmpy.tools.run import read_modelfit_results


def test_create_linearized_model(tmp_path, load_model_for_test, testdata):
    with chdir(tmp_path):
        path = testdata / 'nonmem' / 'pheno_real.mod'
        model = load_model_for_test(path)
        res = read_modelfit_results(path)
        linear_results = run_linearize(model, results=res, name='linearize1')
        assert len(linear_results.final_model.statements) == 9
        ctx = open_context('linearize1', tmp_path)
        derivative_me = ctx.retrieve_model_entry('derivatives')
        input_ofv = res.ofv
        derivative_ofv = derivative_me.modelfit_results.ofv_iterations.iloc[0]
        assert derivative_ofv == pytest.approx(input_ofv, abs=0.5)
        assert linear_results.final_model_results.ofv == pytest.approx(input_ofv, abs=0.5)


def test_run_linearize_fo(tmp_path, load_model_for_test, testdata):
    with chdir(tmp_path):
        path = testdata / 'nonmem' / 'models' / 'mox2.mod'
        model = load_model_for_test(path)
        res = read_modelfit_results(path)
        linear_results = run_linearize(model, results=res, name='linearize1')
        assert len(linear_results.final_model.statements) == 11
        ctx = open_context('linearize1', tmp_path)
        derivative_me = ctx.retrieve_model_entry('derivatives')
        input_ofv = res.ofv
        derivative_ofv = derivative_me.modelfit_results.ofv_iterations.iloc[0]
        assert derivative_ofv != pytest.approx(input_ofv)
