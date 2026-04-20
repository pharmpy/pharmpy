from pharmpy.basic import Expr
from pharmpy.model import AddColumn, Drop, Ignore
from pharmpy.tools import read_modelfit_results
from pharmpy.tools.vpc.frem import prepare_evaluation_model, prepare_frem_model
from pharmpy.workflows import ModelEntry


def test_frem_prepare_evaluation_model(tmp_path, load_model_for_test, testdata):
    model_path = testdata / 'nonmem' / 'frem' / 'pheno' / 'model_4.mod'
    model = load_model_for_test(model_path)
    mfr = read_modelfit_results(model_path)
    me = ModelEntry.create(model, modelfit_results=mfr)
    me_eval = prepare_evaluation_model(me)
    model_eval = me_eval.model
    assert len(model_eval.execution_steps) == 1
    assert model_eval.execution_steps[0].evaluation
    param_ests = mfr.parameter_estimates.to_dict()
    param_inits = model_eval.parameters.inits
    assert all(param_inits[p] == init for p, init in param_ests.items())
    assert not (model_eval.dataset['FREMTYPE'] == 0).any()
    assert model_eval.datainfo.provenance[-1] == Ignore.create('FREMTYPE == 0')


def test_frem_prepare_frem_model(tmp_path, load_model_for_test, testdata, model_entry_factory):
    model_path = testdata / 'nonmem' / 'frem' / 'pheno' / 'model_4.mod'
    model = load_model_for_test(model_path)
    mfr = read_modelfit_results(model_path)
    me = ModelEntry.create(model, modelfit_results=mfr)
    me_eval = prepare_evaluation_model(me)
    me_eval = model_entry_factory([me_eval.model])[0]
    me_frem = prepare_frem_model(model, me_eval)
    model_frem = me_frem.model
    assert Expr.symbol('ETA_1_C') in model_frem.statements.lhs_symbols
    assert 'FREMTYPE' not in model_frem.datainfo.names
    assert model_frem.datainfo.provenance[1] == Ignore.create('FREMTYPE != 0')
    assert AddColumn.create('ET_1') in model_frem.datainfo.provenance
    assert model_frem.datainfo.provenance[-1] == Drop.create('FREMTYPE')
