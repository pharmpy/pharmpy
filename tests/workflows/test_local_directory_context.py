import pytest

from pharmpy.tools import load_example_modelfit_results
from pharmpy.workflows import LocalDirectoryContext
from pharmpy.workflows.hashing import ModelHash
from pharmpy.workflows.model_entry import ModelEntry
from pharmpy.workflows.results import read_results


def test_init(tmp_path):
    ctx = LocalDirectoryContext('mycontext', tmp_path)
    assert ctx.path == tmp_path / 'mycontext'
    assert (ctx.path / 'models').is_dir()
    assert (ctx.path / '.modeldb').is_dir()
    assert (ctx.path / 'log').is_file()
    assert (ctx.path / 'annotations').is_file()
    assert ctx.context_path == 'mycontext'

    subctx = ctx.create_subcontext("mysubcontext")
    assert (subctx.path / 'models').is_dir()
    assert not (subctx.path / '.modeldb').is_dir()
    assert not (subctx.path / 'log.csv').is_file()
    assert (subctx.path / 'annotations').is_file()
    assert subctx.context_path == 'mycontext/mysubcontext'

    parent = subctx.get_parent_context()
    assert parent.context_path == ctx.context_path

    sub = ctx.get_subcontext("mysubcontext")
    with pytest.raises(ValueError):
        ctx.get_subcontext("nonexisting")

    assert sub.context_path == 'mycontext/mysubcontext'

    existing_ctx = LocalDirectoryContext('mycontext', tmp_path)
    assert existing_ctx.path == ctx.path

    subsubctx = subctx.create_subcontext("nextlevel")
    assert subsubctx.context_path == 'mycontext/mysubcontext/nextlevel'


def test_metadata(tmp_path):
    ctx = LocalDirectoryContext(name='mycontext', ref=tmp_path)
    d = {'mymeta': 23, 'other': 'ext'}
    ctx.store_metadata(d)
    retd = ctx.retrieve_metadata()
    assert d == retd


def test_common_options(tmp_path):
    opts = {'ref': 23}
    ctx = LocalDirectoryContext(name='mycontext', ref=tmp_path, common_options=opts)
    assert ctx.retrieve_common_options() == opts
    subctx = ctx.create_subcontext("mysubcontext")
    assert subctx.retrieve_common_options() == opts


def test_log(tmp_path):
    ctx = LocalDirectoryContext(name='mycontext', ref=tmp_path)
    ctx.log_message('error', "This didn't work")
    ctx.log_message('warning', "Potential disaster")
    df = ctx.retrieve_log()
    assert tuple(df.columns) == ('path', 'time', 'severity', 'message')
    assert tuple(df['path']) == ('mycontext', 'mycontext')
    assert tuple(df['severity']) == ('error', 'warning')
    assert df.loc[1, 'message'] == "Potential disaster"

    subctx = ctx.create_subcontext("mysubcontext")
    subctx.log_message('error', "Neither did this")
    df = subctx.retrieve_log()
    assert len(df) == 3
    assert df.loc[2, 'path'] == 'mycontext/mysubcontext'

    df = subctx.retrieve_log(level='current')
    assert len(df) == 1
    assert df.loc[0, 'path'] == 'mycontext/mysubcontext'
    df = subctx.retrieve_log(level='lower')
    assert len(df) == 1
    df = ctx.retrieve_log(level='current')
    assert len(df) == 2

    s = 'String, with, commas'
    s2 = '"Quoted"'
    ctx.log_message('error', s)
    ctx.log_message('error', s2)
    df = ctx.retrieve_log()
    assert df.loc[3, 'message'] == s
    assert df.loc[4, 'message'] == s2


def test_results(tmp_path, testdata):
    ctx = LocalDirectoryContext(name='mycontext', ref=tmp_path)
    res = read_results(testdata / 'results' / 'estmethod_results.json')
    ctx.store_results(res)
    newres = ctx.retrieve_results()
    assert tuple(newres.summary_models.iloc[0]) == tuple(res.summary_models.iloc[0])


def test_store_model(tmp_path, load_example_model_for_test):
    ctx = LocalDirectoryContext(name='myconext', ref=tmp_path)
    model = load_example_model_for_test("pheno")
    ctx.store_model_entry(model)
    me = ctx.retrieve_model_entry("pheno")
    assert me.model == model
    assert me.modelfit_results is None

    res = load_example_modelfit_results("pheno")
    me = ModelEntry.create(model, modelfit_results=res)
    ctx.store_model_entry(me)
    newme = ctx.retrieve_model_entry("pheno")
    assert newme.model == me.model
    assert dict(newme.modelfit_results.parameter_estimates) == dict(
        me.modelfit_results.parameter_estimates
    )
    assert newme.model.name == "pheno"

    ctx.store_input_model_entry(me)
    inputme = ctx.retrieve_input_model_entry()
    assert inputme.model.name == "input"
    assert inputme.model.parameters == newme.model.parameters

    ctx.store_final_model_entry(me)
    finalme = ctx.retrieve_final_model_entry()
    assert finalme.model.name == "final"
    assert finalme.model.parameters == newme.model.parameters


def test_key(tmp_path, load_example_model_for_test):
    ctx = LocalDirectoryContext(name='mycontext', ref=tmp_path)
    model = load_example_model_for_test("pheno")
    ctx.store_model_entry(model)
    key = ctx.retrieve_key("pheno")
    assert isinstance(key, ModelHash)
    name = ctx.retrieve_name(key)
    assert name == "pheno"
    annotation = ctx.retrieve_annotation("pheno")
    assert annotation.startswith("PHENOBARB")
