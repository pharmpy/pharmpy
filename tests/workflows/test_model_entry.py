import pytest

from pharmpy.tools.external.results import parse_modelfit_results
from pharmpy.workflows import ModelEntry


def test_model_entry_init(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')

    model_entry = ModelEntry(model)
    assert model_entry.model.name == model.name


def test_model_entry_create(load_model_for_test, testdata):
    path = testdata / 'nonmem' / 'pheno.mod'
    model = load_model_for_test(path)
    res = parse_modelfit_results(model, path)

    model_entry = ModelEntry.create(model)
    assert model_entry.model.name == model.name

    model_entry_res = ModelEntry(model, modelfit_results=res)
    assert model_entry_res.model.name == model.name

    model_parent = model.replace(name='parent')
    model_entry_parent = ModelEntry.create(model, parent=model_parent)
    assert model_entry_parent.parent.name == model_parent.name

    with pytest.raises(ValueError):
        ModelEntry.create(model, parent=model)


def test_attach_results(load_model_for_test, testdata):
    path = testdata / 'nonmem' / 'pheno.mod'
    model = load_model_for_test(path)
    res = parse_modelfit_results(model, path)

    model_entry = ModelEntry(model)
    assert model_entry.model.name == model.name

    model_entry_res = model_entry.attach_results(modelfit_results=res)
    assert model_entry_res.modelfit_results.ofv == 730.8947268137307

    with pytest.raises(ValueError):
        model_entry_res.attach_results(modelfit_results=res)


def test_repr(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')

    model_entry = ModelEntry(model)

    assert repr(model_entry) == '<Pharmpy model entry object pheno>'
