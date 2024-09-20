import pytest

from pharmpy.tools.external.dummy.run import create_dummy_modelfit_results
from pharmpy.workflows import ModelEntry


@pytest.fixture(scope='session')
def model_entry_factory():
    def _model_entry_factory(models):
        model_entries = []
        for i, model in enumerate(models):
            res = create_dummy_modelfit_results(model, i)
            model_entries.append(ModelEntry.create(model, modelfit_results=res))
        return model_entries

    return _model_entry_factory
