import pytest

from pharmpy.modeling import create_rng
from pharmpy.tools.external.dummy.run import create_dummy_modelfit_results
from pharmpy.workflows import ModelEntry
from pharmpy.workflows.results import ModelfitResults


@pytest.fixture(scope='session')
def model_entry_factory():
    def _model_entry_factory(models, ref_val=None):
        model_entries = []
        for i, model in enumerate(models):
            res = create_dummy_modelfit_results(model)
            if ref_val:
                res_attrs = res.__dict__
                rng = create_rng(i)
                low = ref_val / 2 if ref_val > 0 else ref_val * 2
                res_attrs['ofv'] = rng.uniform(low, ref_val)
                res = ModelfitResults(**res_attrs)
            model_entries.append(ModelEntry.create(model, modelfit_results=res))
        return model_entries

    return _model_entry_factory
