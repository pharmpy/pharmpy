import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.modeling import convert_model, create_basic_kpd_model, create_basic_pd_model
from pharmpy.tools import fit, run_pdsearch


@pytest.mark.parametrize(
    'type, kwargs',
    [('pd', {'treatment_variable': 'TIME'}), ('kpd', {'kpd_driver': 'ir'})],
)
def test_pdsearch_dummy(
    tmp_path,
    testdata,
    type,
    kwargs,
):
    with chdir(tmp_path):
        dataset_path = testdata / 'nonmem' / 'pheno.dta'
        run_pdsearch(input=dataset_path, type=type, esttool='dummy', **kwargs)


@pytest.mark.parametrize(
    'type, kwargs',
    [('pd', {'treatment_variable': 'TIME'}), ('kpd', {'kpd_driver': 'ir'})],
)
def test_pdsearch_from_model_dummy(
    tmp_path,
    testdata,
    type,
    kwargs,
):
    with chdir(tmp_path):
        dataset_path = testdata / 'nonmem' / 'pheno.dta'
        if type == 'pd':
            model = create_basic_pd_model(dataset_path)
        else:
            model = create_basic_kpd_model(dataset_path)
        model = convert_model(model, 'nonmem')  # Needed to parse model/results
        res = fit(model, esttool='dummy')
        run_pdsearch(input=model, type=type, results=res, esttool='dummy', **kwargs)
