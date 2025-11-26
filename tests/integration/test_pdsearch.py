import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.tools import run_pdsearch


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
        run_pdsearch(dataset=dataset_path, type=type, esttool='dummy', **kwargs)
