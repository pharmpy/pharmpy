import pytest

from pharmpy.deps import numpy as np
from pharmpy.internals.fs.cwd import chdir
from pharmpy.internals.immutable import frozenmapping
from pharmpy.modeling import (
    convert_model,
    create_basic_pd_model,
    create_basic_pk_model,
    set_zero_order_absorption,
)


def test_create_basic_pk_model(testdata, tmp_path):
    model = create_basic_pk_model('iv')
    assert len(model.parameters) == 6

    model = create_basic_pk_model('oral')
    assert len(model.parameters) == 8

    dataset_path = testdata / 'nonmem/pheno.dta'
    model = create_basic_pk_model(
        administration='oral', dataset_path=dataset_path, cl_init=0.01, vc_init=1.0, mat_init=0.1
    )
    model = convert_model(model, 'nonmem')
    run1 = set_zero_order_absorption(model)
    assert not model.dataset.empty
    assert not run1.dataset.empty
    assert isinstance(model.dependent_variables, frozenmapping)

    df = model.dataset
    df['CMT'] = np.random.randint(1, 3, size=len(df))

    model = create_basic_pk_model('ivoral')
    assert len(model.statements.ode_system.dosing_compartments) == 2
    assert model.datainfo.id_column

    with chdir(tmp_path):
        df.to_csv('pheno_cmt.csv', index_label=False, index=False)
        model = create_basic_pk_model('ivoral', tmp_path / 'pheno_cmt.csv')
        assert len(model.statements.ode_system.dosing_compartments) == 2

    dataset_path = testdata / 'nonmem/pheno_pd.csv'
    model = create_basic_pk_model(
        administration='oral', dataset_path=dataset_path, cl_init=0.01, vc_init=1.0, mat_init=0.1
    )
    assert not model.dataset.empty
    assert model.datainfo._separator == r'\s+'


def test_create_basic_pk_model_raises(testdata):
    dataset_path = testdata / 'nonmem/pheno.dta'

    with pytest.raises(ValueError):
        create_basic_pk_model('x')

    with pytest.raises(ValueError):
        create_basic_pk_model(
            administration='ivoral',
            dataset_path=dataset_path,
        )


def test_create_basic_pd_model(testdata):
    model = create_basic_pd_model()
    assert model.dataset is None
    assert len(model.parameters) == 2

    dataset_path = testdata / 'nonmem/pheno.dta'
    create_basic_pd_model(dataset_path)
