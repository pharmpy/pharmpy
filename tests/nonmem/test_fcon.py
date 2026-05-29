import pytest

from pharmpy.model.external.fcon.model import group_formats, read_multiline_observations


@pytest.mark.parametrize(
    'fcon, model_path, col',
    [
        (
            'pheno',
            'pheno_real.mod',
            'WGT',
        ),
        (
            'mox2',
            'models/mox2.mod',
            'TIME',
        ),
    ],
)
def test_dataset(load_model_for_test, testdata, fcon, model_path, col):
    model = load_model_for_test(testdata / 'nonmem' / 'fcon' / fcon / 'FCON')
    assert model.internals.code.startswith('FILE')

    df = model.dataset
    assert df.notna().all().all()
    assert df['ID'].dtype == 'int32'

    nmtran = load_model_for_test(testdata / 'nonmem' / model_path)
    assert df.index.equals(nmtran.dataset.index)
    assert list(df[col]) == list(nmtran.dataset[col])


def test_read_multiline_observations(testdata):
    path = testdata / 'nonmem' / 'fcon' / 'mox2' / 'FDATA'
    labels = [
        'ID',
        'VISI',
        'DGRP',
        'DOSE',
        'NEUY',
        'SCR',
        'AGE',
        'SEX',
        'WT',
        'ACE',
        'DIG',
        'DIU',
        'TAD',
        'TIME',
        'CLCR',
        'AMT',
        'SS',
        'II',
        'DV',
        'EVID',
        'MDV',
    ]
    formats = ['4(4E19.0/)', '3E19.0', '2F2.0']
    df = read_multiline_observations(path, labels, formats)
    assert len(df) == 1148
    assert df.notna().all().all()


def test_group_formats():
    formats = ['4(4E19.0/)', '3E19.0', '2F2.0']
    assert group_formats(formats) == [
        ['4E19.0'],
        ['4E19.0'],
        ['4E19.0'],
        ['4E19.0'],
        ['3E19.0', '2F2.0'],
    ]
