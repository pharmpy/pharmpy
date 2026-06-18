import pytest

from pharmpy.model.external.fcon.model import parse_formats, read_fdata_file
from pharmpy.modeling import filter_dataset


def test_dataset_pheno(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'fcon' / 'pheno' / 'FCON')
    assert model.internals.code.startswith('FILE')

    df = model.dataset
    assert df.notna().all().all()
    assert df['ID'].dtype == 'int32'

    nmtran = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    assert df.index.equals(nmtran.dataset.index)
    assert list(df['WGT']) == list(nmtran.dataset['WGT'])


def test_dataset_mox2(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'fcon' / 'mox2' / 'FCON')
    assert model.internals.code.startswith('FILE')

    df = model.dataset
    assert df.notna().all().all()
    assert df['ID'].dtype == 'int32'

    nmtran = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    nmtran = filter_dataset(nmtran, 'VISI == 3.0')
    assert df.index.equals(nmtran.dataset.index)
    assert list(df['TIME']) == list(nmtran.dataset['TIME'])


@pytest.mark.parametrize(
    'code, expected',
    [
        (
            '8E6.0,2F2.0',
            [['8E6.0', '2F2.0']],
        ),
        (
            '8E6.0,2(2F2.0)',
            [['8E6.0', '2F2.0', '2F2.0']],
        ),
        (
            '8E6.0/2F2.0',
            [['8E6.0'], ['2F2.0']],
        ),
        (
            '8E6.0,2F2.0/8E6.0',
            [['8E6.0', '2F2.0'], ['8E6.0']],
        ),
        (
            '3E19.0,2(4E19.0/),2F2.0',
            [['3E19.0', '4E19.0'], ['4E19.0'], ['2F2.0']],
        ),
        (
            '2(4E19.0/),3E19.0,2F2.0',
            [['4E19.0'], ['4E19.0'], ['3E19.0', '2F2.0']],
        ),
        (
            '2(4E19.0/),3E19.0,2F2.0,2(4E19.0/)',
            [['4E19.0'], ['4E19.0'], ['3E19.0', '2F2.0', '4E19.0'], ['4E19.0']],
        ),
        (
            '2(4E19.0/)/2F2.0',
            [['4E19.0'], ['4E19.0'], ['2F2.0']],
        ),
        (
            '2E19.0/2F2.0/',
            [['2E19.0'], ['2F2.0']],
        ),
    ],
)
def test_parse_formats(code, expected):
    assert parse_formats(code) == expected


def test_read_fdata_file(testdata):
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
    formats = parse_formats('4(4E19.0/),3E19.0,2F2.0')
    df = read_fdata_file(path, labels, formats)
    assert len(df) == 541
    assert df.notna().all().all()
