import pytest


@pytest.mark.parametrize("buf,add,kwargs,expected", [
    ('$MODEL NCOMPARTMENTS=23', 'CENTRAL', {}, '$MODEL NCOMPARTMENTS=23 COMPARTMENT=(CENTRAL)'),
    ('$MODEL NCOMPARTMENTS=23\n', 'CENTRAL', {}, '$MODEL NCOMPARTMENTS=23 COMPARTMENT=(CENTRAL)\n'),
    ('$MODEL NCOM=1', 'CENTRAL', {'dosing': True}, '$MODEL NCOM=1 COMPARTMENT=(CENTRAL DEFDOSE)'),
    ('$MODEL ', 'CENTRAL', {}, '$MODEL COMPARTMENT=(CENTRAL)'),
    ])
def test_add_compartment(parser, buf, add, kwargs, expected):
    recs = parser.parse(buf)
    rec = recs.records[0]
    rec.add_compartment('CENTRAL', **kwargs)
    assert str(rec) == expected
