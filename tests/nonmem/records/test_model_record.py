import pytest


@pytest.mark.parametrize(
    "buf,add,kwargs,expected",
    [
        ('$MODEL NCOMPARTMENTS=23', 'CENTRAL', {}, '$MODEL NCOMPARTMENTS=23 COMPARTMENT=(CENTRAL)'),
        (
            '$MODEL NCOMPARTMENTS=23\n',
            'CENTRAL',
            {},
            '$MODEL NCOMPARTMENTS=23 COMPARTMENT=(CENTRAL)\n',
        ),
        (
            '$MODEL NCOM=1',
            'CENTRAL',
            {'dosing': True},
            '$MODEL NCOM=1 COMPARTMENT=(CENTRAL DEFDOSE)',
        ),
        ('$MODEL ', 'CENTRAL', {}, '$MODEL COMPARTMENT=(CENTRAL)'),
    ],
)
def test_add_compartment(parser, buf, add, kwargs, expected):
    recs = parser.parse(buf)
    rec = recs.records[0]
    rec.add_compartment('CENTRAL', **kwargs)
    assert str(rec) == expected


@pytest.mark.parametrize(
    "buf,remove,expected",
    [
        ('$MODEL COMPARTMENT=(CENTRAL)', 'CENTRAL', '$MODEL'),
        (
            '$MODEL COMPARTMENT=(CENTRAL) COMPARTMENT=PERIPH COMPARTMENT=P2',
            'PERIPH',
            '$MODEL COMPARTMENT=(CENTRAL) COMPARTMENT=P2',
        ),
        ('$MODEL COMP=(CENTRAL) COMP=PERIPH COMP=P2', 'PERIPH', '$MODEL COMP=(CENTRAL) COMP=P2'),
    ],
)
def test_remove_compartment(parser, buf, remove, expected):
    recs = parser.parse(buf)
    rec = recs.records[0]
    rec.remove_compartment(remove)
    assert str(rec) == expected


@pytest.mark.parametrize(
    "buf,results",
    [
        ('$MODEL COMP=1', [('1', [])]),
        ('$MODEL  COMP=(1) COMPARTMENT=(2)', [('1', []), ('2', [])]),
        (
            '$MODEL  COMP=(DEPOT DEFDOSE) COMPARTMENT=(CENTRAL DEFOBS)',
            [('DEPOT', ['DEFDOSE']), ('CENTRAL', ['DEFOBSERVATION'])],
        ),
        (
            '$MODEL  COMP=(DEFDOSE DEPOT) COMPARTMENT=(DEFOBS CENTRAL)',
            [('DEPOT', ['DEFDOSE']), ('CENTRAL', ['DEFOBSERVATION'])],
        ),
        (
            '$MODEL  COMP=(DEFDOSE) COMPARTMENT=(DEFOBS CENTRAL)',
            [('COMP1', ['DEFDOSE']), ('CENTRAL', ['DEFOBSERVATION'])],
        ),
    ],
)
def test_compartments(parser, buf, results):
    rec = parser.parse(buf).records[0]
    assert list(rec.compartments()) == results


@pytest.mark.parametrize(
    "buf,name,results",
    [
        ('$MODEL COMP=CENTRAL', 'CENTRAL', 1),
        ('$MODEL COMP=DEPOT COMP=CENTRAL', 'CENTRAL', 2),
        ('$MODEL COMP=DEPOT COMP=CENTRAL', 'DEPOT', 1),
        ('$MODEL COMP=DEPOT COMP=CENTRAL', 'NOTHING', None),
    ],
)
def test_get_compartment_number(parser, buf, name, results):
    rec = parser.parse(buf).records[0]
    assert rec.get_compartment_number(name) == results
