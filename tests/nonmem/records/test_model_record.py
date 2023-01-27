import pytest


@pytest.mark.parametrize(
    "buf,expected",
    [
        ('$MODEL NCOMPARTMENTS=23', 23),
        ('$MODEL NCOMP=18', 18),
        ('$MODEL', None),
        ('$MODEL NCOMPS=2', 2),
        ('$MODEL NCM=4', 4),
    ],
)
def test_ncomps(parser, buf, expected):
    recs = parser.parse(buf)
    rec = recs.records[0]
    assert rec.ncomps == expected


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
    newrec = rec.add_compartment(add, **kwargs)
    assert str(newrec) == expected


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
