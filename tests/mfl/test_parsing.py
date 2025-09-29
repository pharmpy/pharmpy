import pytest

from pharmpy.mfl.features import Absorption, LagTime, Peripherals, Transits
from pharmpy.mfl.parsing import parse


@pytest.mark.parametrize(
    'source, feature_type, options',
    (
        (
            'ABSORPTION(FO)',
            Absorption,
            [('FO',)],
        ),
        (
            'ABSORPTION([FO,ZO])',
            Absorption,
            [('FO',), ('ZO',)],
        ),
        (
            'ABSORPTION([ZO,  FO])',
            Absorption,
            [('FO',), ('ZO',)],
        ),
        (
            'ABSORPTION( [   SEQ-ZO-FO,  FO   ]  )',
            Absorption,
            [('FO',), ('SEQ-ZO-FO',)],
        ),
        (
            'ABSORPTION([zo, fo])',
            Absorption,
            [('FO',), ('ZO',)],
        ),
        (
            'ABSORPTION(FO);ABSORPTION(ZO)',
            Absorption,
            [('FO',), ('ZO',)],
        ),
        (
            'ABSORPTION(FO)\nABSORPTION([FO, SEQ-ZO-FO])',
            Absorption,
            [('FO',), ('FO',), ('SEQ-ZO-FO',)],
        ),
        (
            'ABSORPTION(*)',
            Absorption,
            [('FO',), ('ZO',), ('SEQ-ZO-FO',), ('WEIBULL',)],
        ),
        ('PERIPHERALS(0)', Peripherals, [(0,)]),
        (
            'PERIPHERALS([0, 1])',
            Peripherals,
            [(0,), (1,)],
        ),
        (
            'PERIPHERALS([0, 2, 4])',
            Peripherals,
            [(0,), (2,), (4,)],
        ),
        (
            'PERIPHERALS(0..1)',
            Peripherals,
            [(0,), (1,)],
        ),
        (
            'PERIPHERALS(1..4)',
            Peripherals,
            [(1,), (2,), (3,), (4,)],
        ),
        (
            'PERIPHERALS(1..4); PERIPHERALS(5)',
            Peripherals,
            [(1,), (2,), (3,), (4,), (5,)],
        ),
        (
            'PERIPHERALS(0..1,MET)',
            Peripherals,
            [(0, 'MET'), (1, 'MET')],
        ),
        (
            'PERIPHERALS(0..1);PERIPHERALS(0..1,MET)',
            Peripherals,
            [(0, 'DRUG'), (1, 'DRUG'), (0, 'MET'), (1, 'MET')],
        ),
        (
            'PERIPHERALS(0..1,*)',
            Peripherals,
            [(0, 'DRUG'), (1, 'DRUG'), (0, 'MET'), (1, 'MET')],
        ),
        (
            'PERIPHERALS(0..1,[DRUG,MET])',
            Peripherals,
            [(0, 'DRUG'), (1, 'DRUG'), (0, 'MET'), (1, 'MET')],
        ),
        ('TRANSITS(0)', Transits, [(0, True)]),
        ('TRANSITS(1)', Transits, [(1, True)]),
        ('TRANSITS([0, 1])', Transits, [(0, True), (1, True)]),
        ('TRANSITS([0, 2, 4])', Transits, [(0, True), (2, True), (4, True)]),
        ('TRANSITS(0..1)', Transits, [(0, True), (1, True)]),
        ('TRANSITS(1..4)', Transits, [(1, True), (2, True), (3, True), (4, True)]),
        (
            'TRANSITS(1..4); TRANSITS(5)',
            Transits,
            [(1, True), (2, True), (3, True), (4, True), (5, True)],
        ),
        ('TRANSITS(1, *)', Transits, [(1, True), (1, False)]),
        ('TRANSITS(1, DEPOT)', Transits, [(1, True)]),
        ('TRANSITS(1, NODEPOT)', Transits, [(1, False)]),
        ('TRANSITS(1..4, DEPOT)', Transits, [(1, True), (2, True), (3, True), (4, True)]),
        ('TRANSITS(1..4, NODEPOT)', Transits, [(1, False), (2, False), (3, False), (4, False)]),
        (
            'TRANSITS(1..4, *)',
            Transits,
            [
                (1, True),
                (2, True),
                (3, True),
                (4, True),
                (1, False),
                (2, False),
                (3, False),
                (4, False),
            ],
        ),
        ('LAGTIME(ON)', LagTime, [(True,)]),
        ('LAGTIME ( ON )', LagTime, [(True,)]),
        ('LAGTIME(OFF)', LagTime, [(False,)]),
        ('LAGTIME([ON, OFF])', LagTime, [(False,), (True,)]),
    ),
    ids=repr,
)
def test_parse_one_type(source, feature_type, options):
    features = parse(source)
    assert features == [feature_type.create(*opt) for opt in options]
