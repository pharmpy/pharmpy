import re

import pytest

from pharmpy.mfl.features import (
    IIV,
    IOV,
    Absorption,
    Allometry,
    Covariance,
    Covariate,
    DirectEffect,
    EffectComp,
    Elimination,
    IndirectEffect,
    LagTime,
    Metabolite,
    Peripherals,
    Ref,
    Transits,
)
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
            [(0, True), (1, True)],
        ),
        (
            'PERIPHERALS(0..1);PERIPHERALS(0..1,MET)',
            Peripherals,
            [(0, False), (1, False), (0, True), (1, True)],
        ),
        (
            'PERIPHERALS(0..1,*)',
            Peripherals,
            [(0, False), (1, False), (0, True), (1, True)],
        ),
        (
            'PERIPHERALS(0..1,[DRUG,MET])',
            Peripherals,
            [(0, False), (1, False), (0, True), (1, True)],
        ),
        ('TRANSITS(0)', Transits, [(0, True)]),
        ('TRANSITS(1)', Transits, [(1, True)]),
        ('TRANSITS(N)', Transits, [('N', True)]),
        ('TRANSITS(n)', Transits, [('N', True)]),
        ('TRANSITS([0, 1])', Transits, [(0, True), (1, True)]),
        ('TRANSITS([0, 1]);TRANSITS(N)', Transits, [(0, True), (1, True), ('N', True)]),
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
        ('LAGTIME(*)', LagTime, [(False,), (True,)]),
        ('ELIMINATION(FO)', Elimination, [('FO',)]),
        ('ELIMINATION( *)', Elimination, [('FO',), ('ZO',), ('MM',), ('MIX-FO-MM',)]),
        ('ELIMINATION([ZO,FO])', Elimination, [('FO',), ('ZO',)]),
        ('ELIMINATION([ZO,  FO])', Elimination, [('FO',), ('ZO',)]),
        ('ELIMINATION( [   MIX-FO-MM,  FO   ]  )', Elimination, [('FO',), ('MIX-FO-MM',)]),
        ('elimination([zo, fo])', Elimination, [('FO',), ('ZO',)]),
        (
            'DIRECTEFFECT([LINEAR,SIGMOID,STEP])',
            DirectEffect,
            [('LINEAR',), ('SIGMOID',), ('STEP',)],
        ),
        ('DIRECTEFFECT(LINEAR)', DirectEffect, [('LINEAR',)]),
        (
            'DIRECTEFFECT(*)',
            DirectEffect,
            [('LINEAR',), ('EMAX',), ('SIGMOID',), ('STEP',), ('LOGLIN',)],
        ),
        ('INDIRECTEFFECT(LINEAR,PRODUCTION)', IndirectEffect, [('LINEAR', True)]),
        (
            'INDIRECTEFFECT(LINEAR,[PRODUCTION,DEGRADATION])',
            IndirectEffect,
            [('LINEAR', False), ('LINEAR', True)],
        ),
        (
            'INDIRECTEFFECT(LINEAR,*)',
            IndirectEffect,
            [('LINEAR', False), ('LINEAR', True)],
        ),
        (
            'INDIRECTEFFECT([LINEAR,EMAX,SIGMOID],PRODUCTION)',
            IndirectEffect,
            [('LINEAR', True), ('EMAX', True), ('SIGMOID', True)],
        ),
        (
            'INDIRECTEFFECT(*,PRODUCTION)',
            IndirectEffect,
            [('LINEAR', True), ('EMAX', True), ('SIGMOID', True)],
        ),
        (
            'INDIRECTEFFECT(LINEAR,PRODUCTION);INDIRECTEFFECT(LINEAR,DEGRADATION)',
            IndirectEffect,
            [('LINEAR', True), ('LINEAR', False)],
        ),
        ('EFFECTCOMP([LINEAR,SIGMOID,STEP])', EffectComp, [('LINEAR',), ('SIGMOID',), ('STEP',)]),
        (
            'EFFECTCOMP([LINEAR,SIGMOID]);EFFECTCOMP(STEP)',
            EffectComp,
            [('LINEAR',), ('SIGMOID',), ('STEP',)],
        ),
        ('EFFECTCOMP(LINEAR)', EffectComp, [('LINEAR',)]),
        (
            'EFFECTCOMP(*)',
            EffectComp,
            [('LINEAR',), ('EMAX',), ('SIGMOID',), ('STEP',), ('LOGLIN',)],
        ),
        ('METABOLITE(PSC)', Metabolite, [('PSC',)]),
        ('METABOLITE([PSC,BASIC])', Metabolite, [('PSC',), ('BASIC',)]),
        ('METABOLITE(PSC); METABOLITE(BASIC)', Metabolite, [('PSC',), ('BASIC',)]),
        ('METABOLITE(*)', Metabolite, [('PSC',), ('BASIC',)]),
        ('ALLOMETRY(WT)', Allometry, [('WT', 70)]),
        ('ALLOMETRY(WT,70)', Allometry, [('WT', 70)]),
        ('ALLOMETRY(WT, 70.0)', Allometry, [('WT', 70)]),
        (
            'COVARIATE(CL,WGT,EXP)',
            Covariate,
            [('CL', 'WGT', 'EXP', '*', False)],
        ),
        (
            'COVARIATE([CL],WGT,EXP)',
            Covariate,
            [('CL', 'WGT', 'EXP', '*', False)],
        ),
        (
            'COVARIATE(CL,[WGT],EXP)',
            Covariate,
            [('CL', 'WGT', 'EXP', '*', False)],
        ),
        (
            'COVARIATE(CL,WGT,[EXP])',
            Covariate,
            [('CL', 'WGT', 'EXP', '*', False)],
        ),
        (
            'COVARIATE(CL,[WGT,AGE],EXP)',
            Covariate,
            [('CL', 'AGE', 'EXP', '*', False), ('CL', 'WGT', 'EXP', '*', False)],
        ),
        (
            'COVARIATE([CL,VC,MAT],[WGT,AGE],[EXP,POW])',
            Covariate,
            [
                ('CL', 'AGE', 'EXP', '*', False),
                ('CL', 'AGE', 'POW', '*', False),
                ('CL', 'WGT', 'EXP', '*', False),
                ('CL', 'WGT', 'POW', '*', False),
                ('MAT', 'AGE', 'EXP', '*', False),
                ('MAT', 'AGE', 'POW', '*', False),
                ('MAT', 'WGT', 'EXP', '*', False),
                ('MAT', 'WGT', 'POW', '*', False),
                ('VC', 'AGE', 'EXP', '*', False),
                ('VC', 'AGE', 'POW', '*', False),
                ('VC', 'WGT', 'EXP', '*', False),
                ('VC', 'WGT', 'POW', '*', False),
            ],
        ),
        (
            'COVARIATE?(CL,WGT,EXP)',
            Covariate,
            [('CL', 'WGT', 'EXP', '*', True)],
        ),
        (
            'COVARIATE?(CL,[WGT,AGE],EXP)',
            Covariate,
            [('CL', 'AGE', 'EXP', '*', True), ('CL', 'WGT', 'EXP', '*', True)],
        ),
        (
            'COVARIATE?(CL,WGT,EXP);COVARIATE(VC,WGT,EXP)',
            Covariate,
            [('CL', 'WGT', 'EXP', '*', True), ('VC', 'WGT', 'EXP', '*', False)],
        ),
        ('LET(CONTINUOUS, [AGE, WT]); LET(CATEGORICAL, SEX)', Covariate, []),
        (
            'COVARIATE([CL, MAT, VC], @CONTINUOUS, EXP, *)\n'
            'COVARIATE([CL, MAT, VC], @CATEGORICAL, CAT, +)',
            Covariate,
            [
                ('CL', Ref('CONTINUOUS'), 'EXP', '*', False),
                ('MAT', Ref('CONTINUOUS'), 'EXP', '*', False),
                ('VC', Ref('CONTINUOUS'), 'EXP', '*', False),
                ('CL', Ref('CATEGORICAL'), 'CAT', '+', False),
                ('MAT', Ref('CATEGORICAL'), 'CAT', '+', False),
                ('VC', Ref('CATEGORICAL'), 'CAT', '+', False),
            ],
        ),
        (
            'LET(CONTINUOUS, [AGE, WT]); LET(CATEGORICAL, SEX)\n'
            'COVARIATE?([CL, MAT, VC], @CONTINUOUS, EXP, *)\n'
            'COVARIATE?([CL, MAT, VC], @CATEGORICAL, CAT, +)',
            Covariate,
            [
                ('CL', 'AGE', 'EXP', '*', True),
                ('CL', 'WT', 'EXP', '*', True),
                ('MAT', 'AGE', 'EXP', '*', True),
                ('MAT', 'WT', 'EXP', '*', True),
                ('VC', 'AGE', 'EXP', '*', True),
                ('VC', 'WT', 'EXP', '*', True),
                ('CL', 'SEX', 'CAT', '+', True),
                ('MAT', 'SEX', 'CAT', '+', True),
                ('VC', 'SEX', 'CAT', '+', True),
            ],
        ),
        (
            'COVARIATE?([CL, MAT, VC], @CONTINUOUS, EXP, *)\n'
            'LET(CONTINUOUS, [AGE, WT]); LET(CATEGORICAL, SEX)\n'
            'COVARIATE?([CL, MAT, VC], @CATEGORICAL, CAT, +)',
            Covariate,
            [
                ('CL', 'AGE', 'EXP', '*', True),
                ('CL', 'WT', 'EXP', '*', True),
                ('MAT', 'AGE', 'EXP', '*', True),
                ('MAT', 'WT', 'EXP', '*', True),
                ('VC', 'AGE', 'EXP', '*', True),
                ('VC', 'WT', 'EXP', '*', True),
                ('CL', 'SEX', 'CAT', '+', True),
                ('MAT', 'SEX', 'CAT', '+', True),
                ('VC', 'SEX', 'CAT', '+', True),
            ],
        ),
        (
            'COVARIATE?(@IIV, @CONTINUOUS, *)',
            Covariate,
            [
                (Ref('IIV'), Ref('CONTINUOUS'), 'CAT', '*', True),
                (Ref('IIV'), Ref('CONTINUOUS'), 'CAT2', '*', True),
                (Ref('IIV'), Ref('CONTINUOUS'), 'EXP', '*', True),
                (Ref('IIV'), Ref('CONTINUOUS'), 'LIN', '*', True),
                (Ref('IIV'), Ref('CONTINUOUS'), 'PIECE_LIN', '*', True),
                (Ref('IIV'), Ref('CONTINUOUS'), 'POW', '*', True),
            ],
        ),
        (
            'LET(CONTINUOUS,[AGE,WT])\n' 'COVARIATE?(@IIV,@CONTINUOUS,EXP,*)',
            Covariate,
            [
                (Ref('IIV'), 'AGE', 'EXP', '*', True),
                (Ref('IIV'), 'WT', 'EXP', '*', True),
            ],
        ),
        (
            'COVARIATE?(*,@CONTINUOUS,EXP,*)',
            Covariate,
            [
                (Ref('POP_PARAMS'), Ref('CONTINUOUS'), 'EXP', '*', True),
            ],
        ),
        (
            'COVARIATE?(*,*,EXP,*)',
            Covariate,
            [
                (Ref('POP_PARAMS'), Ref('COVARIATES'), 'EXP', '*', True),
            ],
        ),
        (
            'IIV(CL,EXP)',
            IIV,
            [
                ('CL', 'EXP', False),
            ],
        ),
        (
            'IIV([CL,VC],EXP)',
            IIV,
            [
                ('CL', 'EXP', False),
                ('VC', 'EXP', False),
            ],
        ),
        (
            'IIV?([CL,VC],EXP)',
            IIV,
            [
                ('CL', 'EXP', True),
                ('VC', 'EXP', True),
            ],
        ),
        (
            'IIV?([CL,VC],EXP);IIV(MAT,ADD)',
            IIV,
            [
                ('CL', 'EXP', True),
                ('VC', 'EXP', True),
                ('MAT', 'ADD', False),
            ],
        ),
        (
            'IIV?([CL,VC],[EXP,ADD])',
            IIV,
            [
                ('CL', 'ADD', True),
                ('CL', 'EXP', True),
                ('VC', 'ADD', True),
                ('VC', 'EXP', True),
            ],
        ),
        (
            'IIV(CL,*)',
            IIV,
            [
                ('CL', 'ADD', False),
                ('CL', 'EXP', False),
                ('CL', 'LOG', False),
                ('CL', 'PROP', False),
                ('CL', 'RE_LOG', False),
            ],
        ),
        (
            'IIV?(@PK,[EXP,ADD])',
            IIV,
            [
                (Ref('PK'), 'ADD', True),
                (Ref('PK'), 'EXP', True),
            ],
        ),
        (
            'IIV?(*,[EXP,ADD])',
            IIV,
            [
                (Ref('POP_PARAMS'), 'ADD', True),
                (Ref('POP_PARAMS'), 'EXP', True),
            ],
        ),
        (
            'IOV(CL,EXP)',
            IOV,
            [
                ('CL', 'EXP', False),
            ],
        ),
        (
            'IOV([CL,VC],EXP)',
            IOV,
            [
                ('CL', 'EXP', False),
                ('VC', 'EXP', False),
            ],
        ),
        (
            'IOV?([CL,VC],EXP)',
            IOV,
            [
                ('CL', 'EXP', True),
                ('VC', 'EXP', True),
            ],
        ),
        (
            'IOV?([CL,VC],EXP);IOV(MAT,ADD)',
            IOV,
            [
                ('CL', 'EXP', True),
                ('VC', 'EXP', True),
                ('MAT', 'ADD', False),
            ],
        ),
        (
            'IOV?([CL,VC],[EXP,ADD])',
            IOV,
            [
                ('CL', 'ADD', True),
                ('CL', 'EXP', True),
                ('VC', 'ADD', True),
                ('VC', 'EXP', True),
            ],
        ),
        (
            'IOV(CL,*)',
            IOV,
            [
                ('CL', 'ADD', False),
                ('CL', 'EXP', False),
                ('CL', 'LOG', False),
                ('CL', 'PROP', False),
                ('CL', 'RE_LOG', False),
            ],
        ),
        (
            'IOV?(@PK,[EXP,ADD])',
            IOV,
            [
                (Ref('PK'), 'ADD', True),
                (Ref('PK'), 'EXP', True),
            ],
        ),
        (
            'IOV?(*,[EXP,ADD])',
            IOV,
            [
                (Ref('POP_PARAMS'), 'ADD', True),
                (Ref('POP_PARAMS'), 'EXP', True),
            ],
        ),
        (
            'COVARIANCE(IIV,[CL,VC])',
            Covariance,
            [
                ('IIV', ['CL', 'VC'], False),
            ],
        ),
        (
            'COVARIANCE(IIV,[CL,MAT,VC])',
            Covariance,
            [
                ('IIV', ['CL', 'MAT'], False),
                ('IIV', ['CL', 'VC'], False),
                ('IIV', ['VC', 'MAT'], False),
            ],
        ),
        (
            'COVARIANCE?(IIV,[CL,MAT,VC])',
            Covariance,
            [
                ('IIV', ['CL', 'MAT'], True),
                ('IIV', ['CL', 'VC'], True),
                ('IIV', ['VC', 'MAT'], True),
            ],
        ),
        (
            'COVARIANCE?(IIV,[CL,MAT,MDT,VC])',
            Covariance,
            [
                ('IIV', ['CL', 'MAT'], True),
                ('IIV', ['CL', 'MDT'], True),
                ('IIV', ['CL', 'VC'], True),
                ('IIV', ['MAT', 'MDT'], True),
                ('IIV', ['MAT', 'VC'], True),
                ('IIV', ['MDT', 'VC'], True),
            ],
        ),
        (
            'COVARIANCE(IOV,[CL,MAT,VC])',
            Covariance,
            [
                ('IOV', ['CL', 'MAT'], False),
                ('IOV', ['CL', 'VC'], False),
                ('IOV', ['VC', 'MAT'], False),
            ],
        ),
        (
            'COVARIANCE(IOV,[CL,MAT,VC]);COVARIANCE(IIV,[CL,VC])',
            Covariance,
            [
                ('IOV', ['CL', 'MAT'], False),
                ('IOV', ['CL', 'VC'], False),
                ('IOV', ['VC', 'MAT'], False),
                ('IIV', ['CL', 'VC'], False),
            ],
        ),
        (
            'COVARIANCE([IIV,IOV],[CL,MAT,VC])',
            Covariance,
            [
                ('IIV', ['CL', 'MAT'], False),
                ('IIV', ['CL', 'VC'], False),
                ('IIV', ['VC', 'MAT'], False),
                ('IOV', ['CL', 'MAT'], False),
                ('IOV', ['CL', 'VC'], False),
                ('IOV', ['VC', 'MAT'], False),
            ],
        ),
        (
            'COVARIANCE(*,[CL,MAT,VC])',
            Covariance,
            [
                ('IIV', ['CL', 'MAT'], False),
                ('IIV', ['CL', 'VC'], False),
                ('IIV', ['VC', 'MAT'], False),
                ('IOV', ['CL', 'MAT'], False),
                ('IOV', ['CL', 'VC'], False),
                ('IOV', ['VC', 'MAT'], False),
            ],
        ),
        (
            'COVARIANCE?(IIV,@IIV)',
            Covariance,
            [
                ('IIV', Ref('IIV'), True),
            ],
        ),
        (
            'LET(IIV,[CL,VC,MAT]);COVARIANCE?(IIV,@IIV)',
            Covariance,
            [
                ('IIV', ['CL', 'MAT'], True),
                ('IIV', ['CL', 'VC'], True),
                ('IIV', ['VC', 'MAT'], True),
            ],
        ),
    ),
    ids=repr,
)
def test_parse_one_type(source, feature_type, options):
    features = parse(source)
    assert features == [feature_type.create(*opt) for opt in options]


@pytest.mark.parametrize(
    'source, no_of_features',
    (
        (
            'ELIMINATION(FO);ABSORPTION(ZO)',
            2,
        ),
        (
            'ELIMINATION(FO) ;   ABSORPTION(ZO) ',
            2,
        ),
        (
            'ABSORPTION(FO);ELIMINATION(FO);ABSORPTION(ZO)',
            3,
        ),
        (
            'ABSORPTION(FO);ELIMINATION(FO);ABSORPTION(FO)',
            3,
        ),
        (
            'ABSORPTION([FO,ZO]);ELIMINATION(FO)',
            3,
        ),
        (
            'ABSORPTION(FO);ELIMINATION(FO);PERIPHERALS(0);TRANSITS(0,DEPOT);LAGTIME(OFF)',
            5,
        ),
        (
            'DIRECTEFFECT(LINEAR);INDIRECTEFFECT(LINEAR,PRODUCTION);EFFECTCOMP(LINEAR);METABOLITE(PSC)',
            4,
        ),
        (
            'IIV([CL,MAT,VC],EXP);IOV([CL,MAT,VC],EXP)',
            6,
        ),
    ),
    ids=repr,
)
def test_parse_multiple_types(source, no_of_features):
    features = parse(source)
    assert len(features) == no_of_features
    types = {type(feature) for feature in features}
    assert len(types) > 1


@pytest.mark.parametrize(
    'source, error_type, match',
    (
        ('ABSORPTION(X)', ValueError, "Invalid values in ABSORPTION: ['X']"),
        ('ABSORPTION([X,Y])', ValueError, "Invalid values in ABSORPTION: ['X', 'Y']"),
        ('PERIPHERALS(X)', ValueError, "Error in parsing: token X at line 1 col 13"),
        ('PERIPHERALS(1,X)', ValueError, "Invalid values in PERIPHERALS: ['X']"),
        ('TRANSITS(X)', ValueError, "Error in parsing: token X at line 1 col 10"),
        ('TRANSITS(1,X)', ValueError, "Invalid values in TRANSITS: ['X']"),
        ('LAGTIME(X)', ValueError, "Invalid values in LAGTIME: ['X']"),
        ('ELIMINATION(X)', ValueError, "Invalid values in ELIMINATION: ['X']"),
        ('DIRECTEFFECT(X)', ValueError, "Invalid values in DIRECTEFFECT: ['X']"),
        ('INDIRECTEFFECT(X,PRODUCTION)', ValueError, "Invalid values in INDIRECTEFFECT: ['X']"),
        ('INDIRECTEFFECT(EMAX,X)', ValueError, "Invalid values in INDIRECTEFFECT: ['X']"),
        ('EFFECTCOMP(X)', ValueError, "Invalid values in EFFECTCOMP: ['X']"),
        ('METABOLITE(X)', ValueError, "Invalid values in METABOLITE: ['X']"),
        ('ALLOMETRY(WT,X)', ValueError, "Error in parsing: token X at line 1 col 14"),
        ('COVARIATE(CL,WT,X,*)', ValueError, "Invalid values in COVARIATE: ['X']"),
        ('IIV(CL,X)', ValueError, "Invalid values in IIV: ['X']"),
        ('IOV(CL,X)', ValueError, "Invalid values in IOV: ['X']"),
        ('COVARIANCE(X,[CL,VC])', ValueError, "Invalid values in COVARIANCE: ['X']"),
        ('X', ValueError, "Error in parsing: token X at line 1 col 1"),
    ),
    ids=repr,
)
def test_parse_raises(source, error_type, match):
    with pytest.raises(error_type, match=re.escape(match)):
        parse(source)
