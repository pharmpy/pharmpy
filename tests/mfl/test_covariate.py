import itertools

import pytest

from pharmpy.mfl.features import Covariate
from pharmpy.mfl.features.symbols import Ref
from pharmpy.mfl.model_features import ModelFeatures


def test_init():
    c1 = Covariate('CL', 'VC', 'EXP', '*', False)
    assert c1.args == ('CL', 'VC', 'EXP', '*', False)
    assert c1.args == (c1.parameter, c1.covariate, c1.fp, c1.op, c1.optional)

    c2 = Covariate(Ref('IIV'), 'VC', 'EXP', '*', False)
    assert c2.args == (Ref('IIV'), 'VC', 'EXP', '*', False)
    assert c2.args == (c2.parameter, c2.covariate, c2.fp, c2.op, c2.optional)


def test_create():
    parameters = ['cl', 'vc']
    covariates = ['wgt', 'age']

    for param, cov in itertools.product(parameters, covariates):
        c1 = Covariate.create(parameter=param, covariate=cov, fp='exp')
        assert c1.args == (param.upper(), cov.upper(), 'EXP', '*', False)
        assert c1.args == (c1.parameter, c1.covariate, c1.fp, c1.op, c1.optional)

        c2 = Covariate.create(parameter=param, covariate=cov, fp='exp', op='+', optional=True)
        assert c2.args == (param.upper(), cov.upper(), 'EXP', '+', True)
        assert c2.args == (c2.parameter, c2.covariate, c2.fp, c2.op, c2.optional)

    c3 = Covariate(Ref('IIV'), Ref('CONTINUOUS'), 'EXP', '*', False)
    assert c3.args == (Ref('IIV'), Ref('CONTINUOUS'), 'EXP', '*', False)
    assert c3.args == (c3.parameter, c3.covariate, c3.fp, c3.op, c3.optional)


@pytest.mark.parametrize(
    'new_opt, expected_error',
    [
        ({'parameter': 1}, TypeError),
        ({'covariate': 1}, TypeError),
        ({'fp': 1}, TypeError),
        ({'fp': 'x'}, ValueError),
        ({'op': 1}, TypeError),
        ({'op': 'x'}, ValueError),
        ({'optional': 1}, TypeError),
    ],
)
def test_create_raises(new_opt, expected_error):
    args_dict = {'parameter': 'CL', 'covariate': 'WGT', 'fp': 'EXP', 'op': '*', 'optional': False}
    opt_name = list(new_opt.keys())[0]
    args_dict[opt_name] = new_opt[opt_name]
    with pytest.raises(expected_error):
        Covariate.create(**args_dict)


def test_replace():
    c1 = Covariate.create(parameter='CL', covariate='WGT', fp='EXP')
    assert c1.args == ('CL', 'WGT', 'EXP', '*', False)
    c2 = c1.replace(fp='lin')
    assert c2.args == ('CL', 'WGT', 'LIN', '*', False)

    with pytest.raises(TypeError):
        c1.replace(parameter=1)


def test_expand():
    expand_to = {Ref('IIV'): ['CL', 'VC', 'MAT'], Ref('CONTINUOUS'): ['WGT', 'AGE']}

    c1 = Covariate.create(parameter='CL', covariate='WGT', fp='EXP')
    c1_expanded = c1.expand(expand_to)
    assert c1_expanded == (c1,)

    c2 = Covariate.create(parameter=Ref('IIV'), covariate='WGT', fp='EXP')
    c2_expanded = c2.expand(expand_to)
    assert len(c2_expanded) == 3
    assert c2_expanded[0].parameter == 'CL'

    c3 = Covariate.create(parameter='CL', covariate=Ref('CONTINUOUS'), fp='EXP')
    c3_expanded = c3.expand(expand_to)
    assert len(c3_expanded) == 2
    assert c3_expanded[0].covariate == 'AGE'

    c4 = Covariate.create(parameter=Ref('IIV'), covariate=Ref('CONTINUOUS'), fp='EXP')
    c4_expanded = c4.expand(expand_to)
    assert len(c4_expanded) == 6
    assert c4_expanded[0].parameter == 'CL'
    assert c4_expanded[0].covariate == 'AGE'

    assert c2.expand({Ref('IIV'): tuple()}) == tuple()
    assert c4.expand({Ref('IIV'): tuple(), Ref('CONTINUOUS'): ['WGT', 'AGE']}) == tuple()

    with pytest.raises(ValueError):
        c4.expand({Ref('IIV'): ['CL', 'VC', 'MAT']})

    with pytest.raises(ValueError):
        c4.expand({Ref('CONTINUOUS'): ['WGT', 'AGE']})


def test_repr():
    c1 = Covariate.create(parameter='CL', covariate='WGT', fp='EXP')
    assert repr(c1) == 'COVARIATE(CL,WGT,EXP,*)'
    c2 = Covariate.create(parameter='CL', covariate='WGT', fp='EXP', optional=True)
    assert repr(c2) == 'COVARIATE?(CL,WGT,EXP,*)'
    c3 = Covariate.create(parameter=Ref('IIV'), covariate='WGT', fp='EXP', optional=True)
    assert repr(c3) == 'COVARIATE?(@IIV,WGT,EXP,*)'


def test_eq():
    c1 = Covariate.create(parameter='CL', covariate='WGT', fp='EXP')
    assert c1 == c1
    c2 = Covariate.create(parameter='CL', covariate='WGT', fp='EXP')
    assert c1 == c2
    c3 = Covariate.create(parameter='CL', covariate='WGT', fp='EXP', optional=True)
    assert c3 != c1
    c4 = Covariate.create(parameter=Ref('IIV'), covariate='WGT', fp='EXP', optional=True)
    assert c4 != c1

    assert c1 != 1


def test_lt():
    c1 = Covariate.create(parameter='CL', covariate='WGT', fp='EXP')
    assert not c1 < c1
    c2 = Covariate.create(parameter='VC', covariate='WGT', fp='EXP')
    assert c1 < c2
    c3 = Covariate.create(parameter='VC', covariate='AGE', fp='EXP')
    assert c3 < c2
    c4 = Covariate.create(parameter='VC', covariate='AGE', fp='LIN')
    assert c3 < c4
    c5 = Covariate.create(parameter='VC', covariate='AGE', fp='LIN', op='+')
    assert c4 < c5
    c6 = Covariate.create(parameter='VC', covariate='AGE', fp='LIN', op='+', optional=True)
    assert c5 < c6
    c7 = Covariate.create(parameter=Ref('IIV'), covariate='WGT', fp='EXP', optional=True)
    assert c7 < c6

    with pytest.raises(TypeError):
        c1 < 1


@pytest.mark.parametrize(
    'list_of_kwargs, expected',
    [
        (
            [
                {'parameter': 'CL', 'covariate': 'WGT', 'fp': 'exp'},
                {'parameter': 'VC', 'covariate': 'WGT', 'fp': 'exp'},
                {'parameter': 'CL', 'covariate': 'AGE', 'fp': 'exp'},
                {'parameter': 'VC', 'covariate': 'AGE', 'fp': 'exp'},
            ],
            'COVARIATE([CL,VC],[AGE,WGT],EXP,*)',
        ),
        (
            [
                {'parameter': 'CL', 'covariate': 'WGT', 'fp': 'exp'},
                {'parameter': 'CL', 'covariate': 'AGE', 'fp': 'exp', 'optional': True},
                {'parameter': 'VC', 'covariate': 'WGT', 'fp': 'exp', 'optional': True},
                {'parameter': 'VC', 'covariate': 'AGE', 'fp': 'exp', 'optional': True},
                {'parameter': 'MAT', 'covariate': 'SEX', 'fp': 'cat', 'optional': True},
            ],
            'COVARIATE(CL,WGT,EXP,*);COVARIATE?(CL,AGE,EXP,*);COVARIATE?(VC,[AGE,WGT],EXP,*);COVARIATE?(MAT,SEX,CAT,*)',
        ),
        (
            [
                {'parameter': 'CL', 'covariate': 'WGT', 'fp': 'exp'},
                {'parameter': 'VC', 'covariate': 'WGT', 'fp': 'exp'},
                {'parameter': 'MAT', 'covariate': 'AGE', 'fp': 'exp'},
                {'parameter': 'VC', 'covariate': 'AGE', 'fp': 'exp'},
            ],
            'COVARIATE(CL,WGT,EXP,*);COVARIATE(MAT,AGE,EXP,*);COVARIATE(VC,[AGE,WGT],EXP,*)',
        ),
        (
            [
                {'parameter': 'CL', 'covariate': 'WGT', 'fp': 'exp'},
                {'parameter': 'VC', 'covariate': 'WGT', 'fp': 'exp'},
                {'parameter': 'CL', 'covariate': 'AGE', 'fp': 'exp'},
                {'parameter': 'VC', 'covariate': 'AGE', 'fp': 'exp', 'optional': True},
            ],
            'COVARIATE(CL,[AGE,WGT],EXP,*);COVARIATE(VC,WGT,EXP,*);COVARIATE?(VC,AGE,EXP,*)',
        ),
        (
            [
                {'parameter': 'CL', 'covariate': 'WGT', 'fp': 'exp'},
                {'parameter': 'VC', 'covariate': 'WGT', 'fp': 'exp'},
                {'parameter': 'CL', 'covariate': 'AGE', 'fp': 'exp'},
                {'parameter': 'VC', 'covariate': 'AGE', 'fp': 'exp'},
                {'parameter': Ref('IIV'), 'covariate': 'WGT', 'fp': 'exp'},
            ],
            'COVARIATE(@IIV,WGT,EXP,*);COVARIATE([CL,VC],[AGE,WGT],EXP,*)',
        ),
        (
            [
                {'parameter': 'CL', 'covariate': 'WGT', 'fp': 'exp'},
                {'parameter': 'VC', 'covariate': 'WGT', 'fp': 'exp'},
                {'parameter': Ref('IIV'), 'covariate': 'WGT', 'fp': 'exp'},
            ],
            'COVARIATE([@IIV,CL,VC],WGT,EXP,*)',
        ),
        (
            [
                {'parameter': 'CL', 'covariate': 'WGT', 'fp': 'EXP', 'op': '+'},
                {'parameter': 'VC', 'covariate': 'WGT', 'fp': 'EXP', 'op': '+'},
                {'parameter': 'MAT', 'covariate': 'WGT', 'fp': 'EXP', 'op': '+'},
                {'parameter': 'CL', 'covariate': 'WGT', 'fp': 'LIN', 'op': '+'},
                {'parameter': 'VC', 'covariate': 'WGT', 'fp': 'LIN', 'op': '+'},
                {'parameter': 'MAT', 'covariate': 'WGT', 'fp': 'LIN', 'op': '+'},
            ],
            'COVARIATE([CL,MAT,VC],WGT,[EXP,LIN],+)',
        ),
    ],
)
def test_repr_many(list_of_kwargs, expected):
    features = []
    for kwargs in list_of_kwargs:
        c = Covariate.create(**kwargs)
        features.append(c)
    mfl1 = ModelFeatures.create(features)
    assert Covariate.repr_many(mfl1) == expected
    mfl2 = ModelFeatures.create([features[0]])
    assert Covariate.repr_many(mfl2) == repr(features[0])
