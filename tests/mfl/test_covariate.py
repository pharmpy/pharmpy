import itertools

import pytest

from pharmpy.mfl.features import Covariate


def test_init():
    c1 = Covariate('CL', 'VC', 'EXP', '*', False)
    assert c1.args == ('CL', 'VC', 'EXP', '*', False)


def test_create():
    parameters = ['cl', 'vc']
    covariates = ['wgt', 'age']

    for param, cov in itertools.product(parameters, covariates):
        c1 = Covariate.create(parameter=param, covariate=cov, fp='exp')
        assert c1.args == (param.upper(), cov.upper(), 'EXP', '*', False)
        c2 = Covariate.create(parameter=param, covariate=cov, fp='exp', op='+', optional=True)
        assert c2.args == (param.upper(), cov.upper(), 'EXP', '+', True)


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


def test_repr():
    c1 = Covariate.create(parameter='CL', covariate='WGT', fp='EXP')
    assert repr(c1) == 'COVARIATE(CL,WGT,EXP,*)'
    c2 = Covariate.create(parameter='CL', covariate='WGT', fp='EXP', optional=True)
    assert repr(c2) == 'COVARIATE?(CL,WGT,EXP,*)'


def test_eq():
    c1 = Covariate.create(parameter='CL', covariate='WGT', fp='EXP')
    assert c1 == c1
    c2 = Covariate.create(parameter='CL', covariate='WGT', fp='EXP')
    assert c1 == c2
    c3 = Covariate.create(parameter='CL', covariate='WGT', fp='EXP', optional=True)
    assert c3 != c1

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
    ],
)
def test_repr_many(list_of_kwargs, expected):
    features = []
    for kwargs in list_of_kwargs:
        c = Covariate.create(**kwargs)
        features.append(c)
    assert Covariate.repr_many(features) == expected
