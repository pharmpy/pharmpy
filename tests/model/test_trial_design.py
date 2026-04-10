import pytest

from pharmpy.basic import Expr
from pharmpy.model import Bolus, DataVariable
from pharmpy.model.trial_design import Administration, Arm, Observations


def test_observations():
    dv = DataVariable("CONC", "dv", "ratio")
    obs1 = Observations(dv, 0.0, (0.0, 1.0, 2.0))
    obs2 = Observations.create(dv, 0.0, [0.0, 1.0, 2.0])
    assert obs1 == obs2
    assert obs1 == obs1
    assert obs1 != 1.0
    assert hash(obs1) == hash(obs2)
    assert repr(obs1) == 'Observations(CONC, 0.0, (0.0, 1.0, 2.0))'

    dv2 = DataVariable("DV", "dv", "ratio")
    assert obs1.replace(variable=dv2).variable == dv2
    assert obs1.replace(start_time=1.0).start_time == 1.0
    assert obs1.replace(time_points=[2.0]).time_points == (2.0,)

    d = {
        'class': 'Observations',
        'variable': {
            'count': False,
            'name': 'CONC',
            'properties': {},
            'scale': 'ratio',
            'type': 'dv',
        },
        'start_time': 0.0,
        'time_points': (0.0, 1.0, 2.0),
    }
    assert obs1.to_dict() == d
    assert Observations.from_dict(d) == obs1


def test_administration():
    amt = DataVariable("AMT", "dose", "ratio")
    dose = Bolus.create(100)
    adm1 = Administration(amt, dose, 0.0, (0.0, 1.0, 2.0))
    adm2 = Administration.create(amt, dose, 0.0, [0.0, 1.0, 2.0])
    assert adm1 == adm2
    assert adm1 == adm1
    assert adm1 != 1.0
    assert hash(adm1) == hash(adm2)
    assert repr(adm1) == 'Administration(AMT, Bolus(100, admid=1), 0.0, (0.0, 1.0, 2.0))'

    amt2 = DataVariable("DOSE", "dose", "ratio")
    dose2 = Bolus.create(10 * Expr.symbol("WGT"))
    assert adm1.replace(variable=amt2).variable == amt2
    assert adm1.replace(start_time=1.0).start_time == 1.0
    assert adm1.replace(time_points=[2.0]).time_points == (2.0,)
    assert adm1.replace(dose=dose2).dose == dose2

    d = {
        'class': 'Administration',
        'variable': {
            'count': False,
            'name': 'AMT',
            'properties': {},
            'scale': 'ratio',
            'type': 'dose',
        },
        'dose': {'class': 'Bolus', 'amount': 'Integer(100)', 'admid': 1},
        'start_time': 0.0,
        'time_points': (0.0, 1.0, 2.0),
    }
    assert adm1.to_dict() == d
    assert Administration.from_dict(d) == adm1


def test_arm():
    dv = DataVariable("CONC", "dv", "ratio")
    obs1 = Observations(dv, 0.0, (0.0, 1.0, 2.0))
    arm1 = Arm(size=50, activities=(obs1,))
    arm2 = Arm.create(size=50, activities=[obs1])
    assert arm1 == arm1
    assert arm1 == arm2
    assert hash(arm1) == hash(arm2)
    assert arm1 != 23

    obs2 = Observations(dv, 0.0, (1.0, 3.0))
    assert arm1.replace(size=75).size == 75
    assert arm1.replace(activities=(obs2,)).activities == (obs2,)

    amt = DataVariable("AMT", "dose", "ratio")
    dose = Bolus.create(100)
    adm1 = Administration(amt, dose, 0.0, (0.0, 1.0, 2.0))
    arm4 = Arm.create(size=50, activities=[obs1, adm1])

    d = {
        'size': 50,
        'activities': (
            {
                'class': 'Observations',
                'variable': {
                    'count': False,
                    'name': 'CONC',
                    'properties': {},
                    'scale': 'ratio',
                    'type': 'dv',
                },
                'start_time': 0.0,
                'time_points': (0.0, 1.0, 2.0),
            },
            {
                'class': 'Administration',
                'variable': {
                    'count': False,
                    'name': 'AMT',
                    'properties': {},
                    'scale': 'ratio',
                    'type': 'dose',
                },
                'dose': {'class': 'Bolus', 'amount': 'Integer(100)', 'admid': 1},
                'start_time': 0.0,
                'time_points': (0.0, 1.0, 2.0),
            },
        ),
    }

    assert arm4.to_dict() == d
    assert Arm.from_dict(d) == arm4

    assert len(arm1) == 1

    assert arm1[0] == obs1

    amt = DataVariable("AMT", "dose", "ratio")
    dose = Bolus.create(100)
    adm1 = Administration(amt, dose, 0.0, (0.0, 1.0, 2.0))

    arm3 = Arm.create(size=75, activities=[obs1, obs2, adm1])
    assert len(arm3) == 3
    assert arm3[2] == adm1
    assert arm3[0:2] == Arm.create(size=75, activities=[obs1, obs2])
    assert arm3[1:] == Arm.create(size=75, activities=[obs2, adm1])

    assert obs1 + Arm.create(size=75, activities=[obs2, adm1]) == arm3
    assert [obs1] + Arm.create(size=75, activities=[obs2, adm1]) == arm3
    assert Arm.create(size=75, activities=[obs1, obs2]) + adm1 == arm3
    assert Arm.create(size=75, activities=[obs1, obs2]) + [adm1] == arm3

    with pytest.raises(TypeError):
        23 + arm3

    with pytest.raises(TypeError):
        arm3 + "a"

    with pytest.raises(TypeError):
        arm3 + 1

    assert repr(arm1) == "Arm(size=50, (Observations(CONC, 0.0, (0.0, 1.0, 2.0)),))"
