from pharmpy.basic import Expr
from pharmpy.model import Bolus, DataVariable
from pharmpy.model.trial_design import Administration, Observations


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
