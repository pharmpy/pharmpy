from pharmpy.model import DataVariable
from pharmpy.model.trial_design import Observations


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
