from pharmpy.model.trial_design import Observations


def test_observations():
    obs1 = Observations("CONC", 0.0, (0.0, 1.0, 2.0))
    obs2 = Observations.create("CONC", 0.0, [0.0, 1.0, 2.0])
    assert obs1 == obs2
    assert obs1 == obs1
    assert obs1 != 1.0
    assert hash(obs1) == hash(obs2)
    assert repr(obs1) == 'Observations(CONC, 0.0, (0.0, 1.0, 2.0))'

    assert obs1.replace(variable="DV").variable == "DV"
    assert obs1.replace(start_time=1.0).start_time == 1.0
    assert obs1.replace(time_points=[2.0]).time_points == (2.0,)

    d = {
        'class': 'Observations',
        'variable': 'CONC',
        'start_time': 0.0,
        'time_points': (0.0, 1.0, 2.0),
    }
    assert obs1.to_dict() == d
    assert Observations.from_dict(d) == obs1
