from pharmpy.modeling import add_arm, add_observations, create_trial_design


def test_create_trial_design():
    td = create_trial_design()
    assert td.independent_variable.name == 'TIME'


def test_add_arm():
    td = create_trial_design()
    td = add_arm(td, name="Drug", size=20)
    assert len(td.arms) == 1


def test_add_observations():
    td = create_trial_design()
    td = add_arm(td, name="Drug", size=20)
    td = add_observations(td, arm="Drug", variable="DV", time_points=[0.0, 1.0, 2.0, 4.0, 8.0])
    assert td[0].name == "Drug"
    assert td[0][0].time_points[-1] == 8.0
