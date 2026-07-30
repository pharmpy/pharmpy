from pharmpy.modeling import add_arm, create_trial_design


def test_create_trial_design():
    td = create_trial_design()
    assert td.independent_variable.name == 'TIME'


def test_add_arm():
    td = create_trial_design()
    td = add_arm(td, size=20)
    assert len(td.arms) == 1
