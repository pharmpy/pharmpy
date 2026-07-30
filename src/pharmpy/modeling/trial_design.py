from collections.abc import Sequence

from pharmpy.model import Arm, DataVariable, Observations, TrialDesign


def create_trial_design(idv_name: str = 'TIME') -> TrialDesign:
    """Create an empty trial design

    Parameters
    ----------
    idv_name : str
        Optional name of the independent variable. Default is TIME

    Returns
    -------
    TrialDesign
        An empty trial design

    Example
    -------
    >>> from pharmpy.modeling import create_trial_design
    >>> td = create_trial_design()
    >>> td.is_empty
    True
    >>> td
    Empty trial design object

    """

    idv = DataVariable.create(name=idv_name, type='idv', scale='ratio', count=False)
    new_td = TrialDesign.create(arms=(), independent_variable=idv)
    return new_td


def add_arm(td: TrialDesign, name: str, size: int) -> TrialDesign:
    """Add an empty arm to a trial design

    Parameters
    ----------
    td : TrialDesign
        TrialDesign to add to
    name : str
        Name of the arm
    size : int
        Size of the arm (number of subjects)

    Returns
    -------
    TrialDesign
        An updated TrialDesign

    Example
    -------
    >>> from pharmpy.modeling import create_trial_design, add_arm
    >>> td = create_trial_design()
    >>> td = add_arm(td, name="Placebo", size=100)
    >>> td
    Empty trial design object

    """

    arm = Arm.create(name=name, size=size, activities=())
    new_td = td + arm
    return new_td


def add_observations(
    td: TrialDesign, arm: str, variable: str, time_points: Sequence[float], start_time: float = 0.0
) -> TrialDesign:
    """Add observations to an arm in a trial design

    Parameters
    ----------
    td : TrialDesign
        TrialDesign to add to
    arm : str
        Name of the arm
    variable : str
        Name of the variable to observe
    time_points : Sequence[float]
        List of the time points for the observations
    start_time : float
        Set a start_time to offset the time_points. Default 0.0

    Returns
    -------
    TrialDesign
        An updated TrialDesign

    Example
    -------
    >>> from pharmpy.modeling import create_trial_design, add_arm, add_observations
    >>> td = create_trial_design()
    >>> td = add_arm(td, name="Placebo", size=100)
    >>> td = add_observations(td, arm="Placebo", variable="DV", time_points=[0.0, 1.0, 2.0, 4.0, 16.0])
    >>> td
                ╭────── Observations ──────╮
      Placebo   │ 0.0, 1.0, 2.0, 4.0, 16.0 │
                ╰─────────── DV ───────────╯
                ├──────────────────────────┤
               0.0                       16.0
    <BLANKLINE>

    Empty trial design object

    """

    datavar = DataVariable.create(name=variable, type="dv")
    obs = Observations.create(variable=datavar, start_time=start_time, time_points=time_points)
    oldarm = td[arm]
    newarm = oldarm + obs
    newtd = td.replace_arm(newarm)
    return newtd
