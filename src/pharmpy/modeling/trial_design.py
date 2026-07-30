from pharmpy.model import Arm, DataVariable, TrialDesign


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


def add_arm(td: TrialDesign, size: int) -> TrialDesign:
    """Add an empty arm to a trial design

    Parameters
    ----------
    td : TrialDesign
        TrialDesign to add to
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
    >>> td = add_arm(td, size=100)
    >>> td
    Empty trial design object

    """

    arm = Arm.create(size=size, activities=())
    new_td = td + arm
    return new_td
