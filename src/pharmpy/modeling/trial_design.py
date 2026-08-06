import math
from collections.abc import Sequence

from pharmpy.basic import Expr
from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.model import (
    Administration,
    Arm,
    Bolus,
    DataVariable,
    Model,
    Observations,
    TrialDesign,
    get_and_check_dataset,
)

from .data import get_doses, get_ids, get_observations


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

    """

    datavar = DataVariable.create(name=variable, type="dv")
    obs = Observations.create(variable=datavar, start_time=start_time, time_points=time_points)
    oldarm = td[arm]
    newarm = oldarm + obs
    newtd = td.replace_arm(newarm)
    return newtd


def add_administration(
    td: TrialDesign,
    arm: str,
    variable: str,
    amount: float,
    time_points: Sequence[float],
    start_time: float = 0.0,
) -> TrialDesign:
    """Add observations to an arm in a trial design

    Parameters
    ----------
    td : TrialDesign
        TrialDesign to add to
    arm : str
        Name of the arm
    variable : str
        Name of the dose variable (e.g. AMT)
    amount : float
        The dose amount
    time_points : Sequence[float]
        List of the time points for the administration
    start_time : float
        Set a start_time to offset the time_points. Default 0.0

    Returns
    -------
    TrialDesign
        An updated TrialDesign

    Example
    -------
    >>> from pharmpy.modeling import create_trial_design, add_arm, add_observations
    >>> from pharmpy.modeling import add_administration
    >>> td = create_trial_design()
    >>> td = add_arm(td, name="Drug", size=100)
    >>> td = add_observations(td, arm="Drug", variable="DV", time_points=[0.0, 1.0, 2.0, 4.0, 16.0])
    >>> td = add_administration(td, arm="Drug", variable="AMT", amount=10.0, time_points=[0.0, 8.0])
    >>> td
            ╭────── Observations ──────╮
      Drug  │ 0.0, 1.0, 2.0, 4.0, 16.0 │
            ╰─────────── DV ───────────╯
            ╭───── Administration ─────╮
            │ 0.0, 8.0                 │
            ╰─────── 10.0 Bolus ───────╯
            ├──────────────────────────┤
           0.0                       16.0
    <BLANKLINE>

    """

    datavar = DataVariable.create(name=variable, type="dose")
    dose = Bolus(amount=Expr(amount), admid=1)
    adm = Administration.create(
        variable=datavar, dose=dose, start_time=start_time, time_points=time_points
    )
    oldarm = td[arm]
    newarm = oldarm + adm
    newtd = td.replace_arm(newarm)
    return newtd


def create_dataset_from_design(td: TrialDesign) -> pd.DataFrame:
    """Create a dataset matching a design

    Parameters
    ----------
    td : TrialDesign
        TrialDesign to add to

    Returns
    -------
    pd.DataFrame
        A dataset

    Example
    -------
    >>> from pharmpy.modeling import create_trial_design, add_arm, add_observations
    >>> from pharmpy.modeling import add_administration, create_dataset_from_design
    >>> td = create_trial_design()
    >>> td = add_arm(td, name="Drug", size=100)
    >>> td = add_observations(td, arm="Drug", variable="DV", time_points=[0.0, 1.0, 2.0, 4.0, 16.0])
    >>> td = add_administration(td, arm="Drug", variable="AMT", amount=10.0, time_points=[0.0, 8.0])
    >>> df = create_dataset_from_design(td)
    >>> df
          ID  TIME  EVID   AMT   DV
    0      1   0.0     1  10.0  0.0
    1      1   0.0     0   0.0  0.0
    2      1   1.0     0   0.0  0.0
    3      1   2.0     0   0.0  0.0
    4      1   4.0     0   0.0  0.0
    ..   ...   ...   ...   ...  ...
    695  100   1.0     0   0.0  0.0
    696  100   2.0     0   0.0  0.0
    697  100   4.0     0   0.0  0.0
    698  100   8.0     1  10.0  0.0
    699  100  16.0     0   0.0  0.0
    <BLANKLINE>
    [700 rows x 5 columns]
    """

    next_id = 1
    full_df = pd.DataFrame()
    idv_name = td.independent_variable.name
    for armid, arm in enumerate(td, start=1):
        amount = []
        evid = []
        time = []
        armcol = []
        for act in arm:
            for t in act.time_points:
                curtime = t + act.start_time
                armcol.append(armid)
                time.append(curtime)
                if isinstance(act, Observations):
                    evid.append(0)
                    amount.append(0)
                else:
                    evid.append(1)
                    amount.append(float(act.dose.amount))
        d = {idv_name: time, 'EVID': evid, 'AMT': amount}
        if len(td) > 1:
            d = {'ARM': armcol} | d
        one_id = pd.DataFrame(d)
        df = pd.concat([one_id] * arm.size, ignore_index=True)
        df.insert(0, 'ID', np.repeat(range(next_id, next_id + arm.size), len(one_id)))
        full_df = pd.concat([full_df, df], ignore_index=True)
        next_id += arm.size
    full_df = full_df.sort_values(
        by=["ID", idv_name, "EVID"], ascending=[True, True, False]
    ).reset_index(drop=True)
    full_df['DV'] = 0.0
    return full_df


def infer_design_from_dataset(model: Model) -> TrialDesign:
    df = get_and_check_dataset(model)
    td = create_trial_design(idv_name=model.datainfo.idv_column.name)
    nids = len(get_ids(model))
    td = add_arm(td, name="DRUG", size=nids)

    # Rule 1: observation time points present in over 50% of individuals moved to closest nice number
    # Rule 2: Currently: don't care about the remaining time points

    all_doses = get_doses(model)
    doses = all_doses.reset_index()[['ID', 'TIME']]
    frequent_dosing_times = _get_frequent_time_points(doses)

    # FIXME: This depends on times already being nice numbers
    for dt in frequent_dosing_times:
        amts = df.loc[(df['TIME'] == dt) & df['AMT'] != 0, 'AMT']
        if len(amts.unique()) == 1:
            td = add_administration(
                td, arm="DRUG", variable="AMT", time_points=[dt], amount=amts.iloc[0]
            )

    observations = get_observations(model).reset_index()[['ID', 'TIME']]
    frequent_observation_times = _get_frequent_time_points(observations)

    td = add_observations(
        td, arm="DRUG", variable="DV", time_points=list(frequent_observation_times)
    )
    return td


# Could move to internal
NICE_NUMBERS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _move_to_closest_nice_number(x):
    current_nice = 0.0
    current_delta = 2.0
    for d in NICE_NUMBERS:
        nice_num = math.floor(x) + d
        delta = abs(x - nice_num)
        if delta < current_delta:
            current_delta = delta
            current_nice = nice_num
    return current_nice


def _niceify(x):
    return pd.Series(x).apply(_move_to_closest_nice_number)


def _get_frequent_time_points(df):
    # df has an ID and a TIME column
    nids = len(df['ID'].unique())
    freq = df.groupby('TIME')['ID'].nunique()
    frequent_times = _niceify(freq[freq / nids > 0.5].index)
    return frequent_times
