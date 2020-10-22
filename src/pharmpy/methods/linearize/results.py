from pathlib import Path

import pandas as pd

from pharmpy import Model
from pharmpy.results import Results


class LinearizeResults(Results):
    def __init__(self, iofv=None):
        self.iofv = iofv


def calculate_results(base_model, linear_model):
    baseres = base_model.modelfit_results
    linearres = linear_model.modelfit_results
    iofv = pd.DataFrame(
        {
            'base': baseres.individual_ofv,
            'linear': linearres.individual_ofv,
            'delta': linearres.individual_ofv - baseres.individual_ofv,
        }
    )
    res = LinearizeResults(iofv=iofv)
    return res


def psn_linearize_results(path):
    """Create linearize results from a PsN linearize run

    :param path: Path to PsN linearize run directory
    :return: A :class:`QAResults` object
    """
    path = Path(path)

    base_model = Model(path / 'scm_dir1' / 'derivatives.mod')
    lin_path = list(path.glob('*_linbase.mod'))[0]
    lin_model = Model(lin_path)

    res = calculate_results(base_model, lin_model)
    return res
