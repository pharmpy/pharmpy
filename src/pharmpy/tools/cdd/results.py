import re
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, solve_triangular

from pharmpy import Model
from pharmpy.results import Results
from pharmpy.tools.psn_helpers import model_paths, options_from_command


class CDDResults(Results):
    """CDD Results class"""

    rst_path = Path(__file__).parent / 'report.rst'

    def __init__(self, case_results=None, case_column=None, individual_predictions_plot=None):
        self.case_results = case_results
        self.case_column = case_column
        self.individual_predictions_plot = individual_predictions_plot


def compute_cook_scores(base_estimate, cdd_estimates, covariance_matrix):
    # covariance_matrix may be jackknife estimate of covariance
    try:
        # chol * chol^T = covariance_matrix
        # inv(chol^T) * inv(chol) = inv(covariance_matrix)
        # delta_vector * inv(covariance_matrix) * delta_vector^T =
        #   delta_vector * inv(chol^T) * inv(chol)  * delta_vector^T which is
        #   the 2-norm of x   where x is
        # solution to triangular system  delta_vector^T = chol * x
        # Below we solve for all delta-vectors in one line
        chol, islow = cho_factor(covariance_matrix)
        delta_matrix = cdd_estimates - base_estimate
        x = solve_triangular(chol, delta_matrix.transpose(), lower=islow, trans=1)
        return [np.linalg.norm(v) for v in x.transpose()]
    except Exception:
        return None


def compute_delta_ofv(base_model, cdd_models, skipped_individuals):
    iofv = base_model.modelfit_results.individual_ofv
    if iofv is None:
        return [np.nan for m in cdd_models]

    cdd_ofvs = [m.modelfit_results.ofv if m.modelfit_results else np.nan for m in cdd_models]

    # need to set dtype for index.difference to work
    skipped_indices = [
        pd.Index(np.array(skipped, dtype=iofv.index.dtype)) for skipped in skipped_individuals
    ]

    return [
        sum(iofv[iofv.index.difference(skipped)]) - ofv
        for skipped, ofv in zip(skipped_indices, cdd_ofvs)
    ]


def compute_jackknife_covariance_matrix(cdd_estimates):
    bigN = len(cdd_estimates.index)
    delta_est = cdd_estimates - cdd_estimates.mean()
    return delta_est.transpose() @ delta_est * (bigN - 1) / bigN


def compute_covariance_ratios(cdd_models, covariance_matrix):
    try:
        orig_det = np.linalg.det(covariance_matrix)
        return [
            sqrt(np.linalg.det(m.modelfit_results.covariance_matrix) / orig_det)
            if m.modelfit_results and m.modelfit_results.covariance_matrix is not None
            else np.nan
            for m in cdd_models
        ]
    except Exception:
        return None


def calculate_results(base_model, cdd_models, case_column, skipped_individuals, **kwargs):
    """Calculate CDD results"""

    if base_model.modelfit_results is None:
        raise ValueError('cdd base model has no results')

    cdd_estimates = pd.DataFrame(
        data=[
            pd.Series(m.modelfit_results.parameter_estimates, name=m.name)
            for m in cdd_models
            if m.modelfit_results
        ]
    )

    cdd_model_names = [m.name for m in cdd_models]

    res = CDDResults(case_column=case_column)

    # create Series of NaN values and then replace any computable results
    cook_temp = pd.Series(np.nan, index=cdd_model_names)
    try:
        base_model.modelfit_results.covariance_matrix
    except Exception:
        pass
    else:
        cook_temp.update(
            pd.Series(
                compute_cook_scores(
                    base_model.modelfit_results.parameter_estimates,
                    cdd_estimates,
                    base_model.modelfit_results.covariance_matrix,
                ),
                index=cdd_estimates.index,
            )
        )

    jack_cook_score = None
    if len(cdd_model_names) == cdd_estimates.shape[0]:
        # all models have results
        jackkknife_covariance_matrix = compute_jackknife_covariance_matrix(cdd_estimates)
        jack_cook_score = pd.Series(
            compute_cook_scores(
                base_model.modelfit_results.parameter_estimates,
                cdd_estimates,
                jackkknife_covariance_matrix,
            ),
            index=cdd_model_names,
        )

    dofv = compute_delta_ofv(base_model, cdd_models, skipped_individuals)
    dofv_influential = [elt > 3.86 for elt in dofv]
    infl_list = [
        skipped[0]
        for skipped, infl in zip(skipped_individuals, dofv_influential)
        if infl and len(skipped) == 1
    ]

    if infl_list:
        try:
            iplot = base_model.modelfit_results.plot_individual_predictions(
                individuals=infl_list, predictions=['PRED', 'CIPREDI']
            )
        except KeyError:
            iplot = None
    else:
        iplot = None
    res.individual_predictions_plot = iplot

    try:
        covmatrix = base_model.modelfit_results.covariance_matrix
    except Exception:
        covratios = np.nan
    else:
        covratios = compute_covariance_ratios(cdd_models, covmatrix)

    case_results = pd.DataFrame(
        {
            'cook_score': cook_temp,
            'jackknife_cook_score': jack_cook_score,
            'delta_ofv': dofv,
            'dofv_influential': dofv_influential,
            'covariance_ratio': covratios,
            'skipped_individuals': skipped_individuals,
        },
        index=cdd_model_names,
    )

    case_results.index = pd.RangeIndex(start=1, stop=len(case_results) + 1)
    res.case_results = case_results
    return res


def psn_cdd_options(path):
    path = Path(path)
    options = dict(model_path=None, outside_n_sd_check=None, case_column='ID')
    with open(path / 'meta.yaml') as meta:
        cmd = None
        for row in meta:
            if cmd is not None:
                if re.match(r'\s', row):  # continuation is indented
                    cmd += row  # must not strip
                    continue
                else:  # no continuation: parse and remove
                    for k, v in options_from_command(cmd).items():
                        if 'case_column'.startswith(k):
                            options['case_column'] = v
                            break
                    cmd = None
            row = row.strip()
            if row.startswith('model_files:'):
                row = next(meta).strip()
                options['model_path'] = re.sub(r'^-\s*', '', row)
            elif row.startswith('outside_n_sd_check: '):
                options['outside_n_sd_check'] = int(re.sub(r'\D', '', row))
            elif row.startswith('command_line: '):
                cmd = row
    return options


def psn_cdd_skipped_individuals(path):
    path = Path(path) / 'skipped_individuals1.csv'

    with open(path) as skipped:
        # rows may have different number of values, cannot use pd.read_csv
        a = [row.rstrip().split(',') for row in skipped]
    # If scientific notation convert to proper integer. Only supports integer IDs
    a = [[str(int(float(elt))) for elt in row] for row in a]
    return a


def psn_cdd_results(path):
    """Create cdd results from a PsN CDD run

    :param path: Path to PsN cdd run directory
    :return: A :class:`CDDResults` object

    """
    path = Path(path)
    if not path.is_dir():
        raise IOError(f'Could not find cdd folder: {str(path)}')

    options = psn_cdd_options(path)

    model_path = Path(options['model_path'])
    base_model = Model(model_path)

    cdd_models = [Model(p) for p in model_paths(path, 'cdd_*.mod')]
    skipped_individuals = psn_cdd_skipped_individuals(path)

    res = calculate_results(base_model, cdd_models, options['case_column'], skipped_individuals)
    return res
