import re
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.deps.scipy import linalg
from pharmpy.model import Model, Results
from pharmpy.modeling import plot_individual_predictions
from pharmpy.tools import read_modelfit_results
from pharmpy.tools.psn_helpers import model_paths, options_from_command


@dataclass(frozen=True)
class CDDResults(Results):
    """CDD Results class"""

    rst_path = Path(__file__).resolve().parent / 'report.rst'

    case_results: Optional[Any] = None
    case_column: Optional[Any] = None
    individual_predictions_plot: Optional[Any] = None


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
        chol, islow = linalg.cho_factor(covariance_matrix)
        delta_matrix = cdd_estimates - base_estimate
        x = linalg.solve_triangular(chol, delta_matrix.transpose(), lower=islow, trans=1)
        return list(map(np.linalg.norm, x.transpose()))
    except Exception:
        return None


def compute_delta_ofv(base_model_results, cdd_model_results, skipped_individuals):
    iofv = base_model_results.individual_ofv
    if iofv is None:
        return [np.nan] * len(cdd_model_results)

    cdd_ofvs = [res.ofv if res is not None else np.nan for res in cdd_model_results]

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


def compute_covariance_ratios(cdd_model_results, covariance_matrix):
    try:
        orig_det = np.linalg.det(covariance_matrix)
        return [
            sqrt(np.linalg.det(res.covariance_matrix) / orig_det)
            if res is not None and res.covariance_matrix is not None
            else np.nan
            for res in cdd_model_results
        ]
    except Exception:
        return None


def calculate_results(
    base_model: Model,
    base_model_results,
    cdd_models: List[Model],
    cdd_model_results,
    case_column,
    skipped_individuals,
    **_,
):
    """Calculate CDD results"""

    if base_model_results is None:
        raise ValueError('cdd base model has no results')

    cdd_estimates = pd.DataFrame(
        data=[
            pd.Series(res.parameter_estimates, name=m.name)
            for m, res in zip(cdd_models, cdd_model_results)
            if res is not None
        ]
    )

    cdd_model_names = [m.name for m in cdd_models]

    # create Series of NaN values and then replace any computable results
    cook_temp = pd.Series(np.nan, index=cdd_model_names)
    try:
        base_model_results.covariance_matrix
    except Exception:
        pass
    else:
        cook_temp.update(
            pd.Series(
                compute_cook_scores(
                    base_model_results.parameter_estimates,
                    cdd_estimates,
                    base_model_results.covariance_matrix,
                ),
                index=cdd_estimates.index,
                dtype=np.float64,
            )
        )

    jack_cook_score = None
    if len(cdd_model_names) == cdd_estimates.shape[0]:
        # all models have results
        jackkknife_covariance_matrix = compute_jackknife_covariance_matrix(cdd_estimates)
        jack_cook_score = pd.Series(
            compute_cook_scores(
                base_model_results.parameter_estimates,
                cdd_estimates,
                jackkknife_covariance_matrix,
            ),
            index=cdd_model_names,
        )

    dofv = compute_delta_ofv(base_model_results, cdd_model_results, skipped_individuals)
    dofv_influential = [elt > 3.86 for elt in dofv]
    infl_list = [
        skipped[0]
        for skipped, infl in zip(skipped_individuals, dofv_influential)
        if infl and len(skipped) == 1
    ]

    if infl_list:
        try:
            iplot = plot_individual_predictions(
                base_model,
                base_model_results.predictions[['PRED', 'CIPREDI']],
                individuals=infl_list,
            )
        except Exception:
            iplot = None
    else:
        iplot = None

    try:
        covmatrix = base_model_results.covariance_matrix
    except Exception:
        covratios = np.nan
    else:
        covratios = compute_covariance_ratios(cdd_model_results, covmatrix)

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
    return CDDResults(
        case_column=case_column, individual_predictions_plot=iplot, case_results=case_results
    )


def psn_cdd_options(path: Union[str, Path]):
    path = Path(path)
    options: Dict[str, Any] = dict(model_path=None, outside_n_sd_check=None, case_column='ID')
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


def psn_cdd_skipped_individuals(path: Union[str, Path]):
    path = Path(path) / 'skipped_individuals1.csv'

    with open(path) as skipped:
        # rows may have different number of values, cannot use pd.read_csv
        a = [row.rstrip().split(',') for row in skipped]
    # If scientific notation convert to proper integer. Only supports integer IDs
    a = [[str(int(float(elt))) for elt in row] for row in a]
    return a


def psn_cdd_results(path: Union[str, Path], base_model_path=None):
    """Create cdd results from a PsN CDD run

    :param path: Path to PsN cdd run directory
    :return: A :class:`CDDResults` object

    """
    path = Path(path)
    if not path.is_dir():
        raise IOError(f'Could not find cdd folder: {str(path)}')

    options = psn_cdd_options(path)

    if base_model_path is None:
        base_model_path = Path(options['model_path'])
    base_model = Model.parse_model(base_model_path)
    base_model_results = read_modelfit_results(base_model_path)

    paths = model_paths(path, 'cdd_*.mod')
    cdd_models = list(map(Model.parse_model, paths))
    cdd_results = list(map(read_modelfit_results, paths))
    skipped_individuals = psn_cdd_skipped_individuals(path)

    res = calculate_results(
        base_model,
        base_model_results,
        cdd_models,
        cdd_results,
        options['case_column'],
        skipped_individuals,
    )
    return res
