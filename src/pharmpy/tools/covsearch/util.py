from collections import Counter, defaultdict
from dataclasses import astuple, dataclass, replace
from itertools import count
from typing import Any, Callable, Iterable, List, Literal, Optional, Tuple, Union

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.modeling import get_pk_parameters, remove_covariate_effect, set_estimation_step
from pharmpy.modeling.covariate_effect import get_covariates_allowed_in_covariate_effect
from pharmpy.modeling.lrt import best_of_many as lrt_best_of_many
from pharmpy.modeling.lrt import p_value as lrt_p_value
from pharmpy.modeling.lrt import test as lrt_test
from pharmpy.tools import is_strictness_fulfilled
from pharmpy.tools.common import create_results, update_initial_estimates
from pharmpy.tools.mfl.feature.covariate import EffectLiteral
from pharmpy.tools.mfl.feature.covariate import features as covariate_features
from pharmpy.tools.mfl.feature.covariate import parse_spec, spec
from pharmpy.tools.mfl.helpers import all_funcs
from pharmpy.tools.mfl.parse import parse as mfl_parse
from pharmpy.tools.mfl.statement.feature.covariate import Covariate
from pharmpy.tools.mfl.statement.feature.symbols import Wildcard
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.run import summarize_modelfit_results_from_entries
from pharmpy.tools.scm.results import candidate_summary_dataframe, ofv_summary_dataframe
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder, call_workflow
from pharmpy.workflows.results import ModelfitResults


def set_maxevals(model, results, max_evals=3.1):
    max_eval_number = round(max_evals * results.function_evaluations_iterations.loc[1])
    first_es = model.execution_steps[0]
    model = set_estimation_step(model, first_es.method, 0, maximum_evaluations=max_eval_number)
    return ModelEntry.create(
        model=model.replace(name="input", description=""), parent=None, modelfit_results=results
    )