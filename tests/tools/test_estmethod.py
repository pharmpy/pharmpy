import pytest

from pharmpy.tools.estmethod.algorithms import _create_base_model, _create_candidate_model
from pharmpy.tools.estmethod.tool import SOLVERS, create_workflow, validate_input
from pharmpy.workflows import ModelEntry


@pytest.mark.parametrize(
    'algorithm, methods, solvers, parameter_uncertainty_methods, no_of_models',
    [
        ('exhaustive', ['foce'], None, None, 1),
        ('exhaustive', ['foce', 'laplace'], None, None, 2),
        ('exhaustive', ['foce', 'imp'], ['lsoda'], None, 2),
        ('exhaustive', ['foce'], 'all', None, len(SOLVERS)),
        ('exhaustive_with_update', ['foce'], None, None, 2),
        ('exhaustive_with_update', ['foce'], None, 'all', 6),
        ('exhaustive_with_update', ['foce', 'laplace'], None, None, 4),
        ('exhaustive_with_update', ['laplace'], None, None, 3),
        ('exhaustive_with_update', ['foce'], ['lsoda'], None, 3),
        ('exhaustive_with_update', ['foce'], 'all', None, len(SOLVERS) * 2 + 1),
        ('exhaustive', ['foce'], None, ['sandwich', 'cpg'], 2),
        ('exhaustive', ['foce', 'imp'], None, ['sandwich', 'cpg'], 4),
    ],
)
def test_algorithm(algorithm, methods, solvers, parameter_uncertainty_methods, no_of_models):
    wf = create_workflow(
        algorithm,
        methods=methods,
        solvers=solvers,
        parameter_uncertainty_methods=parameter_uncertainty_methods,
    )
    fit_tasks = [task.name for task in wf.tasks if task.name.startswith('run')]

    assert len(fit_tasks) == no_of_models


@pytest.mark.parametrize(
    'est_rec, eval_rec',
    [
        (
            '$ESTIMATION METHOD=COND INTER MAXEVAL=9999 AUTO=1 PRINT=10',
            '$ESTIMATION METHOD=IMP INTER EONLY=1 MAXEVAL=9999 ISAMPLE=10000 ' 'NITER=10 PRINT=10',
        ),
    ],
)
def test_create_base_model(
    load_model_for_test,
    pheno_path,
    est_rec,
    eval_rec,
    parameter_uncertainty_method=None,
    add_eval_after_est=True,
):
    model = load_model_for_test(pheno_path)
    assert len(model.execution_steps) == 1
    model_entry = ModelEntry.create(model)
    base_model_entry = _create_base_model(
        parameter_uncertainty_method,
        add_eval_after_est,
        model_entry=model_entry,
    )
    base_model = base_model_entry.model
    assert len(base_model.execution_steps) == 2
    assert base_model.code.split('\n')[-5] == est_rec
    assert base_model.code.split('\n')[-4] == eval_rec


@pytest.mark.parametrize(
    'method, est_rec, eval_rec',
    [
        (
            'FO',
            '$ESTIMATION METHOD=ZERO INTER MAXEVAL=9999 AUTO=1 PRINT=10',
            '$ESTIMATION METHOD=IMP INTER EONLY=1 MAXEVAL=9999 ISAMPLE=10000 NITER=10 PRINT=10',
        ),
        (
            'LAPLACE',
            '$ESTIMATION METHOD=COND LAPLACE INTER MAXEVAL=9999 AUTO=1 PRINT=10',
            '$ESTIMATION METHOD=IMP LAPLACE INTER EONLY=1 MAXEVAL=9999 ISAMPLE=10000 '
            'NITER=10 PRINT=10',
        ),
    ],
)
def test_create_candidate_model(
    load_model_for_test,
    pheno_path,
    method,
    est_rec,
    eval_rec,
    parameter_uncertainty_method=None,
    add_eval_after_est=True,
):
    model = load_model_for_test(pheno_path)
    assert len(model.execution_steps) == 1
    model_entry = ModelEntry.create(model)
    candidate_model_entry = _create_candidate_model(
        '',
        method,
        None,
        parameter_uncertainty_method,
        add_eval_after_est,
        update_inits=False,
        only_evaluation=False,
        model_entry=model_entry,
    )
    candidate_model = candidate_model_entry.model
    assert len(candidate_model.execution_steps) == 2
    assert candidate_model.code.split('\n')[-5] == est_rec
    assert candidate_model.code.split('\n')[-4] == eval_rec


@pytest.mark.parametrize(
    (
        'args',
        'exception',
        'match',
    ),
    [
        (
            dict(algorithm='x', methods='all'),
            ValueError,
            'Invalid `algorithm`',
        ),
        (
            dict(algorithm='exhaustive', methods=None, solvers=None),
            ValueError,
            'Invalid search space options',
        ),
        (
            dict(algorithm='exhaustive', solvers=['LSODA']),
            ValueError,
            'Invalid input `model`',
        ),
    ],
)
def test_validate_input(load_model_for_test, pheno_path, args, exception, match):
    model = load_model_for_test(pheno_path)
    kwargs = {**args, 'model': model}
    with pytest.raises(exception, match=match):
        validate_input(**kwargs)
