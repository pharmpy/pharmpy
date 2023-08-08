import pytest

from pharmpy.tools.estmethod.algorithms import _create_candidate_model
from pharmpy.tools.estmethod.tool import SOLVERS, create_workflow, validate_input


@pytest.mark.parametrize(
    'algorithm, methods, solvers, covs, no_of_models',
    [
        ('exhaustive', ['foce'], None, None, 1),
        ('exhaustive', ['foce', 'laplace'], None, None, 2),
        ('exhaustive', ['foce', 'imp'], ['lsoda'], None, 2),
        ('exhaustive', ['foce'], 'all', None, len(SOLVERS)),
        ('exhaustive_with_update', ['foce'], None, None, 2),
        ('exhaustive_with_update', ['foce', 'laplace'], None, None, 4),
        ('exhaustive_with_update', ['laplace'], None, None, 3),
        ('exhaustive_with_update', ['foce'], ['lsoda'], None, 3),
        ('exhaustive_with_update', ['foce'], 'all', None, len(SOLVERS) * 2 + 1),
        ('exhaustive', ['foce'], None, ['sandwich', 'cpg'], 2),
        ('exhaustive', ['foce', 'imp'], None, ['sandwich', 'cpg'], 4),
    ],
)
def test_algorithm(algorithm, methods, solvers, covs, no_of_models):
    wf = create_workflow(algorithm, methods=methods, solvers=solvers, covs=covs)
    fit_tasks = [task.name for task in wf.tasks if task.name.startswith('run')]

    assert len(fit_tasks) == no_of_models


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
def test_create_est_model(load_model_for_test, pheno_path, method, est_rec, eval_rec):
    model = load_model_for_test(pheno_path)
    assert len(model.estimation_steps) == 1
    est_model = _create_candidate_model(
        method, None, None, model=model, update=False, is_eval_candidate=False
    )
    assert len(est_model.estimation_steps) == 2
    assert est_model.model_code.split('\n')[-5] == est_rec
    assert est_model.model_code.split('\n')[-4] == eval_rec


@pytest.mark.parametrize(
    (
        'args',
        'exception',
        'match',
    ),
    [
        (
            dict(algorithm='x'),
            ValueError,
            'Invalid `algorithm`',
        ),
        (
            dict(algorithm='exhaustive', methods=None, solvers=None),
            ValueError,
            'Invalid search space options',
        ),
        (
            dict(algorithm='exhaustive', solvers=['lsoda']),
            ValueError,
            'Invalid input `model`',
        ),
    ],
)
def test_validate_input(load_model_for_test, pheno_path, args, exception, match):
    model = load_model_for_test(pheno_path)
    kwargs = {**args, 'model': model}
    with pytest.raises(ValueError, match=match):
        validate_input(**kwargs)
