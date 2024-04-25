"""
.. highlight:: console

============================
The CLI interface of Pharmpy
============================

Examples
--------

Example argument lines (only preceded by ``pharmpy`` or ``python3 -m pharmpy``)::

    # filter data (as model prescribes) and output to file 'output.csv'
    pharmpy data write run1.mod -o output.csv

Entrypoints and Subcommands
---------------------------

Standard practice of packaging is to define "entrypoints" in the setuptools ``setup.py``. Currently:

.. code-block:: python

  entry_points={
      'console_scripts': [
          'pharmpy           = pharmpy.cli:main',
      ]
  },

Main entrypoint is ``pharmpy`` (which maps :func:`main`), but each subcommand could be given a
shortcutting entrypoint (e.g. :func:`sumo`). This would be accomplished via invoking:

Installing Pharmpy Wheel-build should install all of these to PATH. Then, these are clean targets
for post-install hooks to symlink. Either to pollute the PATH namespace with traditional PsN-styled
"binscripts", as::

    /usr/bin/execute               ->  /usr/bin/pharmpy execute

Or, to keep multiple versions when upgrading Pharmpy.

On Logging
----------

As a CLI we are at the top, interacting with the user. All of Pharmpy will get messages
filtered through here (loggers organize themselves in a hierarchy based on their period-split names).

.. warning::
    Loggers **mustn't** be configured if imported as library (e.g. set handlers and such),
    since that should bubble up into the lap of whomever is at the top (of the Python
    period-separated namespace).

Definitions
===========
"""

import argparse
import pydoc
import sys
import warnings
from collections import namedtuple
from pathlib import Path
from textwrap import dedent

import pharmpy
from pharmpy.internals.fs.path import path_absolute

from .deps import pandas as pd

formatter = argparse.ArgumentDefaultsHelpFormatter


def warnings_formatter(message, *args, **kwargs):
    return f'Warning: {message}\n'


warnings.formatwarning = warnings_formatter


def error(exception):
    """Raise the exception with no traceback printed

    Used for non-recoverables in CLI. Exceptions giving traceback are bugs!
    """

    def exc_only_hook(exception_type, exception, traceback):
        print(f'{exception_type.__name__}: {exception}')

    sys.excepthook = exc_only_hook
    raise exception


def format_keyval_pairs(data_dict, sort=True, right_just=False):
    """Formats lines from *data_dict*."""

    if not data_dict:
        return []

    key_width = max(len(field) for field in data_dict.keys())
    if sort:
        data_dict = dict(sorted(data_dict.items()))
    if right_just:
        line_format = '  %%%ds\t%%s' % key_width
    else:
        line_format = '%%-%ds\t%%s' % key_width

    lines = []
    for key, values in data_dict.items():
        if isinstance(values, str):
            values = values.splitlines()
        keys = [key] + [''] * (len(values) - 1)
        for k, v in zip(keys, values):
            lines += [line_format % (k, v)]

    return lines


def run_bootstrap(args):
    from pharmpy.tools import run_bootstrap

    model, res = args.model
    run_bootstrap(model, res, resamples=args.samples)


def run_execute(args):
    from pharmpy.tools import fit

    models = [model for model, _ in args.models]
    fit(models)


def run_modelsearch(args):
    from pharmpy.tools import run_tool

    model, res = args.model
    run_tool(
        'modelsearch',
        args.mfl,
        args.algorithm,
        rank_type=args.rank_type,
        cutoff=args.cutoff,
        iiv_strategy=args.iiv_strategy,
        results=res,
        model=model,
        path=args.path,
    )


def run_iivsearch(args):
    from pharmpy.tools import run_tool

    model, res = args.model
    run_tool(
        'iivsearch',
        args.algorithm,
        iiv_strategy=args.iiv_strategy,
        rank_type=args.rank_type,
        cutoff=args.cutoff,
        results=res,
        model=model,
        path=args.path,
    )


def run_iovsearch(args):
    from pharmpy.tools import run_tool

    model, res = args.model
    run_tool(
        'iovsearch',
        column=args.column,
        list_of_parameters=args.list_of_parameters,
        rank_type=args.rank_type,
        cutoff=args.cutoff,
        distribution=args.distribution,
        results=res,
        model=model,
        path=args.path,
    )


def run_covsearch(args):
    from pharmpy.tools import run_tool

    model, res = args.model
    run_tool(
        'covsearch',
        effects=args.effects,
        p_forward=args.p_forward,
        max_steps=args.max_steps,
        algorithm=args.algorithm,
        results=res,
        model=model,
        path=args.path,
    )


def run_ruvsearch(args):
    from pharmpy.tools import run_tool

    model, res = args.model
    run_tool(
        'ruvsearch',
        results=res,
        model=model,
        groups=args.groups,
        p_value=args.p_value,
        skip=args.skip,
        path=args.path,
    )


def run_allometry(args):
    from pharmpy.tools import run_tool

    model, res = args.model
    run_tool(
        'allometry',
        results=res,
        model=model,
        allometric_variable=args.allometric_variable,
        reference_value=args.reference_value,
        parameters=args.parameters,
        initials=args.initials,
        lower_bounds=args.lower_bounds,
        upper_bounds=args.upper_bounds,
        fixed=not args.non_fixed,
        path=args.path,
    )


def run_estmethod(args):
    from pharmpy.tools import run_tool

    model, res = args.model
    try:
        methods = args.methods.split(" ")
    except AttributeError:
        methods = args.methods
    try:
        solvers = args.solvers.split(" ")
    except AttributeError:
        solvers = args.solvers

    run_tool(
        'estmethod',
        args.algorithm,
        methods=methods,
        solvers=solvers,
        results=res,
        model=model,
        path=args.path,
    )


def run_amd(args):
    from pharmpy.tools import run_amd

    run_amd(
        args.input_path,
        modeltype=args.modeltype,
        administration=args.administration,
        cl_init=args.cl_init,
        vc_init=args.vc_init,
        mat_init=args.mat_init,
        search_space=args.search_space,
        lloq_method=args.lloq_method,
        lloq_limit=args.lloq_limit,
        order=args.order,
        allometric_variable=args.allometric_variable,
        occasion=args.occasion,
        path=args.path,
        resume=args.resume,
    )


def run_retries(args):
    from pharmpy.tools import run_tool

    model, res = args.model
    run_tool(
        'retries',
        results=res,
        model=model,
        number_of_candidates=args.number_of_candidates,
    )


def data_write(args):
    """Subcommand to write a dataset."""
    try:
        from pharmpy.modeling import write_csv

        # If no output_file supplied will use name of df
        path = write_csv(args.model[0], path=args.output_file, force=args.force)
        print(f'Dataset written to {path}')
    except OSError as e:
        error(e)


def data_print(args):
    """Subcommand to print a dataset to stdout via pager"""
    try:
        df = args.model_or_dataset.dataset
    except AttributeError:
        df = args.model_or_dataset
    if args.columns:
        try:
            df = df[args.columns]
        except KeyError as e:
            error(e)
    pydoc.pager(df.to_string())


def data_filter(args):
    """Subcommand to filter a dataset"""
    try:
        df = args.model_or_dataset.dataset
    except TypeError:
        df = args.model_or_dataset
    expression = ' '.join(args.expressions)
    try:
        df = df.query(expression)
    except SyntaxError:
        error(SyntaxError(f'Invalid syntax of query: "{expression}"'))
    except BaseException as e:
        error(e)
    write_model_or_dataset(args.model_or_dataset, df, args.output_file, args.force)


def data_append(args):
    """Subcommand to append a column to a dataset"""
    try:
        df = args.model_or_dataset.dataset
    except TypeError:
        df = args.model_or_dataset
    try:
        df = df.eval(args.expression)
    except SyntaxError:
        error(SyntaxError(f'Invalid syntax of expression: "{args.expression}"'))
    write_model_or_dataset(args.model_or_dataset, df, args.output_file, args.force)


def write_model_or_dataset(model_or_dataset, new_df, path, force):
    """Write model or dataset to output_path or using default names"""
    if isinstance(model_or_dataset, pd.DataFrame):
        # Is a dataset
        try:
            # If no output_file supplied will use name of df
            if force or not path.is_file():
                path = new_df.to_csv(path, index=False)
                print(f'Dataset written to {path}')
            else:
                error(exception=FileExistsError("Use -f or --force to force an overwrite"))
        except FileExistsError as e:
            error(
                exception=FileExistsError(
                    f'{e.args[0]} Use -f  or --force to ' 'force an overwrite'
                )
            )
    else:
        # Is a model
        model = model_or_dataset
        if new_df is not None and new_df is not model.dataset:
            model = model.replace(dataset=new_df)
        try:
            from pharmpy.modeling import bump_model_number, write_model

            if path:
                if not path.is_dir():
                    model = bump_model_number(model, path)
                write_model(model, path=path, force=force)
            else:
                model = bump_model_number(model, path='.')
                write_model(model, force=force)
        except FileExistsError as e:
            error(FileExistsError(f'{e.args[0]} Use -f or --force to ' 'force an overwrite'))


def data_resample(args):
    """Subcommand to resample a dataset."""
    from pharmpy.modeling import resample_data

    resampler = resample_data(
        args.model_or_dataset,
        args.group,
        resamples=args.resamples,
        stratify=args.stratify,
        replace=args.replace,
        sample_size=args.sample_size,
        name_pattern=f'{args.model_or_dataset.name}_resample_{{}}',
    )
    for resampled_obj, _ in resampler:
        try:
            try:
                from pharmpy.modeling import write_model

                write_model(resampled_obj)
            except AttributeError:
                # FIXME!
                resampled_obj.to_csv()
        except OSError as e:
            error(e)


def data_deidentify(args):
    """Subcommand to deidentify a dataset
    Will be default try to overwrite original file (but won't unless force)
    """
    path = check_input_path(args.dataset)
    dataset = pd.read_csv(path, dtype=str)

    from pharmpy.modeling import deidentify_data

    if args.datecols:
        datecols = args.datecols.split(',')
    else:
        datecols = None
    df = deidentify_data(dataset, id_column=args.idcol, date_columns=datecols)

    if args.output_file:
        output_file = args.output_file
    else:
        output_file = path
    write_model_or_dataset(df, df, output_file, args.force)


def data_reference(args):
    """Subcommand to replace columns with reference values"""
    from pharmpy.modeling import set_reference_values

    d = key_vals(args.colrefs)
    model, _ = args.model
    model = set_reference_values(model, d)
    write_model_or_dataset(model, model.dataset, args.output_file, args.force)


def info(args):
    """Subcommand to print Pharmpy info (and brag a little bit)."""

    import pharmpy

    Install = namedtuple('VersionInfo', ['version', 'authors', 'directory'])
    inst = Install(
        pharmpy.__version__,
        'A list of authors can be found in the AUTHORS.rst',
        str(Path(pharmpy.__file__).resolve().parent),
    )

    lines = format_keyval_pairs(inst._asdict(), right_just=True)
    print('\n'.join(lines))


def model_print(args):
    """Subcommand for formatting/printing model components."""
    lines = []
    for i, (model, _) in enumerate(args.models):
        lines += ['[%d/%d] %r' % (i + 1, len(args.models), model.name)]
        dict_ = {}
        try:
            dict_['dataset'] = repr(model.dataset)
        except FileNotFoundError as e:
            dict_['dataset'] = str(e)
        s = repr(model.parameters) + '\n\n'
        dict_['parameters'] = s
        s = repr(model.random_variables) + '\n\n'
        dict_['random_variables'] = s
        s = str(model.statements) + '\n\n'
        dict_['model_statements'] = s
        dict_lines = format_keyval_pairs(dict_, sort=False)
        lines += ['\t%s' % line for line in dict_lines]
    if len(lines) > 24:
        pydoc.pager('\n'.join(lines))
    else:
        print('\n'.join(lines))


def model_sample(args):
    model, res = args.model
    from pharmpy.modeling import sample_parameters_from_covariance_matrix

    samples = sample_parameters_from_covariance_matrix(
        model,
        res.parameter_estimates,
        res.covariance_matrix,
        n=args.samples,
    )
    for row, params in samples.iterrows():
        from pharmpy.modeling import set_initial_estimates, write_model

        set_initial_estimates(model, params)
        model = model.replace(name=f'sample_{row + 1}')
        write_model(model)


def update_inits(args):
    """Subcommand to update initial estimates from previous output."""
    from pharmpy.modeling import set_initial_estimates

    model, res = args.model
    set_initial_estimates(model, res.parameter_estimates)

    write_model_or_dataset(model, model.dataset, path=args.output_file, force=False)


def results_bootstrap(args):
    """Subcommand to generate bootstrap results"""
    from pharmpy.tools.bootstrap.results import psn_bootstrap_results

    if not args.psn_dir.is_dir():
        error(FileNotFoundError(str(args.psn_dir)))
    try:
        res = psn_bootstrap_results(args.psn_dir)
    except FileNotFoundError as e:
        error(e)
    res.to_json(path=args.psn_dir / 'results.json')
    res.to_csv(path=args.psn_dir / 'results.csv')


def results_cdd(args):
    from pharmpy.tools.cdd.results import psn_cdd_results

    if not args.psn_dir.is_dir():
        error(FileNotFoundError(str(args.psn_dir)))
    res = psn_cdd_results(args.psn_dir)
    res.to_json(path=args.psn_dir / 'results.json')
    res.to_csv(path=args.psn_dir / 'results.csv')


def results_frem(args):
    """Generate frem results"""
    from pharmpy.tools.frem.results import psn_frem_results

    if not args.psn_dir.is_dir():
        error(FileNotFoundError(str(args.psn_dir)))
    res = psn_frem_results(
        args.psn_dir,
        method=args.method,
        force_posdef_covmatrix=args.force_posdef_covmatrix,
        force_posdef_samples=args.force_posdef_samples,
    )
    res.to_json(path=args.psn_dir / 'results.json')
    res.to_csv(path=args.psn_dir / 'results.csv')

    from pharmpy.tools.reporting import create_report

    create_report(res, args.psn_dir)


def results_linearize(args):
    from pharmpy.tools.linearize.results import psn_linearize_results

    if not args.psn_dir.is_dir():
        error(FileNotFoundError(str(args.psn_dir)))
    res = psn_linearize_results(args.psn_dir)
    res.to_json(path=args.psn_dir / 'results.json')
    res.to_csv(path=args.psn_dir / 'results.csv')


def results_resmod(args):
    from pharmpy.tools.ruvsearch.results import psn_resmod_results

    if not args.psn_dir.is_dir():
        error(FileNotFoundError(str(args.psn_dir)))
    res = psn_resmod_results(args.psn_dir)
    res.to_json(path=args.psn_dir / 'results.json')
    res.to_csv(path=args.psn_dir / 'results.csv')


def results_scm(args):
    from pharmpy.tools.scm.results import psn_scm_results

    if not args.psn_dir.is_dir():
        error(FileNotFoundError(str(args.psn_dir)))
    res = psn_scm_results(args.psn_dir)
    res.to_json(path=args.psn_dir / 'results.json')
    res.to_csv(path=args.psn_dir / 'results.csv')


def results_simeval(args):
    from pharmpy.tools.simeval.results import psn_simeval_results

    if not args.psn_dir.is_dir():
        error(FileNotFoundError(str(args.psn_dir)))
    res = psn_simeval_results(args.psn_dir)
    res.to_json(path=args.psn_dir / 'results.json')
    res.to_csv(path=args.psn_dir / 'results.csv')


def results_ofv(args):
    """Subcommand to extract final ofv from multiple results"""
    ofvs = []
    for _, res in args.models:
        try:
            ofv = str(res.ofv)
        except AttributeError:
            ofv = 'NA'
        ofvs.append(ofv)
    print(' '.join(ofvs))


def results_print(args):
    """Subcommand to print any results"""
    if args.dir.is_dir():
        path = args.dir / 'results.json'
    elif args.dir.is_file():
        path = args.dir
    else:
        error(FileNotFoundError(str(args.dir)))
    from pharmpy.workflows.results import read_results

    res = read_results(path)
    print(res)


def results_qa(args):
    from pharmpy.tools.qa.results import psn_qa_results

    if not args.psn_dir.is_dir():
        error(FileNotFoundError(str(args.psn_dir)))
    res = psn_qa_results(args.psn_dir)
    res.to_json(path=args.psn_dir / 'results.json')
    res.to_csv(path=args.psn_dir / 'results.csv')


def results_report(args):
    """Subcommand to generate reports"""
    results_path = args.psn_dir / 'results.json'
    if not results_path.is_file():
        error(FileNotFoundError(results_path))
    from pharmpy.workflows.results import read_results

    res = read_results(results_path)
    from pharmpy.tools.reporting import create_report

    create_report(res, args.psn_dir)


def results_summary(args):
    """Subcommand to output summary of modelfit"""
    for model, res in args.models:
        print(model.name)
        print(res.parameter_summary())
        print()


def check_input_path(path):
    """Resolves path to input file and checks existence.

    Raises if not found or is dir, without tracebacks (see :func:`error_exit`).
    """
    try:
        path = Path(path)
        path = path_absolute(path)
    except FileNotFoundError:
        pass

    if not path.exists():
        exc = FileNotFoundError('No such input file: %r' % str(path))
        error(exc)
    elif path.is_dir():
        exc = IsADirectoryError('Is a directory (not an input file): %r' % str(path))
        error(exc)
    else:
        return path


# ------ Type functions --------------------


def input_model(path):
    """Returns :class:`~pharmpy.model.Model` from *path*.

    Raises if not found or is dir, without tracebacks (see :func:`error_exit`).
    """
    path = check_input_path(path)
    from pharmpy.model import Model
    from pharmpy.tools import read_modelfit_results

    model = Model.parse_model(path)
    res = read_modelfit_results(path)

    return model, res


def input_model_or_dataset(path):
    """Returns :class:`~pharmpy.model.Model` or pd.DataFrame from *path*"""
    path = check_input_path(path)
    try:
        from pharmpy.model import Model

        obj = Model.parse_model(path)
    except TypeError:
        obj = pd.read_csv(path)
    return obj


def comma_list(s):
    if s == '':
        return []
    return s.split(',')


def semicolon_list(s):
    if s == '':
        return []
    return s.split(';')


def key_vals(a):
    d = {}
    for arg in a:
        pair = arg.split('=')
        d[pair[0]] = float(pair[1])
    return d


def random_seed(seed):
    try:
        import numpy as np

        seed = int(seed)
        np.random.seed(seed)
    except ValueError as e:
        # Cannot reraise the ValueError as it would be caught by argparse
        error(argparse.ArgumentTypeError(str(e)))
    return seed


# for commands taking multiple model files as input
args_input = argparse.ArgumentParser(add_help=False)
group_input = args_input.add_argument_group(title='inputs')
group_input.add_argument(
    'models', metavar='FILE', type=input_model, nargs='+', help='input model files'
)

# for commands taking one model file as input
args_model_input = argparse.ArgumentParser(add_help=False)
group_model_input = args_model_input.add_argument_group(title='model')
group_model_input.add_argument('model', metavar='FILE', type=input_model, help='input model file')

# for commands with model file or dataset input
args_model_or_data_input = argparse.ArgumentParser(add_help=False)
group_model_or_data_input = args_model_or_data_input.add_argument_group(title='data_input')
group_model_or_data_input.add_argument(
    'model_or_dataset',
    metavar='FILE',
    type=input_model_or_dataset,
    help='input model or dataset file',
)

# for commands involving randomization
args_random = argparse.ArgumentParser(add_help=False)
group_random = args_random.add_argument_group(title='random seed')
group_random.add_argument(
    '--seed',
    type=random_seed,
    metavar='INTEGER',
    help='Provide a random seed. The seed must be an integer ' 'between 0 and 2^32 - 1',
)

# for commands with file output
args_output = argparse.ArgumentParser(add_help=False)
group_output = args_output.add_argument_group(title='outputs')
group_output.add_argument(
    '-f', '--force', action='store_true', help='remove existing destination files (all)'
)
group_output.add_argument(
    '-o', '--output_file', dest='output_file', metavar='file', type=Path, help='output file'
)

parser_definition = [
    {
        'run': {
            'subs': [
                {
                    'execute': {
                        'help': 'Execute one or more model',
                        'func': run_execute,
                        'parents': [args_input],
                    }
                },
                {
                    'bootstrap': {
                        'help': 'Bootstrap',
                        'func': run_bootstrap,
                        'parents': [args_model_input],
                        'args': [
                            {
                                'name': '--samples',
                                'type': int,
                                'help': 'Number of bootstrap datasets',
                            },
                        ],
                    }
                },
                {
                    'modelsearch': {
                        'help': 'Search for structural best model',
                        'func': run_modelsearch,
                        'parents': [args_model_input],
                        'args': [
                            {
                                'name': 'mfl',
                                'type': str,
                                'help': 'Search space to test',
                            },
                            {
                                'name': 'algorithm',
                                'type': str,
                                'help': 'Algorithm to use',
                            },
                            {
                                'name': '--rank_type',
                                'type': str,
                                'help': 'Name of function to use for ranking '
                                'candidates (default is bic).',
                                'default': 'bic',
                            },
                            {
                                'name': '--cutoff',
                                'type': float,
                                'help': 'Which selection criteria to rank models on',
                                'default': None,
                            },
                            {
                                'name': '--iiv_strategy',
                                'type': str,
                                'help': 'If/how IIV should be added to candidate models',
                                'default': 'absorption_delay',
                            },
                            {
                                'name': '--path',
                                'type': Path,
                                'help': 'Path to output directory',
                            },
                        ],
                    }
                },
                {
                    'iivsearch': {
                        'help': 'Search for best model IIV model',
                        'func': run_iivsearch,
                        'parents': [args_model_input],
                        'args': [
                            {
                                'name': 'algorithm',
                                'type': str,
                                'help': 'Algorithm to use',
                            },
                            {
                                'name': '--iiv_strategy',
                                'type': str,
                                'help': 'If/how IIV should be added to start model',
                                'default': 'no_add',
                            },
                            {
                                'name': '--rank_type',
                                'type': str,
                                'help': 'Name of function to use for ranking '
                                'candidates (default is bic).',
                                'default': 'bic',
                            },
                            {
                                'name': '--cutoff',
                                'type': float,
                                'help': 'Which selection criteria to rank models on',
                                'default': None,
                            },
                            {
                                'name': '--path',
                                'type': Path,
                                'help': 'Path to output directory',
                            },
                        ],
                    }
                },
                {
                    'iovsearch': {
                        'help': 'Search for best model IOV model',
                        'func': run_iovsearch,
                        'parents': [args_model_input],
                        'args': [
                            {
                                'name': '--column',
                                'type': str,
                                'help': 'Name of column in dataset to use as '
                                'occasion column (default is "OCC")',
                                'default': 'OCC',
                            },
                            {
                                'name': '--list_of_parameters',
                                'type': comma_list,
                                'help': 'List of parameters to test IOV on',
                                'default': None,
                            },
                            {
                                'name': '--rank_type',
                                'type': str,
                                'help': 'Name of function to use for ranking '
                                'candidates (default is bic).',
                                'default': 'bic',
                            },
                            {
                                'name': '--cutoff',
                                'type': float,
                                'help': 'Which selection criteria to rank models on',
                                'default': None,
                            },
                            {
                                'name': '--distribution',
                                'type': str,
                                'help': 'Which distribution added IOVs should have '
                                '(default is same-as-iiv)',
                                'default': 'same-as-iiv',
                            },
                            {
                                'name': '--path',
                                'type': Path,
                                'help': 'Path to output directory',
                            },
                        ],
                    }
                },
                {
                    'covsearch': {
                        'help': 'Identify covariates that explain some of the IIV',
                        'func': run_covsearch,
                        'parents': [args_model_input],
                        'args': [
                            {
                                'name': '--effects',
                                'type': str,
                                'help': 'The candidate effects to search through (required)',
                            },
                            {
                                'name': '--p_forward',
                                'type': float,
                                'help': 'The p-value threshold for forward '
                                'steps (default is `0.05`)',
                                'default': 0.05,
                            },
                            {
                                'name': '--p_backward',
                                'type': float,
                                'help': 'The p-value threshold for backward '
                                'steps (default is `0.01`)',
                                'default': 0.01,
                            },
                            {
                                'name': '--max_steps',
                                'type': int,
                                'help': 'The maximum number of search '
                                'algorithm steps to perform, or `-1` '
                                'for no maximum (default).',
                                'default': -1,
                            },
                            {
                                'name': '--algorithm',
                                'type': str,
                                'help': "The search algorithm to use (default "
                                "is `'scm-forward-then-backward'`)",
                                'default': 'scm-forward-then-backward',
                            },
                            {
                                'name': '--path',
                                'type': Path,
                                'help': 'Path to output directory',
                            },
                        ],
                    }
                },
                {
                    'ruvsearch': {
                        'help': 'Search for best residual error model',
                        'func': run_ruvsearch,
                        'parents': [args_model_input],
                        'args': [
                            {
                                'name': '--groups',
                                'type': int,
                                'help': 'Number of groups for the time varying models',
                                'default': 4,
                            },
                            {
                                'name': '--p_value',
                                'type': float,
                                'help': 'p_value to use for the likelihood ratio test',
                                'default': 0.05,
                            },
                            {
                                'name': '--skip',
                                'type': comma_list,
                                'help': 'List of models to not test',
                                'default': None,
                            },
                            {
                                'name': '--path',
                                'type': Path,
                                'help': 'Path to output directory',
                            },
                        ],
                    }
                },
                {
                    'allometry': {
                        'help': 'Add allometric scaling',
                        'func': run_allometry,
                        'parents': [args_model_input],
                        'args': [
                            {
                                'name': '--allometric_variable',
                                'type': str,
                                'help': 'Name of the allometric variable',
                                'default': 'WT',
                            },
                            {
                                'name': '--reference_value',
                                'type': float,
                                'help': 'Reference value for the allometric variable',
                                'default': 70.0,
                            },
                            {
                                'name': '--parameters',
                                'type': comma_list,
                                'help': 'List of parameters to apply scaling to',
                                'default': None,
                            },
                            {
                                'name': '--initials',
                                'type': comma_list,
                                'help': 'Initial estimates for the exponents',
                                'default': None,
                            },
                            {
                                'name': '--lower_bounds',
                                'type': comma_list,
                                'help': 'Lower bounds for the exponents',
                                'default': None,
                            },
                            {
                                'name': '--upper_bounds',
                                'type': comma_list,
                                'help': 'Upper bounds for the exponents',
                                'default': None,
                            },
                            {
                                'name': '--non_fixed',
                                'help': 'Should the exponents not be fixed',
                                'action': 'store_true',
                            },
                            {
                                'name': '--path',
                                'type': Path,
                                'help': 'Path to output directory',
                            },
                        ],
                    }
                },
                {
                    'estmethod': {
                        'help': 'Find best estimation method',
                        'func': run_estmethod,
                        'parents': [args_model_input],
                        'args': [
                            {
                                'name': 'algorithm',
                                'type': str,
                                'help': 'Algorithm to use',
                            },
                            {
                                'name': '--methods',
                                'type': str,
                                'default': None,
                                'help': 'List of methods to try, mark group of methods in single '
                                'quote separated by spaces. Supported are: FOCE, FO, IMP, '
                                'IMPMAP, ITS, SAEM, LAPLACE, BAYES, or \'all\'. Default is None.',
                            },
                            {
                                'name': '--solvers',
                                'type': str,
                                'default': None,
                                'help': 'List of solvers to try, mark group of methods in single '
                                'quote separated by spaces. Supported are: CVODES '
                                '(ADVAN14), DGEAR (ADVAN8), DVERK (ADVAN6), IDA '
                                '(ADVAN15), LSODA (ADVAN13) and LSODI (ADVAN9), or \'all\'. '
                                'Default is None.',
                            },
                            {
                                'name': '--path',
                                'type': Path,
                                'help': 'Path to output directory',
                            },
                        ],
                    }
                },
                {
                    'amd': {
                        'help': 'Use Automatic Model Development tool to select PK model',
                        'func': run_amd,
                        'args': [
                            {
                                'name': 'input_path',
                                'type': str,
                                'help': 'Path to output directory',
                            },
                            {
                                'name': '--modeltype',
                                'type': str,
                                'default': 'basic_pk',
                                'help': 'Type of model to build. Currently only "basic_pk"',
                            },
                            {
                                'name': '--administration',
                                'type': str,
                                'default': 'oral',
                                'help': 'Type of model to build. One of "iv", "oral" or "ivoral"',
                            },
                            {
                                'name': '--cl_init',
                                'type': float,
                                'default': 0.01,
                                'help': 'Initial estimate for the population clearance',
                            },
                            {
                                'name': '--vc_init',
                                'type': float,
                                'default': 0.1,
                                'help': 'Initial estimate for the central compartment population volume',
                            },
                            {
                                'name': '--mat_init',
                                'type': float,
                                'default': 0.01,
                                'help': 'Initial estimate for the mean absorption time (not for iv models)',
                            },
                            {
                                'name': '--search_space',
                                'type': str,
                                'default': None,
                                'help': 'MFL for search space for structural model',
                            },
                            {
                                'name': '--lloq_method',
                                'type': str,
                                'default': None,
                                'help': 'Method for how to remove LOQ data. See '
                                '`transform_blq` for list of available methods',
                            },
                            {
                                'name': '--lloq_limit',
                                'type': float,
                                'default': None,
                                'help': 'Lower limit of quantification. If None LLOQ column '
                                'from dataset will be used',
                            },
                            {
                                'name': '--order',
                                'type': comma_list,
                                'default': None,
                                'help': 'Runorder of components',
                            },
                            {
                                'name': '--allometric_variable',
                                'type': str,
                                'default': None,
                                'help': 'Variable to use for allometry',
                            },
                            {
                                'name': '--occasion',
                                'type': str,
                                'default': None,
                                'help': 'Name of occasion column',
                            },
                            {
                                'name': '--path',
                                'type': str,
                                'default': None,
                                'help': 'Path to run AMD in',
                            },
                            {
                                'name': '--resume',
                                'type': bool,
                                'default': True,
                                'help': 'Whether to allow resuming previous run',
                            },
                        ],
                    }
                },
            ],
            'help': 'Run a tool',
            'title': 'Pharmpy commands for running tools',
            'metavar': 'TOOL',
        }
    },
    {
        'data': {
            'subs': [
                {
                    'write': {
                        'help': 'Write dataset',
                        'parents': [args_model_input, args_output],
                        'func': data_write,
                        'description': 'Write a dataset from model as the model sees '
                        'it. For NM-TRAN models this means to filter all IGNORE and '
                        'ACCEPT statements in $DATA and to convert the dataset to '
                        'csv format.',
                    }
                },
                {
                    'print': {
                        'help': 'Print dataset to stdout',
                        'parents': [args_model_or_data_input],
                        'func': data_print,
                        'description': 'Print whole dataset or selected columns from model or csv '
                        'file via a pager to stdout. For NM-TRAN models the dataset will '
                        'be filtered first.',
                        'args': [
                            {
                                'name': '--columns',
                                'metavar': 'COLUMNS',
                                'type': comma_list,
                                'help': 'Select specific columns (default is all)',
                            }
                        ],
                    }
                },
                {
                    'filter': {
                        'help': 'Filter rows of dataset',
                        'description': 'Filter rows of dataset via expressions. '
                        'All rows matching all expressions will be kept.abs '
                        'A new model with the filtered dataset connected is created.',
                        'parents': [args_model_or_data_input, args_output],
                        'func': data_filter,
                        'args': [{'name': 'expressions', 'nargs': argparse.REMAINDER}],
                    }
                },
                {
                    'append': {
                        'help': 'Append column to dataset',
                        'description': 'Append a column to dataset given an assignment expression.'
                        'The expression can contain already present columns of the dataset.',
                        'parents': [args_model_or_data_input, args_output],
                        'func': data_append,
                        'args': [{'name': 'expression'}],
                    }
                },
                {
                    'resample': {
                        'help': 'Resample dataset',
                        'description': 'Bootstrap resample datasets'
                        'Multiple new models and datasets will be created',
                        'parents': [args_model_or_data_input, args_random],
                        'func': data_resample,
                        'args': [
                            {
                                'name': '--group',
                                'metavar': 'COLUMN',
                                'type': str,
                                'default': 'ID',
                                'help': 'Column to use for grouping (default is ID)',
                            },
                            {
                                'name': '--resamples',
                                'metavar': 'NUMBER',
                                'type': int,
                                'default': 1,
                                'help': 'Number of resampled datasets (default 1)',
                            },
                            {
                                'name': '--stratify',
                                'metavar': 'COLUMN',
                                'type': str,
                                'help': 'Column to use for stratification',
                            },
                            {
                                'name': '--replace',
                                'action': 'store_true',
                                'help': 'Sample with replacement (default is without)',
                            },
                            {
                                'name': '--sample_size',
                                'metavar': 'NUMBER',
                                'type': int,
                                'default': None,
                                'help': 'Number of groups to sample for each resample',
                            },
                        ],
                    }
                },
                {
                    'deidentify': {
                        'help': 'Deidentify dataset',
                        'description': 'Deidentify dataset by renumbering the id column and '
                        'changing dates.',
                        'parents': [args_output],
                        'func': data_deidentify,
                        'args': [
                            {
                                'name': 'dataset',
                                'metavar': 'FILE',
                                'type': str,
                                'help': 'A csv file dataset',
                            },
                            {
                                'name': '--idcol',
                                'metavar': 'COLUMN',
                                'type': str,
                                'default': 'ID',
                                'help': 'id column name (default ID)',
                            },
                            {
                                'name': '--datecols',
                                'type': str,
                                'metavar': 'COLUMNS',
                                'default': '',
                                'help': 'Comma separated list of date column names',
                            },
                        ],
                    }
                },
                {
                    'reference': {
                        'help': 'Set reference values in dataset',
                        'description': 'Set values in specific columns to provided reference values',
                        'parents': [args_model_input, args_output],
                        'func': data_reference,
                        'args': [
                            {
                                'name': 'colrefs',
                                'type': str,
                                'nargs': '+',
                                'metavar': 'COLNAME=VALUE',
                                'default': '',
                                'help': 'List of pairs of column names and reference values',
                            },
                        ],
                    }
                },
            ],
            'help': 'Data manipulations',
            'title': 'Pharmpy data commands',
            'metavar': 'ACTION',
        }
    },
    {'info': {'help': 'Show Pharmpy information', 'title': 'Pharmpy information', 'func': info}},
    {
        'model': {
            'subs': [
                {
                    'print': {
                        'help': 'Format and print model overview',
                        'description': 'Print an overview of a model.',
                        'func': model_print,
                        'parents': [args_input],
                        'args': [
                            {
                                'name': '--explicit-odes',
                                'action': 'store_true',
                                'help': 'Print the ODE system explicitly instead of as a '
                                'compartmental graph',
                            }
                        ],
                    }
                },
                {
                    'sample': {
                        'help': 'Sampling of parameter inits using uncertainty given by covariance '
                        'matrix',
                        'description': 'Sample parameter initial estimates using uncertainty given'
                        'by covariance matrix.',
                        'func': model_sample,
                        'parents': [args_model_input, args_random],
                        'args': [
                            {
                                'name': '--samples',
                                'metavar': 'NUMBER',
                                'type': int,
                                'default': 1,
                                'help': 'Number of sampled models',
                            }
                        ],
                    }
                },
                {
                    'update_inits': {
                        'help': 'Update inits using modelfit results.',
                        'description': 'Update inits using modelfit results.',
                        'func': update_inits,
                        'parents': [args_model_input, args_output],
                    }
                },
            ],
            'help': 'Model manipulations',
            'title': 'Pharmpy model commands',
            'metavar': 'ACTION',
        }
    },
    {
        'results': {
            'subs': [
                {
                    'bootstrap': {
                        'help': 'Generate bootstrap results',
                        'description': 'Generate results from a PsN bootstrap run',
                        'func': results_bootstrap,
                        'args': [
                            {
                                'name': 'psn_dir',
                                'metavar': 'PsN directory',
                                'type': Path,
                                'help': 'Path to PsN bootstrap run directory',
                            }
                        ],
                    }
                },
                {
                    'cdd': {
                        'help': 'Generate cdd results',
                        'description': 'Generate results from a PsN cdd run',
                        'func': results_cdd,
                        'args': [
                            {
                                'name': 'psn_dir',
                                'metavar': 'PsN directory',
                                'type': Path,
                                'help': 'Path to PsN cdd run directory',
                            }
                        ],
                    }
                },
                {
                    'frem': {
                        'help': 'Generate FREM results',
                        'description': 'Generate results from a PsN frem run',
                        'func': results_frem,
                        'args': [
                            {
                                'name': 'psn_dir',
                                'metavar': 'PsN directory',
                                'type': Path,
                                'help': 'Path to PsN frem run directory',
                            },
                            {
                                'name': '--method',
                                'choices': ['cov_sampling', 'bipp'],
                                'default': 'cov_sampling',
                                'help': 'Method to use for uncertainty of covariate effects',
                            },
                            {
                                'name': '--force_posdef_covmatrix',
                                'action': 'store_true',
                                'help': 'Should covariance matrix be forced to become positive '
                                'definite',
                            },
                            {
                                'name': '--force_posdef_samples',
                                'type': int,
                                'default': 500,
                                'help': 'Number of sampling tries to do before starting to force '
                                'posdef',
                            },
                        ],
                    }
                },
                {
                    'linearize': {
                        'help': 'Generate linearize results',
                        'description': 'Generate results from a PsN linearize run',
                        'func': results_linearize,
                        'args': [
                            {
                                'name': 'psn_dir',
                                'metavar': 'PsN directory',
                                'type': Path,
                                'help': 'Path to PsN linearize run directory',
                            }
                        ],
                    }
                },
                {
                    'ofv': {
                        'help': 'Extract OFVs from model runs',
                        'description': 'Extract OFVs from one or more model runs',
                        'parents': [args_input],
                        'func': results_ofv,
                    }
                },
                {
                    'print': {
                        'help': 'Print results',
                        'description': 'Print results from PsN run to stdout',
                        'func': results_print,
                        'args': [
                            {
                                'name': 'dir',
                                'metavar': 'file or directory',
                                'type': Path,
                                'help': 'Path to directory containing results.json '
                                'or directly to json results file',
                            }
                        ],
                    }
                },
                {
                    'qa': {
                        'help': 'Generate qa results',
                        'description': 'Generate results from a PsN qa run',
                        'func': results_qa,
                        'args': [
                            {
                                'name': 'psn_dir',
                                'metavar': 'PsN directory',
                                'type': Path,
                                'help': 'Path to PsN qa run directory',
                            }
                        ],
                    }
                },
                {
                    'report': {
                        'help': 'Generate results report',
                        'description': 'Generate results report for PsN run (currently only frem)',
                        'func': results_report,
                        'args': [
                            {
                                'name': 'psn_dir',
                                'metavar': 'PsN directory',
                                'type': Path,
                                'help': 'Path to PsN run directory',
                            }
                        ],
                    }
                },
                {
                    'ruvsearch': {
                        'help': 'Generate ruvsearch results',
                        'description': 'Generate results from a PsN ruvsearch run',
                        'func': results_resmod,
                        'args': [
                            {
                                'name': 'psn_dir',
                                'metavar': 'PsN directory',
                                'type': Path,
                                'help': 'Path to PsN ruvsearch run directory',
                            }
                        ],
                    }
                },
                {
                    'scm': {
                        'help': 'Generate scm results',
                        'description': 'Generate results from a PsN scm run',
                        'func': results_scm,
                        'args': [
                            {
                                'name': 'psn_dir',
                                'metavar': 'PsN directory',
                                'type': Path,
                                'help': 'Path to PsN scm run directory',
                            }
                        ],
                    }
                },
                {
                    'simeval': {
                        'help': 'Generate simeval results',
                        'description': 'Generate results from a PsN simeval run',
                        'func': results_simeval,
                        'args': [
                            {
                                'name': 'psn_dir',
                                'metavar': 'PsN directory',
                                'type': Path,
                                'help': 'Path to PsN simeval run directory',
                            }
                        ],
                    }
                },
                {
                    'summary': {
                        'help': 'Modelfit summary',
                        'description': 'Print a summary of a model estimates to stdout.',
                        'parents': [args_input],
                        'func': results_summary,
                    }
                },
            ],
            'help': 'Result extraction and generation',
            'title': 'Pharmpy result generation commands',
            'metavar': 'ACTION',
        }
    },
]


def generate_parsers(parsers):
    for command in parser_definition:
        ((cmd_name, cmd_dict),) = command.items()
        cmd_parser = parsers.add_parser(
            cmd_name, allow_abbrev=True, help=cmd_dict['help'], formatter_class=formatter
        )
        if 'subs' in cmd_dict:
            subs = cmd_parser.add_subparsers(title=cmd_dict['title'], metavar=cmd_dict['metavar'])
            for sub_command in cmd_dict['subs']:
                ((sub_name, sub_dict),) = sub_command.items()
                args = sub_dict.pop('args', [])
                func = sub_dict.pop('func')

                sub_parser = subs.add_parser(sub_name, **sub_dict, formatter_class=formatter)
                for arg in args:
                    name = arg.pop('name')
                    sub_parser.add_argument(name, **arg)
                sub_parser.set_defaults(func=func)


parser = argparse.ArgumentParser(
    prog='pharmpy',
    description=dedent(
        """
    Welcome to the command line interface of Pharmpy!

    Functionality is split into various subcommands
        - try --help after a COMMAND
        - all keyword arguments can be abbreviated if unique


    """
    ).strip(),
    epilog=dedent(
        """
        Examples:
            # Create 100 bootstrap datasets
            pharmpy data resample pheno_real.mod --resamples=100 --replace

            # prettyprint model
            pharmpy model print pheno_real.mod

            # version/install information
            pharmpy info
    """
    ).strip(),
    formatter_class=formatter,
    allow_abbrev=True,
)
parser.add_argument('--version', action='version', version=pharmpy.__version__)

# subcommand parsers
subparsers = parser.add_subparsers(title='Pharmpy commands', metavar='COMMAND')
generate_parsers(subparsers)


# -- entry point of CLI (pharmpy) ---------------------------------------------------------


def main(args):
    """Entry point of ``pharmpy`` CLI util and ``python3 -m pharmpy`` (via ``__main__.py``)."""
    # parse
    args = parser.parse_args(args)
    # dispatch subcommand
    if 'func' in args:
        args.func(args)
    else:
        parser.print_usage()
