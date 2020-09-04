"""
.. highlight:: console

============================
The CLI interface of Pharmpy
============================

Examples
--------

Example argument lines (only preceeded by ``pharmpy`` or ``python3 -m pharmpy``)::

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
filtered through here (loggers organize themselves in a hiearchy based on their period-split names).

.. warning::
    Loggers **mustn't** be configurated if imported as library (e.g. set handlers and such),
    since that should bubble up into the lap of whomever is at the top (of the Python
    period-separated namespace).

Definitions
===========
"""

import argparse
import importlib
import pathlib
import pydoc
import re
import sys
import types
from collections import OrderedDict, namedtuple
from textwrap import dedent


class LazyLoader(types.ModuleType):
    """Class that masquerades as a module and lazily loads it when accessed the first time

       The code for the class is taken from TensorFlow and is under the Apache 2.0 license

       This is needed for the cli to be able to do things as version and help quicker than
       if all modules were imported.
    """
    def __init__(self, local_name, parent_module_globals, name):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals

        super(LazyLoader, self).__init__(name)

    def _load(self):
        # Import the target module and insert it into the parent's namespace
        module = importlib.import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module

        # Update this object's dict so that if someone keeps a reference to the
        #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
        #   that fail).
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)


pharmpy = LazyLoader('pharmpy', globals(), 'pharmpy')
np = LazyLoader('numpy', globals(), 'numpy')
iterators = LazyLoader('iterators', globals(), 'pharmpy.data.iterators')
plugin_utils = LazyLoader('plugin_utils', globals(), 'pharmpy.plugins.utils')

formatter = argparse.ArgumentDefaultsHelpFormatter


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
        data_dict = OrderedDict(sorted(data_dict.items()))
    if right_just:
        line_format = '  %%%ds\t%%s' % key_width
    else:
        line_format = '%%-%ds\t%%s' % key_width

    lines = []
    for key, values in data_dict.items():
        if isinstance(values, str):
            values = values.splitlines()
        keys = [key] + ['']*(len(values) - 1)
        for k, v in zip(keys, values):
            lines += [line_format % (k, v)]

    return lines


def run_execute(args):
    import pharmpy.methods.modelfit as modelfit
    modelfit.run(args.models)


def data_write(args):
    """Subcommand to write a dataset."""
    try:
        df = args.model_or_dataset.dataset
    except plugin_utils.PluginError:
        df = args.model_or_dataset
    path = args.output_file
    try:
        # If no output_file supplied will use name of df
        path = df.pharmpy.write_csv(path=path, force=args.force)
        print(f'Dataset written to {path}')
    except OSError as e:
        error(e)


def data_print(args):
    """Subcommand to print a dataset to stdout via pager
    """
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
    except plugin_utils.PluginError:
        df = args.model_or_dataset
    expression = ' '.join(args.expressions)
    try:
        df.query(expression, inplace=True)
    except SyntaxError:
        error(SyntaxError(f'Invalid syntax of query: "{expression}"'))
    except BaseException as e:
        error(e)
    write_model_or_dataset(args.model_or_dataset, df, args.output_file, args.force)


def data_append(args):
    """Subcommand to append a column to a dataset"""
    try:
        df = args.model_or_dataset.dataset
    except plugin_utils.PluginError:
        df = args.model_or_dataset
    try:
        df.eval(args.expression, inplace=True)
    except SyntaxError:
        error(SyntaxError(f'Invalid syntax of expression: "{args.expression}"'))
    write_model_or_dataset(args.model_or_dataset, df, args.output_file, args.force)


def write_model_or_dataset(model_or_dataset, new_df, path, force):
    """Write model or dataset to output_path or using default names
    """
    if hasattr(model_or_dataset, 'pharmpy'):
        # Is a dataset
        try:
            # If no output_file supplied will use name of df
            path = new_df.pharmpy.write_csv(path=path, force=force)
            print(f'Dataset written to {path}')
        except FileExistsError as e:
            error(exception=FileExistsError(f'{e.args[0]} Use -f  or --force to '
                                            'force an overwrite'))
    else:
        # Is a model
        model = model_or_dataset
        if new_df is not None:
            model.dataset = new_df
        try:
            if path:
                if not path.is_dir():
                    bump_model_number(model, path)
                model.write(path=path, force=force)
            else:
                bump_model_number(model)
                model.write(force=force)
        except FileExistsError as e:
            error(FileExistsError(f'{e.args[0]} Use -f or --force to '
                                  'force an overwrite'))


def bump_model_number(model, path='.'):
    """ If model name ends in a number increase to next available file
        else do nothing.
        FIXME: Could be moved to model class. Need the name of the model and the source object
    """
    path = pathlib.Path(path)
    name = model.name
    m = re.search(r'(.*?)(\d+)$', name)
    if m:
        stem = m.group(1)
        n = int(m.group(2))
        while True:
            n += 1
            new_name = f'{stem}{n}'
            new_path = path / new_name
            if not new_path.exists():
                break
        model.name = new_name


def data_resample(args):
    """Subcommand to resample a dataset."""
    resampler = iterators.Resample(
            args.model_or_dataset, args.group,
            resamples=args.resamples, stratify=args.stratify, replace=args.replace,
            sample_size=args.sample_size,
            name_pattern=f'{args.model_or_dataset.name}_resample_{{}}')
    for resampled_obj, _ in resampler:
        try:
            try:
                resampled_obj.write()
            except AttributeError:
                resampled_obj.pharmpy.write_csv()
        except OSError as e:
            error(e)


def data_anonymize(args):
    """Subcommand to anonymize a dataset
       anonymization is really a special resample case where the specified group
       (usually the ID) is given new random unique values and reordering.
       Will be default try to overwrite original file (but won't unless force)
    """
    path = check_input_path(args.dataset)
    dataset = pharmpy.data.read_csv(path, raw=True, parse_columns=[args.group])
    resampler = pharmpy.data.iterators.Resample(dataset, args.group)
    df, _ = next(resampler)
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = path
    write_model_or_dataset(df, df, output_file, args.force)


def info(args):
    """Subcommand to print Pharmpy info (and brag a little bit)."""

    Install = namedtuple('VersionInfo', ['version', 'authors', 'directory'])
    inst = Install(pharmpy.__version__,
                   'A list of authors can be found in the AUTHORS.rst',
                   str(pathlib.Path(pharmpy.__file__).parent))

    lines = format_keyval_pairs(inst._asdict(), right_just=True)
    print('\n'.join(lines))


def model_print(args):
    """Subcommand for formatting/printing model components."""
    lines = []
    for i, model in enumerate(args.models):
        if args.explicit_odes:
            from pharmpy.modeling import explicit_odes
            explicit_odes(model)
        lines += ['[%d/%d] %r' % (i+1, len(args.models), model.name)]
        dict_ = OrderedDict()
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
    model = args.model
    from pharmpy.parameter_sampling import sample_from_covariance_matrix
    samples = sample_from_covariance_matrix(model, n=args.samples)
    for row, params in samples.iterrows():
        model.parameters = params
        model.name = f'sample_{row + 1}'
        model.write()


def add_covariate_effect(args):
    """Subcommand to add covariate effect to model."""
    from pharmpy.modeling import add_covariate_effect

    model = args.model
    add_covariate_effect(model, args.param, args.covariate, args.effect)

    write_model_or_dataset(model, model.dataset, path=args.output_file, force=False)


def model_explicit_odes(args):
    from pharmpy.modeling import explicit_odes
    model = args.model
    explicit_odes(model)
    write_model_or_dataset(model, None, path=args.output_file, force=args.force)


def model_absorption(args):
    from pharmpy.modeling import absorption
    model = args.model
    absorption(model, order=args.order, rate=args.rate)
    write_model_or_dataset(model, None, path=args.output_file, force=args.force)


def results_bootstrap(args):
    """Subcommand to generate bootstrap results"""
    if len(args.models) < 3:
        error(ValueError('Need at least the original model and 2'
                         'other models'))
    from pharmpy.methods.bootstrap.results import BootstrapResults
    res = BootstrapResults(original_model=args.base,
                           bootstrap_models=args.models)
    print(res)


def results_frem(args):
    """Generate frem results"""
    from pharmpy.methods.frem.results import psn_frem_results
    if not args.psn_dir.is_dir():
        error(FileNotFoundError(str(args.psn_dir)))
    res = psn_frem_results(args.psn_dir, method=args.method,
                           force_posdef_covmatrix=args.force_posdef_covmatrix,
                           force_posdef_samples=args.force_posdef_samples)
    res.add_plots()
    res.to_json(path=args.psn_dir / 'results.json')
    res.to_csv(path=args.psn_dir / 'results.csv')


def results_ofv(args):
    """Subcommand to extract final ofv from multiple results"""
    ofvs = []
    for model in args.models:
        try:
            ofv = str(model.modelfit_results.ofv)
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
    from pharmpy.results import read_results
    res = read_results(path)
    print(res)


def results_report(args):
    """Subcommand to generate reports"""
    results_path = args.psn_dir / 'results.json'
    if not results_path.is_file():
        error(FileNotFoundError(results_path))
    from pharmpy.results import read_results
    res = read_results(results_path)
    res.create_report(args.psn_dir)


def results_summary(args):
    """Subcommand to output summary of modelfit"""
    for model in args.models:
        print(model.name)
        model.modelfit_results.reparameterize([
            pharmpy.random_variables.NormalParametrizationSd,
            pharmpy.random_variables.MultivariateNormalParametrizationSdCorr
        ])
        print(model.modelfit_results.parameter_summary())
        print()


def check_input_path(path):
    """Resolves path to input file and checks existence.

    Raises if not found or is dir, without tracebacks (see :func:`error_exit`).
    """
    try:
        path = pathlib.Path(path)
        path = path.resolve()
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
    return pharmpy.Model(path)


def input_model_or_dataset(path):
    """Returns :class:`~pharmpy.model.Model` or :class:`~pharmpy.data.PharmDataFrame` from *path*
    """
    path = check_input_path(path)
    try:
        obj = pharmpy.Model(path)
    except pharmpy.plugins.utils.PluginError:
        obj = pharmpy.data.read_csv(path)
    return obj


def comma_list(self, s):
    if s == '':
        return []
    return s.split(',')


def random_seed(seed):
    try:
        seed = int(seed)
        np.random.seed(seed)
    except ValueError as e:
        # Cannot reraise the ValueError as it would be caught by argparse
        error(argparse.ArgumentTypeError(str(e)))
    return seed


# for commands taking multiple model files as input
args_input = argparse.ArgumentParser(add_help=False)
group_input = args_input.add_argument_group(title='inputs')
group_input.add_argument('models', metavar='FILE', type=input_model, nargs='+',
                         help='input model files')

# for commands taking one model file as input
args_model_input = argparse.ArgumentParser(add_help=False)
group_model_input = args_model_input.add_argument_group(title='model')
group_model_input.add_argument('model', metavar='FILE', type=input_model,
                               help='input model file')

# for commands with model file or dataset input
args_model_or_data_input = argparse.ArgumentParser(add_help=False)
group_model_or_data_input = args_model_or_data_input.add_argument_group(title='data_input')
group_model_or_data_input.add_argument('model_or_dataset',
                                       metavar='FILE', type=input_model_or_dataset,
                                       help='input model or dataset file')

# for commands involving randomization
args_random = argparse.ArgumentParser(add_help=False)
group_random = args_random.add_argument_group(title='random seed')
group_random.add_argument('--seed', type=random_seed, metavar='INTEGER',
                          help='Provide a random seed. The seed must be an integer '
                          'between 0 and 2^32 - 1')

# for commands with file output
args_output = argparse.ArgumentParser(add_help=False)
group_output = args_output.add_argument_group(title='outputs')
group_output.add_argument('-f', '--force',
                          action='store_true',
                          help='remove existing destination files (all)')
group_output.add_argument('-o', '--output_file',
                          dest='output_file', metavar='file', type=pathlib.Path,
                          help='output file')

parser_definition = [
    {'run': {'subs': [
        {'execute': {'help': 'Execute one or more model', 'func': run_execute,
                     'parents': [args_input]}},
    ], 'help': 'Run a method',
       'title': 'Pharmpy commands for running methods',
       'metavar': 'METHOD',
    }},
    {'data': {'subs': [
        {'write': {'help': 'Write dataset', 'parents': [args_model_or_data_input, args_output],
                   'func': data_write,
                   'description': 'Write a dataset from model or csv file as the model sees it. '
                                  'For NM-TRAN models this means to filter all IGNORE and '
                                  'ACCEPT statements in $DATA and to convert the dataset to '
                                  'csv format.'}},
        {'print': {'help': 'Print dataset to stdout', 'parents': [args_model_or_data_input],
                   'func': data_print,
                   'description': 'Print whole dataset or selected columns from model or csv file '
                                  'via a pager to stdout. For NM-TRAN models the dataset will '
                                  'be filtered first.',
                   'args': [
                {'name': '--columns', 'metavar': 'COLUMNS', 'type': comma_list,
                 'help': 'Select specific columns (default is all)'}]}},
        {'filter': {
            'help': 'Filter rows of dataset',
            'description': 'Filter rows of dataset via expressions. '
                           'All rows matching all expressions will be kept.abs '
                           'A new model with the filtered dataset connected is created.',
            'parents': [args_model_or_data_input, args_output],
            'func': data_filter,
            'args': [{'name': 'expressions', 'nargs': argparse.REMAINDER}]}},
        {'append': {
            'help': 'Append column to dataset',
            'description': 'Append a column to dataset given an assignment expression.'
                           'The expression can contain already present columns of the dataset.',
            'parents': [args_model_or_data_input, args_output],
            'func': data_append,
            'args': [{'name': 'expression'}]}},
        {'resample': {
            'help': 'Resample dataset',
            'description': 'Bootstrap resample datasets'
                           'Multiple new models and datasets will be created',
            'parents': [args_model_or_data_input, args_random], 'func': data_resample,
            'args': [{'name': '--group',
                      'metavar': 'COLUMN',
                      'type': str,
                      'default': 'ID',
                      'help': 'Column to use for grouping (default is ID)'},
                     {'name': '--resamples',
                      'metavar': 'NUMBER',
                      'type': int,
                      'default': 1,
                      'help': 'Number of resampled datasets (default 1)'},
                     {'name': '--stratify',
                      'metavar': 'COLUMN',
                      'type': str,
                      'help': 'Column to use for stratification'},
                     {'name': '--replace',
                      'action': 'store_true',
                      'help': 'Sample with replacement (default is without)'},
                     {'name': '--sample_size',
                      'metavar': 'NUMBER',
                      'type': int,
                      'default': None,
                      'help': 'Number of groups to sample for each resample'}]}},
        {'anonymize': {
            'help': 'Anonymize dataset by renumbering a specific data column',
            'description': 'Anonymize dataset by renumbering a specific data column',
            'parents': [args_output], 'func': data_anonymize,
            'args': [{'name': 'dataset',
                      'metavar': 'FILE',
                      'type': str,
                      'help': 'A csv file dataset'},
                     {'name': '--group',
                      'metavar': 'COLUMN',
                      'type': str,
                      'default': 'ID',
                      'help': 'column to anonymize (default ID)'}]}},
        ], 'help': 'Data manipulations', 'title': 'Pharmpy data commands', 'metavar': 'ACTION'}},
    {'info': {'help': 'Show Pharmpy information', 'title': 'Pharmpy information', 'func': info}},
    {'model': {'subs': [
        {'print': {
            'help': 'Format and print model overview',
            'description': 'Print an overview of a model.',
            'func': model_print,
            'parents': [args_input],
            'args': [{'name': '--explicit-odes',
                      'action': 'store_true',
                      'help': 'Print the ODE system explicitly instead of as a '
                              'compartmental graph'}]}},
        {'sample': {
            'help': 'Sampling of parameter inits using uncertainty given by covariance '
                    'matrix',
            'description': 'Sample parameter initial estimates using uncertainty given'
                           'by covariance matrix.',
            'func': model_sample, 'parents': [args_model_input, args_random],
            'args': [{'name': '--samples',
                      'metavar': 'NUMBER',
                      'type': int,
                      'default': 1,
                      'help': 'Number of sampled models'}]}},
        {'add_cov_effect': {
            'help': 'Adds covariate effect',
            'description': 'Add covariate effect to a model parameter',
            'func': add_covariate_effect,
            'parents': [args_model_input, args_output],
            'args': [{'name': 'param',
                      'type': str,
                      'help': 'Individual parameter'},
                     {'name': 'covariate',
                      'type': str,
                      'help': 'Covariate'},
                     {'name': 'effect',
                      'type': str,
                      'help': 'Type of covariate effect'},
                     ]}},
        {'explicit_odes': {
            'help': 'Make ODEs explicit',
            'description': 'Convert from compartmental or implicit description of '
                           'ODE-system to explicit. I.e. create a $DES',
            'func': model_explicit_odes,
            'parents': [args_model_input, args_output],
                       }},
        {'absorption': {
            'help': 'Set absorption rate for a PK model',
            'description': 'Change absorption rate of a PK model to either 0th or 1th order',
            'func': model_absorption,
            'parents': [args_model_input, args_output],
            'args': [{'name': 'order',
                      'choices': [0, 1],
                      'type': int,
                      'help': 'Order of absorption'},
                     {'name': '--rate',
                      'type': str,
                      'help': 'Symbol for absorption rate'},
                     ]}},
    ], 'help': 'Model manipulations',
       'title': 'Pharmpy model commands',
       'metavar': 'ACTION',
    }},
    {'results': {'subs': [
        {'bootstrap': {
            'help': 'Generate bootstrap results',
            'description': 'Generate results from a PsN bootstrap run',
            'func': results_bootstrap,
            'parents': [args_input], 'args': [
                   {'name': '--base', 'metavar': 'FILE', 'type': input_model,
                    'help': 'Base model'}]}},
        {'frem': {
            'help': 'Generate FREM results',
            'description': 'Generate results from a PsN frem run',
            'func': results_frem, 'args': [
                {'name': 'psn_dir', 'metavar': 'PsN directory', 'type': pathlib.Path,
                 'help': 'Path to PsN frem run directory'},
                {'name': '--method', 'choices': ['cov_sampling', 'bipp'], 'default': 'cov_sampling',
                 'help': 'Method to use for uncertainty of covariate effects'},
                {'name': '--force_posdef_covmatrix', 'action': 'store_true',
                 'help': 'Should covariance matrix be forced to become positive definite'},
                {'name': '--force_posdef_samples', 'type': int, 'default': 500,
                 'help': 'Number of sampling tries to do before starting to force posdef'}]}},
        {'ofv': {
            'help': 'Extract OFVs from model runs',
            'description': 'Extract OFVs from one or more model runs',
            'parents': [args_input],
            'func': results_ofv}},
        {'print': {
            'help': 'Print results',
            'description': 'Print results from PsN run to stdout',
            'func': results_print,
            'args': [
                {'name': 'dir', 'metavar': 'file or directory', 'type': pathlib.Path,
                 'help': 'Path to directory containing results.json '
                         'or directly to json results file'}]}},
        {'report': {
            'help': 'Generate results report',
            'description': 'Generate results report for PsN run (currently only frem)',
            'func': results_report,
            'args': [
                {'name': 'psn_dir', 'metavar': 'PsN directory', 'type': pathlib.Path,
                 'help': 'Path to PsN run directory'}]}},
        {'summary': {
            'help': 'Modelfit summary',
            'description': 'Print a summary of a model estimates to stdout.',
            'parents': [args_input],
            'func': results_summary}}],
        'help': 'Result extraction and generation',
        'title': 'Pharmpy result generation commands', 'metavar': 'ACTION'}}]


def generate_parsers(parsers):
    for command in parser_definition:
        (cmd_name, cmd_dict), = command.items()
        cmd_parser = parsers.add_parser(cmd_name, allow_abbrev=True, help=cmd_dict['help'],
                                        formatter_class=formatter)
        if 'subs' in cmd_dict:
            subs = cmd_parser.add_subparsers(title=cmd_dict['title'], metavar=cmd_dict['metavar'])
            for sub_command in cmd_dict['subs']:
                (sub_name, sub_dict), = sub_command.items()
                args = sub_dict.pop('args', [])
                func = sub_dict.pop('func')

                sub_parser = subs.add_parser(sub_name, **sub_dict, formatter_class=formatter)
                for arg in args:
                    name = arg.pop('name')
                    sub_parser.add_argument(name, **arg)
                sub_parser.set_defaults(func=func)


parser = argparse.ArgumentParser(
    prog='pharmpy',
    description=dedent("""
    Welcome to the command line interface of Pharmpy!

    Functionality is split into various subcommands
        - try --help after a COMMAND
        - all keyword arguments can be abbreviated if unique


    """).strip(),
    epilog=dedent("""
        Examples:
            # Create 100 bootstrap datasets
            pharmpy data resample pheno_real.mod --resamples=100 --replace

            # prettyprint model
            pharmpy model print pheno_real.mod

            # version/install information
            pharmpy info
    """).strip(),
    formatter_class=formatter,
    allow_abbrev=True,
)
parser.add_argument('--version', action='version', version='0.5.0')

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
