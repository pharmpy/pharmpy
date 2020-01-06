"""
.. highlight:: console

============================
The CLI interface of PharmPy
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

.. code-block:: python

    CLI(*args_of_sumo, subcommand='sumo')

to skip straight to subcommand parser.

Installing PharmPy Wheel-build should install all of these to PATH. Then, these are clean targets
for post-install hooks to symlink. Either to pollute the PATH namespace with traditional PsN-styled
"binscripts", as::

    /usr/bin/execute               ->  /usr/bin/pharmpy execute

Or, to keep multiple versions when upgrading PharmPy.

On Logging
----------

As a CLI we are at the top, interacting with the user. All of PharmPy will get messages
filtered through here (loggers organize themselves in a hiearchy based on their period-split names).

.. warning::
    Loggers **mustn't** be configurated if imported as library (e.g. set handlers and such),
    since that should bubble up into the lap of whomever is at the top (of the Python
    period-separated namespace).

Definitions
===========
"""

import argparse
import logging
import pathlib
import pydoc
import sys
from collections import OrderedDict, namedtuple
from textwrap import dedent

import pharmpy
import pharmpy.data.iterators
import pharmpy.plugins.utils


class CLI:
    """Main CLI interface, based on subcommands (like git).

    Parses and arguments executes all tasks.

    Args:
        *args: Arguments (overrides all of :attr:`sys.argv`).
        subcommand: If given, skip straight to command (sub)parser.
    """

    def __init__(self, *args, subcommand=None):
        """Initialize and parse command line arguments"""

        # root parser and common arguments
        parser = argparse.ArgumentParser(
            prog='pharmpy',
            description=dedent("""
            Welcome to the command line interface of PharmPy!

            Functionality is split into various subcommands
                - try --help after a COMMAND
                - all keyword arguments can be abbreviated if unique


            """).strip(),
            epilog=dedent("""
                Examples:
                    # Create 100 bootstrap datasets
                    pharmpy data resample pheno_real.mod --resamples=100 --replace

                    # prettyprint model
                    pharmpy print pheno_real.mod

                    # version/install information
                    pharmpy info
            """).strip(),
            formatter_class=argparse.RawTextHelpFormatter,
            allow_abbrev=True,
        )
        self._init_common_args(parser)
        parser.add_argument('--version', action='version', version=pharmpy.__version__)

        # subcommand parsers
        subparsers = parser.add_subparsers(title='PharmPy commands', metavar='COMMAND')
        self._init_commands_tools(subparsers)
        self._init_commands_misc(subparsers)

        # parse
        if args and sys.argv:
            self.logger.debug('Call args overriding, ignoring sys.argv=%r', sys.argv)
        elif not args:
            args = sys.argv[1:]
        if subcommand:
            self.logger.debug('Skipping to subcommand (%r) CLI', subcommand)
            args = [subcommand] + list(args)
        self.logger.debug('%d args to parse: %r' % (len(args), args))
        args = parser.parse_args(args)

        # output/logging setup
        self.set_loglevel(args.loglevel)
        self.logger.debug('Parsed into %d args: %r' % (len(vars(args)), vars(args)))
        if args.hide_traceback:
            self.hide_traceback()

        # dispatch subcommand
        if 'func' in args:
            args.func(args)
        else:
            parser.print_usage()

    # -- additional initializers ---------------------------------------------------------------

    def _init_common_args(self, parser):
        """Initializes common arguments on *parser* and ``self``.

        Groups common to all parsers (activated here)

            .. list-table::
                :widths: 20 80
                :stub-columns: 1

                - - configuration (*idea only*)
                  - To override current (project) configuration file.
                - - logging & verbosity
                  - Sets loglevel and behaviour (e.g. color, logfile copy, etc.) for everything.

        Superclasses for some commands (*parents* in :func:`argparse.ArgumentParser.add_parser`)

            .. list-table::
                :widths: 20 80
                :stub-columns: 1

                - - :attr:`self._args_input`
                  - Common arguments for reading files as input.
                - - :attr:`self._args_output`
                  - Common arguments for outputting files.
        """

        # common to: all
        verbosity = parser.add_argument_group(title='logging & verbosity')
        verbosity.add_argument('--hide_traceback',
                               action='store_true',
                               help="avoid intimidating exception tracebacks")
        loglevel = verbosity.add_mutually_exclusive_group()
        loglevel.add_argument('-v', '--verbose',
                              action='store_const', dest='loglevel',
                              const=logging.INFO, default=logging.WARNING,
                              help='print additional info')
        loglevel.add_argument('-vv', '--debug',
                              action='store_const', dest='loglevel', const=logging.DEBUG,
                              help='show debug messages')

        # common to: commands with model file input
        args_input = argparse.ArgumentParser(add_help=False)
        group_input = args_input.add_argument_group(title='inputs')
        group_input.add_argument('models',
                                 metavar='FILE', type=self.input_model, nargs='+',
                                 help='input model file')
        self._args_input = args_input

        # common to: command with model file or dataset input
        args_model_or_data_input = argparse.ArgumentParser(add_help=False)
        group_model_or_data_input = args_model_or_data_input.add_argument_group(title='data_input')
        group_model_or_data_input.add_argument('model_or_dataset',
                                               metavar='FILE', type=self.input_model_or_dataset,
                                               help='input model or dataset file')
        self._args_model_or_data_input = args_model_or_data_input

        # common to: commands with file output
        args_output = argparse.ArgumentParser(add_help=False)
        group_output = args_output.add_argument_group(title='outputs')
        group_output.add_argument('-f', '--force',
                                  action='store_true',
                                  help='remove existing destination files (all)')
        group_output.add_argument('-o', '--output_file',
                                  dest='output_file', metavar='file', type=pathlib.Path,
                                  help='output file')
        self._args_output = args_output

    def _init_commands_tools(self, parsers):
        """Initializes all "tool-like" subcommands."""

        # -- print -----------------------------------------------------------------------------
        cmd_print = parsers.add_parser('print', prog='pharmpy print',
                                       parents=[self._args_input],
                                       help='Format & print APIs/aspects of models',
                                       allow_abbrev=True)
        group_apis = cmd_print.add_argument_group(title='APIs to print')
        group_apis.add_argument('--all',
                                action='store_true', default=True)
        group_apis.add_argument('--source',
                                action='store_true',
                                help='SourceResource (e.g. source code)')
        group_apis.add_argument('--input',
                                action='store_true',
                                help='ModelInput (e.g. dataset)')
        group_apis.add_argument('--output',
                                action='store_true',
                                help='ModelOutput (e.g. estimates)')
        group_apis.add_argument('--parameters',
                                action='store_true',
                                help='ParameterModel (e.g. inits)')
        cmd_print.set_defaults(func=self.cmd_print)

        # -- data ------------------------------------------------------------------------------
        cmd_data = parsers.add_parser('data',
                                      help='Data manipulations',
                                      allow_abbrev=True)

        cmd_data_subs = cmd_data.add_subparsers(title='PharmPy data commands', metavar='ACTION')

        cmd_data_write = cmd_data_subs.add_parser('write', help='Write dataset', allow_abbrev=True,
                                                  parents=[self._args_model_or_data_input,
                                                           self._args_output])
        cmd_data_write.set_defaults(func=self.data_write)

        cmd_data_resample = cmd_data_subs.add_parser('resample', help='Resample dataset',
                                                     allow_abbrev=True,
                                                     parents=[self._args_model_or_data_input])
        cmd_data_resample.add_argument('--group', metavar='COLUMN', type=str, default='ID',
                                       help='Column to use for grouping (default is ID)')
        cmd_data_resample.add_argument('--resamples', metavar='NUMBER', type=int, default=1,
                                       help='Number of resampled datasets (default 1)')
        cmd_data_resample.add_argument('--stratify', metavar='COLUMN', type=str,
                                       help='Column to use for stratification')
        cmd_data_resample.add_argument('--replace', type=bool, default=False,
                                       help='Sample with replacement (default is without)')
        cmd_data_resample.add_argument('--sample_size', metavar='NUMBER', type=int, default=None,
                                       help='Number of groups to sample for each resample')
        cmd_data_resample.set_defaults(func=self.data_resample)

    def _init_commands_misc(self, parsers):
        """Initializes miscellanelous other (non-tool) subcommands."""

        # -- info ---------------------------------------------------------------------------
        cmd_info = parsers.add_parser('info', prog='pharmpy info',
                                      help='Show pharmpy information',
                                      allow_abbrev=True)
        cmd_info.set_defaults(func=self.cmd_info)

    # -- subcommand launchers ------------------------------------------------------------------

    def cmd_print(self, args):
        """Subcommand for formatting/printing model components."""

        self.welcome('print')
        lines = []
        for i, model in enumerate(args.models):
            lines += ['[%d/%d] %r' % (i+1, len(args.models), model)]
            dict_ = OrderedDict()
            if args.source or args.all:
                dict_['model.source'] = repr(model.source)
            if args.input or args.all:
                dict_['model.input.dataset'] = repr(model.input.dataset)
            if args.output or args.all:
                pass
            if args.parameters or args.all:
                s = ''
                for param in model.parameters:
                    s += repr(param) + '\n'
                dict_['model.parameters'] = s
            dict_lines = self.format_keyval_pairs(dict_, sort=False)
            lines += ['\t%s' % line for line in dict_lines]
        if len(lines) > 24:
            pydoc.pager('\n'.join(lines))
        else:
            print('\n'.join(lines))

    def data_write(self, args):
        """Subcommand to write a dataset."""
        try:
            df = args.model_or_dataset.input.dataset
        except pharmpy.plugins.utils.PluginError:
            df = args.model_or_dataset
        path = args.output_file
        try:
            # If no output_file supplied will use name of df
            df.pharmpy.write_csv(path=path, force=args.force)
        except FileExistsError as e:
            self.error_exit(exception=e)

    def data_resample(self, args):
        """Subcommand to resample a dataset."""
        resampler = pharmpy.data.iterators.Resample(
                args.model_or_dataset, args.group,
                resamples=args.resamples, stratify=args.stratify, replace=args.replace,
                sample_size=args.sample_size)
        for resampled_obj, _ in resampler:
            try:
                try:
                    resampled_obj.write()
                except AttributeError:
                    resampled_obj.pharmpy.write_csv()
            except FileExistsError as e:
                self.error_exit(exception=e)

    def cmd_info(self, args):
        """Subcommand to print PharmPy info (and brag a little bit)."""

        lines = self.format_keyval_pairs(self.install._asdict(), right_just=True)
        print('\n'.join(lines))

    # -- helpers -------------------------------------------------------------------------------

    @property
    def logger(self):
        """Returns current CLI logger.

        Initializes, too, if not already configured (= no handlers).
        """

        logger = logging.getLogger(__file__)
        if logger.hasHandlers():
            return logger

        msg = logging.StreamHandler(sys.stdout)
        err = logging.StreamHandler(sys.stderr)

        msg.addFilter(lambda record: record.levelno <= logging.INFO)
        err.setLevel(logging.WARNING)

        logger.addHandler(msg)
        logger.addHandler(err)
        self.set_loglevel(logging.NOTSET)

        return logger

    @property
    def install(self):
        """:class:`namedtuple` with information about install."""
        Install = namedtuple('VersionInfo', ['version', 'authors', 'directory'])
        return Install(pharmpy.__version__,
                       pharmpy.__authors__,
                       str(pathlib.Path(pharmpy.__file__).parent))

    def welcome(self, subcommand):
        ver, dir = self.install.version, self.install.directory
        self.logger.info('Welcome to pharmpy %s (%s @ %s)' % (subcommand, ver, dir))

    def input_model(self, path):
        """Returns :class:`~pharmpy.model.Model` from *path*.

        Raises if not found or is dir, without tracebacks (see :func:`error_exit`).
        """

        path = self.check_input_path(path)
        return pharmpy.Model(path)

    def input_model_or_dataset(self, path):
        """Returns :class:`~pharmpy.model.Model` or :class:`~pharmpy.data.PharmDataFrame` from *path*
        """
        path = self.check_input_path(path)
        try:
            obj = pharmpy.Model(path)
        except pharmpy.plugins.utils.PluginError:
            obj = pharmpy.data.read_csv(path)
        return obj

    def check_input_path(self, path):
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
            self.error_exit(exception=exc)
        elif path.is_dir():
            exc = IsADirectoryError('Is a directory (not an input file): %r' % str(path))
            self.error_exit(exception=exc)
        else:
            return path

    def check_output_path(self, path, force):
        """Resolves *path* to output file and checks existence.

        Raises if *path* exists (and not *force*), without tracebacks (see :func:`error_exit`).
        """

        try:
            path = pathlib.Path(path)
            path = path.resolve()
        except FileNotFoundError:
            pass

        if path.exists() and not force:
            msg = "Output file exists: %r\nRefusing to overwrite without --force." % str(path)
            self.error_exit(exception=FileExistsError(msg))
        else:
            return path

    def error_exit(self, msg=None, exception=None, status=1):
        """Error exits with code *status*, with optional *msg* logged and *exception* raised.

        Used for non-recoverables in CLI (raises without tracebacks) irrelevant to pharmpy.
        """

        if msg:
            self.logger.critical(str(msg))
        if exception:
            self.hide_traceback(hide=True)
            raise exception
        else:
            sys.exit(status)

    def format_keyval_pairs(self, data_dict, sort=True, right_just=False):
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

    # -- configuration -------------------------------------------------------------------------

    def set_loglevel(self, level):
        """Sets current loglevel (and message formatting)."""

        self.logger.setLevel(level)
        if level <= logging.DEBUG and level != logging.NOTSET:
            fmt = "%(asctime)s.%(msecs)03d %(filename)-25s %(lineno)4d %(levelname)-8s %(message)s"
            datefmt = "%H:%M:%S"
        else:
            fmt = '%(message)s'
            datefmt = "%Y-%m-%d %H:%M:%S"
        for handler in self.logger.handlers:
            handler.setFormatter(logging.Formatter(fmt, datefmt))
        self.logger.debug('Logging config: level=logging.%s, fmt=%r, datefmt=%r',
                          logging.getLevelName(level), fmt, datefmt)

    def hide_traceback(self, hide=True, original_hook=sys.excepthook):
        """Override default exception hook to hide all tracebacks.

        .. note:: Even without traceback, the exception and message is shown.
        """

        def exc_only_hook(exception_type, exception, traceback):
            self.logger.critical('%s: %s', exception_type.__name__, exception)

        if hide:
            self.logger.debug('Overriding sys.excepthook to hide tracebacks: %r', sys.excepthook)
            sys.excepthook = exc_only_hook
        else:
            self.logger.debug('Restoring sys.excepthook to reset tracebacks: %r', original_hook)
            sys.excepthook = original_hook


# -- entry point of main CLI (pharmpy) ---------------------------------------------------------


def main(*args):
    """Entry point of ``pharmpy`` CLI util and ``python3 -m pharmpy`` (via ``__main__.py``)."""
    CLI(*args)
