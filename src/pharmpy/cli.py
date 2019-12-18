#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
.. highlight:: console

============================
The CLI interface of PharmPy
============================

Yes, CLI is anticipated to remain the main interface! But... maybe with 16 colors?

Examples
--------

Example argument lines (only preceeded by ``pharmpy`` or ``python3 -m pharmpy`` or whatever)::

    # filter data (as model prescribes) and output to file 'output.csv'
    transform --filter --data output.csv

Entrypoints and Subcommands
---------------------------

Standard practice of packaging is to define "entrypoints" in the setuptools ``setup.py``. Currently:

.. code-block:: python

  entry_points={
      'console_scripts': [
          'pharmpy           = pharmpy.cli:main',
          'pharmpy-clone     = pharmpy.cli:cmd_clone',
          'pharmpy-execute   = pharmpy.cli:cmd_execute',
          'pharmpy-help      = pharmpy.cli:cmd_help',
          'pharmpy-print     = pharmpy.cli:cmd_print',
          'pharmpy-sumo      = pharmpy.cli:cmd_sumo',
          'pharmpy-transform = pharmpy.cli:cmd_transform',
          'pharmpy-version   = pharmpy.cli:cmd_version',
      ]
  },

Main entrypoint is ``pharmpy`` (which maps :func:`main`), but each subcommand has a shortcutting
entrypoint (e.g. :func:`sumo`). This is accomplished via invoking:

.. code-block:: python

    CLI(*args_of_sumo, subcommand='sumo')

to skip straight to subcommand parser, enabling linking as if subcommands were different scripts::

    pharmpy --help                 ->  pharmpy --help
    pharmpy-version                ->  pharmpy version
    pharmpy-transform FILE arg..   ->  pharmpy transform FILE arg..

Installing PharmPy Wheel-build should install all of these to PATH. Then, these are clean targets
for post-install hooks to symlink. Etheir to pollute the PATH namespace with traditional PsN-styled
"binscripts", as::

    /usr/bin/execute               ->  /usr/bin/pharmpy-execute

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
from collections import namedtuple, OrderedDict
from textwrap import dedent

import pharmpy
import pharmpy.data.iterators


class CLI:
    """Main CLI interface, based on subcommands (like Git).

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
            Welcome to the commandline interface of PharmPy!
                ~~ a snaky re-imagining of PsN ~~

            Functionality is split into various subcommands
                - try --help after a COMMAND
                - print, transform and version started
                - all keyword arguments can be abbreviated if unique


            """).strip(),
            epilog=dedent("""
                Examples:
                    # apply data filters in model & output to new data file
                    pharmpy -vv transform --filter --data filtered.csv pheno_real.mod

                    # prettyprint APIs of model in pager
                    pharmpy print --all pheno_real.mod

                    # version/install information
                    pharmpy version

                ~~ THIS SOFTWARE IS HIGHLY EXPERIMENTAL AND NOT PRODUCTION READY ~~
            """).strip(),
            formatter_class=argparse.RawTextHelpFormatter,
            allow_abbrev=True,
        )
        self._init_common_args(parser)

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

        # common to: commands with file input
        args_input = argparse.ArgumentParser(add_help=False)
        group_input = args_input.add_argument_group(title='inputs')
        group_input.add_argument('models',
                                 metavar='FILE', type=self.InputModel, nargs='+',
                                 help='input model file')
        self._args_input = args_input

        # common to: commands with file output
        args_output = argparse.ArgumentParser(add_help=False)
        group_output = args_output.add_argument_group(title='outputs')
        group_output.add_argument('-f', '--force',
                                  action='store_true',
                                  help='remove existing destination files (all)')
        group_output.add_argument('-o', '--output_model',
                                  dest='output_model', metavar='file', type=pathlib.Path, nargs=1,
                                  help='output model file')
        self._args_output = args_output

    def _init_commands_tools(self, parsers):
        """Initializes all "tool-like" subcommands."""

        # -- clone -----------------------------------------------------------------------------
        cmd_clone = parsers.add_parser('clone', prog='pharmpy-clone',
                                       parents=[self._args_input, self._args_output],
                                       help='Duplicate model or run',
                                       allow_abbrev=True)
        cmd_clone.add_argument('--ui', '--update_inits', action='store_true',
                               help='set initial estimates (← final estimates)')
        cmd_clone.set_defaults(func=self.cmd_clone)

        # -- execute ---------------------------------------------------------------------------
        cmd_execute = parsers.add_parser('execute', prog='pharmpy-execute',
                                         parents=[self._args_input],
                                         help='Execute model task/workflow',
                                         allow_abbrev=True)
        cmd_execute.set_defaults(func=self.cmd_execute)

        # -- print -----------------------------------------------------------------------------
        cmd_print = parsers.add_parser('print', prog='pharmpy-print',
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

        # -- sumo ------------------------------------------------------------------------------
        cmd_sumo = parsers.add_parser('sumo', prog='pharmpy-sumo',
                                      parents=[self._args_input],
                                      help='Summarize model or run',
                                      allow_abbrev=True)
        cmd_sumo.set_defaults(func=self.cmd_sumo)

        # -- data ------------------------------------------------------------------------------
        cmd_data = parsers.add_parser('data', prog='pharmpy-data',
                                      help='Data manipulations',
                                      allow_abbrev=True)

        cmd_data_subs = cmd_data.add_subparsers(title='PharmPy data commands', metavar='ACTION')

        cmd_data_resample = cmd_data_subs.add_parser('resample', help='Resample dataset', allow_abbrev=True, 
                                      parents=[self._args_input, self._args_output])
        cmd_data_resample.add_argument('--group', metavar='COLUMN', type=str, default='ID',
                                       help='Column to use for grouping (default is ID)')
        cmd_data_resample.add_argument('--samples', metavar='NUMBER', type=int, default=1,
                                       help='Number of resampled datasets (default 1)')
        cmd_data_resample.add_argument('--stratify', metavar='COLUMN', type=str,
                                       help='Column to use for stratification')
        cmd_data_resample.add_argument('--replace', type=bool, default=False,
                                       help='Sample with replacement (default is without)')
        cmd_data_resample.set_defaults(func=self.data_resample)

        # -- transform -------------------------------------------------------------------------
        cmd_transform = parsers.add_parser('transform', prog='pharmpy-transform',
                                           parents=[self._args_input, self._args_output],
                                           help='Common model transformations',
                                           allow_abbrev=True)
        cmd_transform.add_argument('--filter_data',
                                   action='store_true',
                                   help='apply input data filtering')
        cmd_transform.add_argument('--data',
                                   dest='output_data', metavar='PATH', type=pathlib.Path,
                                   help='set model input data (← PATH); also output of new data')
        cmd_transform.set_defaults(func=self.cmd_transform)

    def _init_commands_misc(self, parsers):
        """Initializes miscellanelous other (non-tool) subcommands."""

        # -- help ------------------------------------------------------------------------------
        cmd_help = parsers.add_parser('help', prog='pharmpy-help',
                                      help='PharmPy help central',
                                      allow_abbrev=True)
        cmd_help.set_defaults(func=self.cmd_help)
        cmd_help.add_argument('search_terms', metavar='term', nargs='*', help='search terms')

        # -- version ---------------------------------------------------------------------------
        cmd_version = parsers.add_parser('version', prog='pharmpy-version',
                                         help='Show version information',
                                         allow_abbrev=True)
        cmd_version.set_defaults(func=self.cmd_version)

    # -- subcommand launchers ------------------------------------------------------------------

    def cmd_clone(self, args):
        """Subcommand to clone a model/update initial estimates."""

        self.welcome('clone')
        self.error_exit(exception=NotImplementedError("Command (clone) is not available yet!"))

    def cmd_execute(self, args):
        """Subcommand to execute a model default task/workflow."""

        self.welcome('execute')
        self.error_exit(exception=NotImplementedError("Command (execute) is not available yet!"))

    def cmd_help(self, args):
        """Subcommand for built-in help on PharmPy and such."""

        self.welcome('help')
        self.error_exit(exception=NotImplementedError("Command (help) is not available yet!"))

    def cmd_print(self, args):
        """Subcommand for formatting/printing model components."""

        self.welcome('print')
        lines = []
        for i, model in enumerate(args.models):
            lines += ['[%d/%d] %r' % (i+1, len(args.models), model)]
            dict_ = OrderedDict()
            if args.source or args.all:
                dict_['model.source'] = repr(model.source)
                dict_['model.content'] = model.content.splitlines()
            if args.input or args.all:
                dict_['model.input'] = repr(model.input)
                dict_['model.input.path'] = repr(model.input.path)
                dict_['model.input.filters'] = repr(model.input.filters)
                dict_['model.input.id_column'] = repr(model.input.id_column)
                dict_['model.input.data_frame'] = repr(model.input.data_frame)
            if args.output or args.all:
                dict_['model.output'] = repr(model.output)
            if args.parameters or args.all:
                dict_['model.parameters'] = repr(model.parameters)
                dict_['model.parameters.inits'] = str(model.parameters.inits)
            dict_lines = self.format_keyval_pairs(dict_, sort=False)
            lines += ['\t%s' % line for line in dict_lines]
        if len(lines) > 24:
            pydoc.pager('\n'.join(lines))
        else:
            print('\n'.join(lines))

    def cmd_sumo(self, args):
        """Subcommand to continue PsN ``sumo`` (summarize output)."""

        self.welcome('sumo')
        self.error_exit(exception=NotImplementedError("Command (sumo) is not available yet!"))

    def data_resample(self, args):
        """Subcommand to resample a dataset."""
        for model in args.models:
            df = model.input.read_raw_dataset(parse_columns=[args.group])
            resampler = pharmpy.data.iterators.Resample(df, args.group, resamples=args.samples, stratify=args.stratify, replace=args.replace)
            for resampled_df, _ in resampler:
                model.input.dataset = resampled_df
                model.name = model.input.dataset.name
                model.write()

    def cmd_transform(self, args):
        """Subcommand to transform a model."""

        self.welcome('transform')
        for i, model in enumerate(args.models):
            self.logger.info('[%d/%d] Processing %r', i+1, len(args.models), model)
            write_model = False
            write_data = False

            # -- input/data transforms ---------------------------------------------------------
            if args.filter_data:
                model.input.apply_and_remove_filters()
                write_model = True
                write_data = True

            if args.output_data:
                model.input.path = args.output_data.resolve()
                write_model = True

            # -- input/data file write ---------------------------------------------------------
            if write_data:
                self.OutputFile(model.input.path, args.force)
                model.input.write_dataset()

            # -- model transforms --------------------------------------------------------------
            if args.output_model:
                model.path = args.output_model.resolve()
                write_model = True

            # -- model file write --------------------------------------------------------------
            if write_model:
                self.OutputFile(model.path, args.force)
                model.write()
            elif not write_data:
                self.logger.warning('No-Op (no file write) for %r.', model)

    def cmd_version(self, args):
        """Subcommand to print PharmPy version (and brag a little bit)."""

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
        self.logger.info('Welcome to pharmpy-%s (%s @ %s)' % (subcommand, ver, dir))

    def InputModel(self, path):
        """Returns :class:`~pharmpy.generic.Model` from *path*.

        Raises if not found or is dir, without tracebacks (see :func:`error_exit`).
        """

        path = self.InputFile(path)
        return pharmpy.Model(path)

    def InputFile(self, path):
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

    def OutputFile(self, path, force):
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


# -- entry points of command CLIs (pharmpy-COMMAND) --------------------------------------------


def cmd_clone(*args):
    """Entry point of ``pharmpy-clone`` CLI util, a separable link to ``pharmpy clone``."""
    CLI(*args, subcommand='clone')


def cmd_execute(*args):
    """Entry point of ``pharmpy-execute`` CLI util, a separable link to ``pharmpy execute``."""
    CLI(*args, subcommand='execute')


def cmd_help(*args):
    """Entry point of ``pharmpy-help`` CLI util, a separable link to ``pharmpy help``."""
    CLI(*args, subcommand='help')


def cmd_print(*args):
    """Entry point of ``pharmpy-print`` CLI util, a separable link to ``pharmpy print``."""
    CLI(*args, subcommand='print')


def cmd_sumo(*args):
    """Entry point of ``pharmpy-sumo`` CLI util, a separable link to ``pharmpy sumo``."""
    CLI(*args, subcommand='sumo')


def cmd_transform(*args):
    """Entry point of ``pharmpy-transform`` CLI util, a separable link to ``pharmpy transform``."""
    CLI(*args, subcommand='transform')


def cmd_version(*args):
    """Entry point of ``pharmpy-version`` CLI util, a separable link to ``pharmpy version``."""
    CLI(*args, subcommand='version')
