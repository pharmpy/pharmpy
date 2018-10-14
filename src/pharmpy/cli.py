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
          'pharmpy-clone     = pharmpy.cli:clone',
          'pharmpy-execute   = pharmpy.cli:execute',
          'pharmpy-help      = pharmpy.cli:help',
          'pharmpy-sumo      = pharmpy.cli:sumo',
          'pharmpy-transform = pharmpy.cli:transform',
          'pharmpy-version   = pharmpy.cli:version',
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
import sys
from collections import namedtuple, OrderedDict
from textwrap import dedent

import pharmpy


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
            description='CLI interface of PharmPy',
            epilog=dedent("""
                Examples:
                    pharmpy -vv transform --filter --data filtered.csv pheno_real.mod

                THIS SOFTWARE IS HIGHLY EXPERIMENTAL AND NOT PRODUCTION READY.
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
                  - Common arguments for commands reading files as input.
                - - :attr:`self._args_output`
                  - Common arguments for commands outputting files.
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
        group_input = args_input.add_argument_group(title='file input')
        group_input.add_argument('models',
                                 metavar='FILE', type=self.InputModel, nargs='+',
                                 help='input model file')
        self._args_input = args_input

        # common to: commands with file output
        args_output = argparse.ArgumentParser(add_help=False)
        group_output = args_output.add_argument_group(title='file output')
        group_output.add_argument('-f', '--force',
                                  action='store_true',
                                  help='remove existing destination files (all)')
        group_output.add_argument('-o', '--output',
                                  dest='output_model', metavar='file', type=pathlib.Path, nargs=1,
                                  help='output model file')
        self._args_output = args_output

    def _init_commands_tools(self, parsers):
        """Initializes all "tool-like" subcommands."""

        # -- clone -----------------------------------------------------------------------------
        clone = parsers.add_parser('clone', prog='pharmpy-clone',
                                   parents=[self._args_input, self._args_output],
                                   help='Duplicate model or run')
        clone.add_argument('--ui', '--update_inits', action='store_true',
                           help='set initial estimates (← final estimates)')
        clone.set_defaults(func=self.clone)

        # -- execute ---------------------------------------------------------------------------
        execute = parsers.add_parser('execute', prog='pharmpy-execute',
                                     parents=[self._args_input],
                                     help='Execute model task/workflow')
        execute.set_defaults(func=self.execute)

        # -- sumo ------------------------------------------------------------------------------
        sumo = parsers.add_parser('sumo', prog='pharmpy-sumo',
                                  parents=[self._args_input],
                                  help='Summarize model or run')
        sumo.set_defaults(func=self.sumo)

        # -- transform -------------------------------------------------------------------------
        transform = parsers.add_parser('transform', prog='pharmpy-transform',
                                       parents=[self._args_input, self._args_output],
                                       help='Common model transformations')
        transform.add_argument('--filter_data',
                               action='store_true',
                               help='apply input data filtering')
        transform.add_argument('--data',
                               dest='output_data', metavar='PATH', type=pathlib.Path, nargs=1,
                               help='set model input data (← PATH); also output of new data')
        transform.set_defaults(func=self.transform)

    def _init_commands_misc(self, parsers):
        """Initializes miscellanelous other (non-tool) subcommands."""

        # -- help ------------------------------------------------------------------------------
        help = parsers.add_parser('help', prog='pharmpy-help',
                                  help='PharmPy help central')
        help.set_defaults(func=self.help)
        help.add_argument('search_terms', metavar='term', nargs='*', help='search terms')

        # -- version ---------------------------------------------------------------------------
        version = parsers.add_parser('version', prog='pharmpy-version',
                                     help='Show version information')
        version.set_defaults(func=self.version)

    # -- subcommand launchers ------------------------------------------------------------------

    def clone(self, args):
        """Subcommand to clone a model/update initial estimates."""

        self.welcome('clone')
        raise NotImplementedError("Subcommand not available yet. Working on it!")

    def execute(self, args):
        """Subcommand to execute a model default task/workflow."""

        self.welcome('execute')
        raise NotImplementedError("Subcommand not available yet. Working on it!")

    def help(self, args):
        """Subcommand for built-in help on PharmPy and such."""

        self.welcome('help')
        raise NotImplementedError("Subcommand not available yet. Working on it!")

    def sumo(self, args):
        """Subcommand to continue PsN ``sumo`` (summarize output)."""

        self.welcome('sumo')
        raise NotImplementedError("Subcommand not available yet. Working on it!")

    def transform(self, args):
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
                self.OutputFile(model.input.path)
                model.input.write_dataset()

            # -- model transforms --------------------------------------------------------------
            if args.output_model:
                model.path = args.output_model.resolve()
                write_model = True

            # -- model file write --------------------------------------------------------------
            if write_model:
                self.OutputFile(model.path)
                model.write()
            elif not write_data:
                self.logger.warning('No-Op (no file write) for %r.', model)

    def version(self, args):
        """Subcommand to print PharmPy version (and brag a little bit)."""

        install_tuple = self.install
        field_width = max(len(field) for field in install_tuple._fields)
        line_format = '  %%%ds\t%%s' % (field_width,)
        install_dict = OrderedDict(sorted(install_tuple._asdict().items()))
        for key, values in install_dict.items():
            if isinstance(values, str):
                values = [values]
            keys = [key] + ['']*len(values)
            for k, v in zip(keys, values):
                print(line_format % (k, v))

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
        self.logger.info('Welcome to pharmpy-%s (%s @ %s)' % (subcommand,
                                                              self.install.ver, self.install.dir))

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


def clone(*args):
    """Entry point of ``pharmpy-clone`` CLI util, a separable link to ``pharmpy clone``."""
    CLI(*args, subcommand='clone')


def execute(*args):
    """Entry point of ``pharmpy-execute`` CLI util, a separable link to ``pharmpy execute``."""
    CLI(*args, subcommand='execute')


def help(*args):
    """Entry point of ``pharmpy-help`` CLI util, a separable link to ``pharmpy help``."""
    CLI(*args, subcommand='help')


def sumo(*args):
    """Entry point of ``pharmpy-sumo`` CLI util, a separable link to ``pharmpy sumo``."""
    CLI(*args, subcommand='sumo')


def transform(*args):
    """Entry point of ``pharmpy-transform`` CLI util, a separable link to ``pharmpy transform``."""
    CLI(*args, subcommand='transform')


def version(*args):
    """Entry point of ``pharmpy-version`` CLI util, a separable link to ``pharmpy version``."""
    CLI(*args, subcommand='version')
