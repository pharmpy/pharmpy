#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import argparse
import logging
from pathlib import Path
import sys
from textwrap import dedent

from . import __author__, __pkgname__, __version__
from .psn import detectAPI, getAPI, Model


def cli_args(args):
    """Initialize and parse command line arguments"""
    logger = logging.getLogger()

    # CLI meta data
    pre = dedent('''
        PysN CLI interface
    ''').strip()
    post = dedent('''
        version : %s
        authors : %s
                  %s
    ''' % (__version__, __authors__[0], __authors__[1])
    ).strip()
    path = os.path.abspath(__file__)
    prog = os.path.splitext(os.path.basename(path))[0]
    parser = argparse.ArgumentParser(
        prog=prog,
        description=pre,
        epilog=post,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # CLI arguments
    parser.add_argument(
        'path', metavar='PATH', nargs='1', action='store',
        help='try to load model with PysN',
    )
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        '-v', '--verbose',
        action='store_const', dest='loglevel', const=logging.INFO,
        default=logging.WARNING,
        help='print additional info',
    )
    verbosity_group.add_argument(
        '--debug',
        action='store_const', dest='loglevel', const=logging.DEBUG,
        help='show debug messages',
    )

    # argument parse and return
    logger.debug('parsing CLI args, in: %s' % (str(args),))
    args = parser.parse_args(args=args)
    logger.debug('parsed CLI args, out: %s' % (str(args),))
    return args


def cli_logging(stream_level):
    h1_formatter = logging.Formatter('%(message)s')
    h2_formatter = logging.Formatter(
        '(%(levelname)s from %(name)s:%(lineno)d) %(funcName)s: %(message)s'
    )

    # __pkgname__ is "hack" (imported from __init__.py):
    #     guarantees CLI logger gets top name in logger path
    logger = logging.getLogger(__pkgname__)
    logger.setLevel(stream_level)

    # msg >= logging.WARNING --> STDERR
    # msg <= logging.INFO --> STDOUT
    h1 = logging.StreamHandler(sys.stdout)
    h2 = logging.StreamHandler(sys.stderr)
    h1.setLevel(logging.DEBUG)
    h2.setLevel(logging.WARNING)
    h1.addFilter(lambda record: record.levelno <= logging.INFO)
    h1.setFormatter(h1_formatter)
    h2.setFormatter(h2_formatter)
    logger.addHandler(h1)
    logger.addHandler(h2)

    return logger


def main(args=None):
    args = cli_args(args=args)
    logger = cli_logging(args.loglevel)

    path = Path(args.path).resolve()
    print('Trying pysn.Model(%s)...' % (path,))
    model = pysn.Model(path)
    print(model)
