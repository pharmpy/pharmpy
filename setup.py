#!/usr/bin/env python
from __future__ import absolute_import, print_function

import io
import re
from glob import glob
from os.path import basename, dirname, join, splitext
from textwrap import dedent

from setuptools import find_packages, setup


def read(*names, **kwargs):
    """Read file path relative to setup.py (in encoding; default: utf8)"""
    return io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ).read()


def strip_refs(text_str):
    """Strip ref text roles (not valid long_description markup)"""
    return re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', text_str)


def longdesc(text_str):
    """Extract blocks between start-longdesc and end-longdesc directives"""
    pat = r'(?<=^\.\. start-longdesc).*?(?=^\.\. end-longdesc)'
    txt = re.compile(pat, re.MULTILINE | re.DOTALL).findall(text_str)
    txt = [dedent(block).strip() for block in txt]
    return '\n\n'.join(txt)


setup(
    name='pharmpy',
    version='0.2.0',
    license='GNU Lesser General Public License v3 (LGPLv3)',
    description='Pharmacometric model parsing (PsN reimagining)',
    long_description='%s\n%s' % (
        strip_refs(longdesc(read('README.rst'))),
        strip_refs(read('CHANGELOG.rst'))
    ),
    author='Rikard Nordgren',
    author_email='rikard.nordgren@farmbio.uu.se',
    url='https://github.com/yngman/pharmpy',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        # 'Operating System :: Unix',
        'Operating System :: POSIX',
        # 'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        # uncomment if you test on these interpreters:
        # 'Programming Language :: Python :: Implementation :: PyPy',
        # 'Programming Language :: Python :: Implementation :: IronPython',
        # 'Programming Language :: Python :: Implementation :: Jython',
        # 'Programming Language :: Python :: Implementation :: Stackless',
        'Topic :: Utilities',
    ],
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    install_requires=[
        'lark-parser', 'sympy', 'pandas', 'altair', 'toil',
        # eg: 'aspectlib==1.1.1', 'six>=1.7',
    ],
    extras_require={
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    entry_points={
        'console_scripts': [
            'pharmpy           = pharmpy.cli:main',
        ]
    },
)
