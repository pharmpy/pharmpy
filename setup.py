import io
import re
from glob import glob
from os.path import abspath, basename, dirname, join, splitext
from textwrap import dedent

from setuptools import find_packages, setup


def read(*names, **kwargs):
    """Read file path relative to setup.py (in encoding; default: utf8)"""
    return io.open(
        join(dirname(abspath(__file__)), *names), encoding=kwargs.get('encoding', 'utf8')
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
    name='pharmpy-core',
    version='0.98.0',
    license='GNU Lesser General Public License v3 (LGPLv3)',
    description='Pharmacometric modeling',
    long_description='%s\n\n%s'
    % (strip_refs(longdesc(read('README.rst'))), strip_refs(read('CHANGELOG.rst'))),
    author='Rikard Nordgren',
    author_email='rikard.nordgren@farmaci.uu.se',
    url='https://pharmpy.github.io',
    project_urls={
        "Bug Tracker": 'https://github.com/pharmpy/pharmpy/issues',
        "Source Code": 'https://github.com/pharmpy/pharmpy',
    },
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Operating System :: Unix',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering',
    ],
    keywords=[
        'pharmacometrics',
    ],
    install_requires=[
        'lark>=1.1.4',
        'sympy>=1.9',
        'symengine>=0.9.2',
        'pandas>=1.4',
        'numexpr',
        'altair>=4.2.0',
        'jsonschema',
        'sphinx',
        'csscompressor',
        'beautifulsoup4',
        'lxml',
        'numpy>=1.17',
        'scipy',
        'dask>=2022.12.1, <=2023.7.0',
        'distributed>=2022.12.1, <=2023.7.0',
        'networkx',
        'appdirs',
        'rich',
        'jupyter-sphinx',
    ],
    extras_require={
        'nlmixr': ["pyreadr"],
    },
    entry_points={
        'console_scripts': [
            'pharmpy             = pharmpy.__main__:run',
            'psn-pharmpy-wrapper = pharmpy.tools.psn_helpers:pharmpy_wrapper',
        ]
    },
)
