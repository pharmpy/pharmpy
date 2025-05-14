import io
from glob import glob
from os.path import abspath, basename, dirname, join, splitext

from setuptools import find_packages, setup


def read(*names, **kwargs):
    """Read file path relative to setup.py (in encoding; default: utf8)"""
    return io.open(
        join(dirname(abspath(__file__)), *names), encoding=kwargs.get('encoding', 'utf8')
    ).read()


setup(
    name='pharmpy-core',
    version='1.7.1',
    license='GNU Lesser General Public License v3 (LGPLv3)',
    description='Pharmacometric modeling',
    long_description=read('README.rst'),
    long_description_content_type='text/x-rst',
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
    python_requires='>=3.11',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Operating System :: Unix',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering',
    ],
    keywords=[
        'pharmacometrics',
    ],
    install_requires=[
        'lark>=1.1.4',
        'sympy>=1.9',
        'symengine>=0.13.0',
        'pandas>=1.4, !=2.1.0',
        'numexpr>=2.7.0',
        'altair>=5.5.0',
        'itables>=2.2.5',
        'jsonschema>=3.1.0',
        'sphinx>=2.2.0',
        'csscompressor>=0.9.5',
        'beautifulsoup4>=4.13.0',
        'lxml>=4.4.0',
        'numpy>=1.17',
        'scipy>=1.4.0',
        'dask>=2022.12.1',
        'distributed>=2022.12.1',
        'networkx>=2.3',
        'appdirs>=1.4.4',
        'rich>=4.2.0',
        'jupyter-sphinx>=0.2.0',
        'ipykernel>=5.4.0',
        'statsmodels>=0.14.1',
    ],
    extras_require={
        'nlmixr': ["pyreadr>=0.5.3"],
    },
    entry_points={
        'console_scripts': [
            'pharmpy             = pharmpy.__main__:run',
            'psn-pharmpy-wrapper = pharmpy.tools.psn_helpers:pharmpy_wrapper',
        ]
    },
)
