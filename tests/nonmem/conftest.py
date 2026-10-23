# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pytest


@pytest.fixture(scope='session')
def pheno_ext(datadir):
    return datadir / 'pheno_real.ext'


@pytest.fixture(scope='session')
def pheno_phi(datadir):
    return datadir / 'pheno_real.phi'


@pytest.fixture(scope='session')
def pheno_cov(datadir):
    return datadir / 'pheno_real.cov'


@pytest.fixture(scope='session')
def pheno_lst(datadir):
    return datadir / 'pheno_real.lst'


@pytest.fixture(scope='session')
def pheno_data(datadir):
    return datadir / 'pheno.dta'
