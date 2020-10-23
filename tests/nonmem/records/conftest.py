import pytest

import pharmpy.plugins.nonmem.nmtran_parser
import pharmpy.plugins.nonmem.records


@pytest.fixture
def records():
    return pharmpy.plugins.nonmem.records


@pytest.fixture
def parser():
    return pharmpy.plugins.nonmem.nmtran_parser.NMTranParser()
