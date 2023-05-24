import pytest

import pharmpy.model.external.nonmem.nmtran_parser
import pharmpy.model.external.nonmem.records


@pytest.fixture
def records():
    return pharmpy.model.external.nonmem.records


@pytest.fixture
def parser():
    return pharmpy.model.external.nonmem.nmtran_parser.NMTranParser()
