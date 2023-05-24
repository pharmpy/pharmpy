from pathlib import Path
from typing import Dict, Hashable, Optional, Union

import pytest


@pytest.fixture(scope='session')
def testdata():
    """Test data (root) folder."""
    return Path(__file__).resolve().parent / 'testdata'


@pytest.fixture(scope='session')
def datadir(testdata):
    return testdata / 'nonmem'


@pytest.fixture(scope='session')
def pheno_path(datadir):
    return datadir / 'pheno_real.mod'


@pytest.fixture(scope='session')
def pheno(load_model_for_test, pheno_path):
    return load_model_for_test(pheno_path)


@pytest.fixture(scope='session')
def load_model_for_test(tmp_path_factory):
    from pharmpy.model import Model

    _cache: Dict[Hashable, Model] = {}

    def _load(given_path: Union[str, Path]) -> Model:
        # TODO Cache based on file contents instead.

        def _parse_model():
            model = Model.parse_model(given_path)
            try:
                model.dataset  # NOTE Force parsing of dataset
            except FileNotFoundError:
                pass  # NOTE The error will resurface later if needed
            return model

        basetemp = tmp_path_factory.getbasetemp().resolve()

        resolved_path = Path(given_path).resolve()

        try:
            # NOTE This skips caching when we are reading from a temporary
            # directory.
            resolved_path.relative_to(basetemp)
            return _parse_model()
        except ValueError:
            # NOTE This is raised when resolved_path is not descendant of
            # basetemp. With Python >= 3.9 we could use is_relative_to instead.
            pass

        from pharmpy.tools.external.nonmem import conf

        key = (str(conf), str(resolved_path))

        if key not in _cache:
            _cache[key] = _parse_model()

        return _cache[key]

    return _load


@pytest.fixture(scope='session')
def load_example_model_for_test():
    from pharmpy.model import Model
    from pharmpy.modeling import load_example_model

    _cache: Dict[Hashable, Model] = {}

    def _load(given_name: str) -> Model:
        def _parse_model():
            return load_example_model(given_name)

        from pharmpy.tools.external.nonmem import conf

        key = (str(conf), given_name)

        if key not in _cache:
            _cache[key] = _parse_model()

        return _cache[key]

    return _load


@pytest.fixture(scope='session')
def create_model_for_test(load_example_model_for_test):
    from pharmpy.model import Model

    def _create(code: str, dataset: Optional[str] = None) -> Model:
        model = Model.parse_model_from_string(code)
        datapath = model.datainfo.path
        if dataset is not None:
            # NOTE This yields a copy of the dataset through Model#copy
            model = model.replace(
                dataset=load_example_model_for_test(dataset).dataset,
                datainfo=model.datainfo.replace(path=datapath),
            )
        return model

    return _create
