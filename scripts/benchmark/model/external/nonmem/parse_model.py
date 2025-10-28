import re
import sys

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path, PurePath
from typing import Iterable, Literal, Optional, Union, cast

from pharmpy.deps import pandas as pd

from pharmpy.model.datainfo import DataInfo
from pharmpy.model.external.nonmem.advan import _find_observation_column
from pharmpy.model.external.nonmem.model import Model, make_model
from pharmpy.model.external.nonmem.dataset import _idcol
from pharmpy.model.external.nonmem.nmtran_parser import NMTranParser
from pharmpy.model.external.nonmem.parsing import parse_datainfo, parse_dataset
from pharmpy.tools.external.nonmem.results import parse_modelfit_results

@dataclass(frozen=True)
class ModelfitResults:
    ofv: Optional[float]
    parameter_estimates: Optional[pd.Series]
    individual_estimates: Optional[pd.DataFrame]
    standard_errors: Optional[pd.Series]
    relative_standard_errors: Optional[pd.Series]
    minimization_successful: Optional[bool]
    covariance_matrix: Optional[pd.DataFrame]
    correlation_matrix: Optional[pd.DataFrame]

def parse_columns_required(di: DataInfo):
    if (idcol := _idcol(di)) is not None:
        yield idcol

    try:
        yield _find_observation_column(di)
    except ValueError:
        pass

    if "CMT" in di and not di["CMT"].drop:
        yield "CMT"

    for k, v in di.get_dtype_dict().items():
        if v != "str":
            yield k

def debug(*args, **kwargs):
    print(*args, **kwargs, file = sys.stderr)

@dataclass
class TimeInterval:
    begin: datetime
    end: Optional[datetime] = None

    def snapshot(self):
        return TimeInterval(
            begin=self.begin,
            end=datetime.now()
        ) if self.end is None else self

    @property
    def elapsed(self):
        if self.end is None:
            return self.snapshot().elapsed
        return self.end - self.begin


@contextmanager
def elapsed():
    interval = TimeInterval(begin=datetime.now())
    try:
        yield interval
    finally:
        interval.end = datetime.now()

FailureReason = Union[
    Literal['cs'],
    Literal['di'],
    Literal['raw'],
    Literal['dataset'],
]

@dataclass(frozen=True)
class Failure:
    model_path: Path
    data_path: Optional[Path]
    reason: FailureReason
    error: Exception
    duration: Optional[TimeInterval] = None

@dataclass(frozen=True)
class Success:
    model_path: Path
    data_path: Optional[Path]
    duration: TimeInterval
    model: Model
    results: Optional[ModelfitResults]


Results = list[Union[Failure, Success]]

def process(path: Path, convert: bool, ofv: bool, results: bool):
    with open(path, encoding="latin-1") as fp:
        code = fp.read()

    with elapsed() as interval:

        try:
            cs = NMTranParser().parse(code)
        except Exception as e:
            return Failure(path, None, 'cs', e, interval.snapshot())

        try:
            di = parse_datainfo(cs, path)
        except Exception as e:
            return Failure(path, None, 'di', e, interval.snapshot())

        _di = di if convert else di.replace(
            columns = map(lambda column: column.replace(datatype='str'), di)
        )
        parse_columns = tuple(parse_columns_required(_di))
        debug(parse_columns)
        if not convert:
            di = di.replace(
                columns = map(lambda column: column if column.name in parse_columns else column.replace(datatype='str'), di)
            )

        try:
            data = parse_dataset(di, cs, raw=False, parse_columns=parse_columns)
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                return Failure(path, di.path, 'dataset', e, interval.snapshot())
            raise e

        model = make_model(cs, di, di, path, data)
        _results = parse_modelfit_results(model, path)

    return Success(
        path,
        di.path,
        interval,
        model,
        None if _results is None else ModelfitResults(
            ofv = _results.ofv if ofv else None,
            parameter_estimates = _results.parameter_estimates if results else None,
            individual_estimates = _results.individual_estimates if results else None,
            standard_errors = _results.standard_errors if results else None,
            relative_standard_errors = _results.relative_standard_errors if results else None,
            minimization_successful = _results.minimization_successful if results else None,
            covariance_matrix = _results.covariance_matrix if results else None,
            correlation_matrix = _results.correlation_matrix if results else None,
        )
    )


def _models(roots: Iterable[PurePath], pattern: re.Pattern[str]):
    for root in roots:
        path = Path(root)
        if path.is_file():
            yield path
        else:
            for dirpath, _, filenames in path.walk():
                for filename in filenames:
                    if pattern.search(filename):
                        yield Path(dirpath, filename)

def parse_args(argv: list[str]):
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("roots", type=PurePath, nargs='+')
    parser.add_argument("-p", "--profile", type=PurePath)
    parser.add_argument("-c", "--convert", action='store_true')
    parser.add_argument("-o", "--ofv", action='store_true')
    parser.add_argument("-r", "--results", action='store_true')
    parser.add_argument("-f", "--filter", type=re.compile, default=r'\.mod$|\.ctl$')

    return parser.parse_args(argv)


def main(argv: list[str]):

    args = parse_args(argv)

    profdir = None if args.profile is None else Path(args.profile, datetime.now().isoformat())

    results: Results = []

    @contextmanager
    def profile(out: Optional[Path], model: Path):
        if out is None:
            yield
            return

        from pyinstrument import Profiler
        with Profiler(use_timing_thread=True) as profiler:
            yield

        profpath = out.joinpath(model).with_suffix('.html')
        profpath.parent.mkdir(parents=True, exist_ok=True)
        profiler.write_html(profpath)

    for i, model in enumerate(_models(args.roots, args.filter), start=1):
        debug(f'{i} {model}')

        with profile(profdir, model):
            result = process(model, convert=args.convert, ofv=args.ofv, results=args.results)

        results.append(result)

    successes = sorted(
        cast(
            list[Success],
            filter(
                lambda result: isinstance(result, Success),
                results
            )
        ),
        key = lambda success: success.duration.elapsed,
        reverse = True
    )

    failures = list(
        filter(
            lambda result: isinstance(result, Failure),
            results
        )
    )

    for i, result in enumerate(successes, start=1):
        assert result.data_path is not None
        print(f'{i} {result.model_path} {result.data_path.relative_to(Path.cwd())} {result.duration.elapsed}')

    total = sum(map(lambda result: result.duration.elapsed, successes), start = timedelta(0))
    print(f'total: {total}')


if __name__ == "__main__":
    main(sys.argv[1:])
