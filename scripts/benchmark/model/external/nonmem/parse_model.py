import re
import os
import sys
import time
import traceback

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path, PurePath
from typing import Callable, Iterable, Literal, Optional, Union, cast

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
    condition_number: Optional[float]

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

@contextmanager
def stop_watch(now: Callable[[], float]):
    interval = TimeInterval(now, now())
    try:
        yield interval
    finally:
        interval.end = now()


@dataclass
class TimeInterval:
    now: Callable[[], float]
    begin: float
    end: Optional[float] = None

    def snapshot(self):
        return TimeInterval(
            now = self.now,
            begin=self.begin,
            end=self.now()
        ) if self.end is None else self

    @property
    def elapsed(self):
        if self.end is None:
            return self.snapshot().elapsed
        return timedelta(seconds = self.end - self.begin)


def wall_time():
    return stop_watch(time.perf_counter)

def cpu_time():
    return stop_watch(time.process_time)


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
    wall_time: TimeInterval
    cpu_time: TimeInterval
    reason: FailureReason
    error: Exception

@dataclass(frozen=True)
class Success:
    model_path: Path
    data_path: Optional[Path]
    wall_time: TimeInterval
    cpu_time: TimeInterval
    model: Model
    results: Optional[ModelfitResults]


Results = list[Union[Failure, Success]]

def process(path: Path, convert: bool, ofv: bool, results: bool):
    with open(path, encoding="latin-1") as fp:
        code = fp.read()

    with wall_time() as wt, cpu_time() as ct:

        try:
            cs = NMTranParser().parse(code)
        except Exception as e:
            return Failure(path, None, wt.snapshot(), ct.snapshot(), 'cs', e)

        try:
            di = parse_datainfo(cs, path)
        except Exception as e:
            return Failure(path, None, wt.snapshot(), ct.snapshot(), 'di', e)

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
            return Failure(path, di.path, wt.snapshot(), ct.snapshot(), 'dataset', e)

        model = make_model(cs, di, di, path, data)
        lazy_results = parse_modelfit_results(model, path, with_log=False, lazy=True)
        _results = None if lazy_results is None else ModelfitResults(
            ofv = lazy_results.ofv if ofv else None,
            parameter_estimates = lazy_results.parameter_estimates if results else None,
            individual_estimates = lazy_results.individual_estimates if results else None,
            standard_errors = lazy_results.standard_errors if results else None,
            relative_standard_errors = lazy_results.relative_standard_errors if results else None,
            minimization_successful = lazy_results.minimization_successful if results else None,
            covariance_matrix = lazy_results.covariance_matrix if results else None,
            condition_number=lazy_results.condition_number if results else None,
        )

    return Success(
        path,
        di.path,
        wt,
        ct,
        model,
        _results
    )


def _models(roots: Iterable[PurePath], pattern: re.Pattern[str]):
    for root in roots:
        path = Path(root)
        if path.is_file():
            yield path
        else:
            for dirpath, _, filenames in os.walk(path):
                for filename in filenames:
                    if pattern.search(filename):
                        yield Path(dirpath, filename)

def parse_args(argv: list[str]):
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("roots", type=PurePath, nargs='+')
    parser.add_argument("-p", "--profile", type=PurePath)
    parser.add_argument("-c", "--convert", action='store_true')
    parser.add_argument("-d", "--debug", action='store_true')
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
        key = lambda success: success.wall_time.elapsed,
        reverse = True
    )

    for i, result in enumerate(successes, start=1):
        assert result.data_path is not None
        print(f'{i} {result.model_path} {result.data_path.relative_to(Path.cwd())} {result.wall_time.elapsed} ({result.cpu_time.elapsed})')

    total_wt = sum(map(lambda result: result.wall_time.elapsed, successes), start = timedelta(0))
    total_ct = sum(map(lambda result: result.cpu_time.elapsed, successes), start = timedelta(0))
    print(f'total: {total_wt} ({total_ct})')

    if args.debug:

        failures = cast(
            Iterable[Failure],
            filter(
                lambda result: isinstance(result, Failure),
                results
            )
        )

        for failure in failures:
            debug(f'{failure.model_path} {failure.data_path} {failure.wall_time.elapsed} ({failure.cpu_time.elapsed})')
            traceback.print_exception(failure.error, file=sys.stderr)

if __name__ == "__main__":
    main(sys.argv[1:])
