# -*- encoding: utf-8 -*-

from pharmpy import generic
from .records.raw_record import RawRecord
from .records.factory import create_record


class SourceIO(generic.SourceResource.SourceIO):
    _problem_cache = []
    _records_cache = dict()

    def iter_problems(self):
        offset = 0
        first = True
        buffer = self.read()
        while True:
            idx = buffer.find('$PROBLEM', offset+1)
            if first and idx >= 0:
                first = False
                idx = buffer.find('$PROBLEM', idx+1)
            if idx < 0:
                break
            yield buffer[offset:idx]
            offset = idx
        yield buffer[offset:]

    @property
    def problems(self):
        if not self._problem_cache:
            self._problem_cache = list(self.iter_problems())
        return self._problem_cache

    def iter_records(self, index=0):
        problem = next(text for i, text in enumerate(self.problems) if i == index)
        (first, _, text) = problem.partition('$')
        yield RawRecord(first)
        yield from iter(create_record(x) for x in text.split('$'))

    def get_records(self, index=0):
        if index not in self._records_cache:
            self._records_cache[index] = list(self.iter_records(index))
        return self._records_cache[index]

    def __iter__(self):
        return self.iter_records()


class SourceResource(generic.SourceResource):
    SourceIO = SourceIO
