# -*- encoding: utf-8 -*-

from pharmpy import generic
from .records.raw_record import RawRecord
from .records.factory import create_record


class SourceIO(generic.SourceResource.SourceIO):
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

    def iter_records(self, index=0):
        it = enumerate(self.iter_problems())
        problem = next(text for i, text in it if i == index)
        (first, _, text) = problem.partition('$')
        yield RawRecord(first)
        yield from iter(create_record(x) for x in text.split('$'))

    def __iter__(self):
        return self.iter_records()


class SourceResource(generic.SourceResource):
    SourceIO = SourceIO
