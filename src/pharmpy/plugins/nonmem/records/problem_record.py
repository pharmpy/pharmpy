from pharmpy.parse_utils import AttrTree

from .record import Record


class ProblemRecord(Record):
    @property
    def title(self):
        return str(self.root.raw_title).lstrip()

    @title.setter
    def title(self, new_title):
        node = AttrTree.create('raw_title', dict(ANYTHING=" " + new_title))
        self.root.set('raw_title', node)
