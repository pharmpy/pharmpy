from pharmpy.parse_utils import AttrTree, AttrToken
from .record import Record

class ProblemRecord(Record):
    @property
    def title(self):
        max_number_of_characters_in_title = 72
        return str(self.root.title)[:max_number_of_characters_in_title]

    @title.setter
    def title(self, new_title):
        if new_title != new_title.strip():
            raise ValueError(f"Can't set ProblemRecord.string to whitespace-padded {new_title}")
        node = AttrTree.create('title', dict(REST_OF_LINE=new_title))
        self.root.set('title', node)
        self.root.set('WS_INLINE', AttrToken('WS_INLINE', ' '))
