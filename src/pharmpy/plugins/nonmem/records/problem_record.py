from pharmpy.parse_utils import AttrTree, AttrToken
from .record import Record


class ProblemRecord(Record):
    @property
    def title(self):
        max_number_of_characters_in_title = 72
        return str(self.root.raw_title).strip()[:max_number_of_characters_in_title]

    @title.setter
    def title(self, new_title):
        #if new_title != new_title.strip():
        #    raise ValueError(f"Can't set ProblemRecord.title to whitespace-padded {new_title}")
        if len(new_title.strip()) > 72:
            raise ValueError(f"The provided title is {len(new_title.strip())} characters long but can be maximum 72 characters long (not counting initial and trailing whitespace)")
        if new_title == new_title.lstrip():
            new_title = ' ' + new_title
        node = AttrTree.create('raw_title', dict(REST_OF_LINE=new_title))
        self.root.set('raw_title', node)
