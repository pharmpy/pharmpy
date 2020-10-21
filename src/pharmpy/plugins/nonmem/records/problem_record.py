from pharmpy.parse_utils import AttrTree

from .record import Record


class ProblemRecord(Record):
    @property
    def title(self):
        first_line = str(self.root.raw_title).split('\n')[0].strip()
        max_number_of_characters_in_title = 72
        return first_line[:max_number_of_characters_in_title]

    @title.setter
    def title(self, new_title):
        if len(new_title.strip()) > 72:
            raise ValueError(
                f'The provided title is {len(new_title.strip())} characters long but '
                f'can be maximum 72 characters long (not counting initial and trailing'
                f'whitespace)'
            )
        if new_title == new_title.lstrip():
            new_title = ' ' + new_title
        lines = str(self.root.raw_title).split('\n')
        lines[0] = new_title
        s = '\n'.join(lines)
        node = AttrTree.create('raw_title', dict(ANYTHING=s))
        self.root.set('raw_title', node)
