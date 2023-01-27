from pharmpy.internals.parse import AttrToken, AttrTree

from .record import Record

_ws = {' ', '\x00', '\t'}


class ProblemRecord(Record):
    @property
    def title(self):
        return str(self.root.subtree('raw_title'))

    def set_title(self, new_title):
        if new_title and new_title[0] in _ws:
            raise ValueError(
                f'Invalid title "{new_title}". Title cannot start with any of {tuple(map(repr, sorted(_ws)))}.'
            )

        title_tree = AttrTree.create('raw_title', dict(ANYTHING=new_title))

        _, _, after = self.root.partition('raw_title')

        root = AttrTree(
            self.root.rule,
            (
                AttrToken('WS', ' '),
                title_tree,
            )
            + after,
        )
        return ProblemRecord(self.name, self.raw_name, root)
