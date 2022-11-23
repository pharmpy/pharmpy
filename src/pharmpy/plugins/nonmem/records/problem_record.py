from dataclasses import dataclass, replace

from pharmpy.internals.parse import AttrToken, AttrTree

from .record import ReplaceableRecord, replace_tree, with_parsed_and_generated

_ws = {' ', '\x00', '\t'}


@with_parsed_and_generated
@dataclass(frozen=True)
class ProblemRecord(ReplaceableRecord):
    @property
    def title(self):
        return str(self.tree.subtree('raw_title'))

    def replace_title(self, new_title):
        if new_title and new_title[0] in _ws:
            raise ValueError(
                f'Invalid title "{new_title}". Title cannot start with any of {tuple(map(repr, sorted(_ws)))}.'
            )

        title_tree = AttrTree.create('raw_title', dict(ANYTHING=new_title))

        _, _, after = self.tree.partition('raw_title')

        return replace_tree(
            self,
            replace(
                self.tree,
                children=(
                    AttrToken('WS', ' '),
                    title_tree,
                )
                + after,
            ),
        )
