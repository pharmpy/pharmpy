import re

from lark import Tree, Visitor

_numbered = re.compile(r'([a-z]+)[\d]+')


class RenameNumbered(Visitor):
    def __default__(self, tree: Tree):
        rule = tree.data
        if match := _numbered.match(rule):
            tree.data = match.group(1)
