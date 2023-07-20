from importlib.metadata import version
from typing import Iterable, Iterator, List, Tuple, Union

from lark import Token, Transformer, Tree
from lark.tree import Meta

WS = {' ', '\x00', '\t'}
LF = {'\r', '\n'}


def _tokenize_ignored_characters(s: str, i: int, j: int) -> Iterable[Token]:
    # TODO propagate line/column information
    head = i
    while i < j:
        first = s[head]
        head += 1
        if first in WS:
            while head < j and s[head] in WS:
                head += 1

            yield Token('WS', s[i:head], start_pos=i, end_pos=head)

        elif first == ';':
            while head < j and s[head] not in LF:
                head += 1

            yield Token('COMMENT', s[i:head], start_pos=i, end_pos=head)

        elif first == '\r':
            assert head < j and s[head] == '\n'
            head += 1

            yield Token('NEWLINE', s[i:head], start_pos=i, end_pos=head)

        elif first == '\n':
            yield Token('NEWLINE', s[i:head], start_pos=i, end_pos=head)

        else:
            assert first == '&'
            while head < j and s[head] not in LF:
                head += 1

            yield Token('CONT', s[i:head], start_pos=i, end_pos=head)

        i = head


def _item_range(x: Union[Tree, Token]) -> Tuple[int, int]:
    if isinstance(x, Tree):
        i = x.meta.start_pos
        j = x.meta.end_pos
        if version('lark') == '1.1.6':
            j = _get_new_end_pos(x)
    else:
        i = x.start_pos
        j = x.end_pos

    assert isinstance(i, int)
    assert isinstance(j, int)
    return (i, j)


def _get_new_end_pos(x):
    # FIXME: temporary workaround, see https://github.com/lark-parser/lark/issues/1304
    x = x.children[-1]
    if isinstance(x, Token):
        return x.end_pos
    return _get_new_end_pos(x)


def _interleave_ignored(source: str, it: Iterator[Union[Tree, Token]]):
    x = next(it)
    yield x

    i = _item_range(x)[1]

    x = next(it)
    while True:
        j, k = _item_range(x)

        if i < j:
            yield from _tokenize_ignored_characters(source, i, j)

        yield x

        try:
            x = next(it)
        except StopIteration:
            break

        i = k


def interleave_ignored(source: str, children: List[Union[Tree, Token]]):
    return children if len(children) < 2 else list(_interleave_ignored(source, iter(children)))


class InterleaveIgnored(Transformer):
    def __init__(self, source: str):
        self.source = source

    def __default__(self, data, children, meta):
        new_children = interleave_ignored(self.source, children)
        return Tree(data, new_children, meta)


def with_ignored_tokens(source, tree):
    new_tree = InterleaveIgnored(source).transform(tree)

    if version('lark') == '1.1.6' and new_tree.children:
        new_tree.meta.end_pos = _get_new_end_pos(new_tree)

    final_meta = Meta()
    # TODO propagate line/column information
    final_meta.start_pos = 0
    final_meta.end_pos = len(source)
    final_meta.empty = False

    final_children: List[Union[Tree, Token]] = (
        (
            list(_tokenize_ignored_characters(source, 0, new_tree.meta.start_pos))
            + new_tree.children
            + list(_tokenize_ignored_characters(source, new_tree.meta.end_pos, len(source)))
        )
        if new_tree.children
        else list(_tokenize_ignored_characters(source, 0, len(source)))
    )

    final_data = new_tree.data
    return Tree(final_data, final_children, final_meta)
