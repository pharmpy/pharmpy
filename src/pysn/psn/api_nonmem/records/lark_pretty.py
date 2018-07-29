from lark import Tree


def preview(content, maxlen=20):
    cut = len(content) - maxlen
    if cut > 0:
        preview = repr(content[0:maxlen])
        preview = "%s.. +%d chars" % (preview, cut)
    else:
        preview = repr(content)
    return preview

def tree_lines(obj, ind, root=False, verbose=True):
    if isinstance(obj, Tree):
        lines_all = []
        content_all = ''
        for child in obj.children:
            lines_child, content_child = tree_lines(child, ind, root, verbose)
            lines_all += [(ind + '%s') % line for line in lines_child]
            content_all += content_child
        if root or not verbose:
            lines = [ind + '%s' % (obj.data,)]
        else:
            lines = [ind + '%s %s' % (obj.data, preview(content_all))]
        lines += lines_all
        return (lines, content_all)
    else:
        content = str(obj)
        if verbose:
            lines = [ind + '%s %s' % (obj.type, preview(content))]
        else:
            lines = [ind + '%s' % (repr(content),)]
        return (lines, content)

def prettyTree(tree, ind='  ', verbose=True):
    out = []
    lines, content = tree_lines(tree, ind, True, verbose)
    out = ['INPUT LINES:']
    for i, line in enumerate(content.splitlines()):
        out += [ind + '%d: %s' % (i, repr(line))]
    out += ['TREE:']
    out += lines
    return '\n'.join(out)
