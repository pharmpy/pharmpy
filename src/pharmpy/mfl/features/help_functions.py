from collections import defaultdict


def group_args(args, i):
    if i == 0:
        return args

    groups = defaultdict(list)
    for a in args:
        head, tail = a[0:i], a[i:]
        groups[tail].append(head)

    args_new = []
    for tail, heads in groups.items():
        heads_grouped = defaultdict(list)
        for head in heads:
            heads_grouped[head[:-1]].append(head[-1])

        # Heads could not be grouped
        if len(heads_grouped) == len(heads):
            new = tuple(head + tail for head in heads)
            args_new.extend(new)
            continue
        for head, group in heads_grouped.items():
            head_new = []
            if head:
                head_new.append(head[0] if len(head) == 1 else tuple(head))
            head_new.append(group[0] if len(group) == 1 else tuple(group))
            args_new.append(tuple(head_new) + tail)

    return group_args(tuple(args_new), i - 1)


def get_repr(arg):
    if isinstance(arg, tuple) or isinstance(arg, list):
        return f"[{','.join(arg)}]"
    else:
        return arg
