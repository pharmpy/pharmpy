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
                if len(head) == 1:
                    head_new.append(head[0])
                else:
                    head_new.extend(head)
            head_new.append(group[0] if len(group) == 1 else tuple(group))
            args_new.append(tuple(head_new) + tail)

    return group_args(tuple(args_new), i - 1)


def get_repr(arg):
    if isinstance(arg, tuple) or isinstance(arg, list):
        arg = [str(a) for a in arg]
        return f"[{','.join(arg)}]"
    else:
        return arg


def format_numbers(numbers, as_range=False):
    if len(numbers) == 1:
        return f'{numbers[0]}'

    numbers_sorted = sorted(numbers)
    if as_range and all(b - a == 1 for a, b in zip(numbers_sorted, numbers_sorted[1:])):
        numbers_formatted = f'{numbers[0]}..{numbers[-1]}'
    else:
        numbers_formatted = f"[{','.join(str(n) for n in numbers_sorted)}]"
    return numbers_formatted
