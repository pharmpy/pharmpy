from pathlib import Path

template = """$GENERAL
NODES={nodes} PARSE_TYPE=2 PARSE_NUM=200 TIMEOUTI=600 TIMEOUT=10000 PARAPRINT=0 TRANSFER_TYPE=1

$COMMANDS
{commands}

$DIRECTORIES
{directories}
"""


def create_parafile(path: Path, nodedict: dict[str, int]):
    nodes = list(nodedict.keys())
    ncores = list(nodedict.values())
    s = template.format(
        nodes=len(nodes),
        directories=_create_directories(nodes),
        commands=_create_commands(nodes, ncores),
    )
    with open(path, "w") as f:
        f.write(s)


def _create_commands(nodes: list[str], ncores: list[int]) -> str:
    s = '1:mpirun -wdir "$PWD" -n 1 ./nonmem $*\n'
    for i, (node, n) in enumerate(zip(nodes, ncores)):
        if i == 0:
            dirname = '"$PWD"'
        else:
            dirname = f'"$PWD/worker{i}"'
        command = f'{i + 2}: -wdir {dirname} -n {n} -host {node} ./nonmem -worker\n'
        s += command
    return s


def _create_directories(nodes: list[str]) -> str:
    s = "1:NONE\n2:NONE\n"
    for i in range(3, len(nodes) + 2):
        s += f"{i}:worker{i - 2}\n"
    return s
