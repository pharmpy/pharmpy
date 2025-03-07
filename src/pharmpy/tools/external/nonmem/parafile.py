from pathlib import Path

template = """$GENERAL
NODES={nodes} PARSE_TYPE=2 PARSE_NUM=200 TIMEOUTI=600 TIMEOUT=10000 PARAPRINT=0 TRANSFER_TYPE=1

{commands}

$DIRECTORIES
1-{nodes}:NONE
"""


def create_parafile(path: Path, nodes: list[str], ncores: list[int]):
    s = template.format(nodes=len(nodes), commands=_create_commands(nodes, ncores))
    with open(path / "parafile.pnm", "w") as f:
        f.write(s)


def _create_commands(nodes: list[str], ncores: list[int]) -> str:
    s = '$COMMANDS\n1:mpirun -wdir "$PWD" -n 1 ./nonmem $*\n'
    for i, (node, n) in enumerate(zip(nodes, ncores)):
        dirname = '"$PWD"'
        command = f'{i + 2}: -wdir {dirname} -n {n} -host {node} ./nonmem -worker\n'
        s += command
    return s
