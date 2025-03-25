from pathlib import Path

template = """$GENERAL
NODES={nodes} PARSE_TYPE=2 PARSE_NUM=200 TIMEOUTI=600 TIMEOUT=10000 PARAPRINT=0 TRANSFER_TYPE=1

$COMMANDS
{commands}

$DIRECTORIES
{directories}
"""


def create_parafile(path: Path, nodedict: dict[str, int], tmp_path: Path):
    ncores = list(nodedict.values())
    s = template.format(
        nodes=sum(ncores),
        directories=_create_directories(ncores, tmp_path),
        commands=_create_commands(ncores, tmp_path),
    )
    with open(path, "w") as f:
        f.write(s)


def _create_commands(ncores: list[int], tmp_path: Path) -> str:
    s = '1:mpirun -wdir "$PWD" -n 1 ./nonmem $*\n'
    for i, n in enumerate(ncores):
        if i == 0:
            dirname = '"$PWD"'
            n -= 1
        else:
            ref = tmp_path if tmp_path else '"$PWD"'
            dirname = f'"{ref}/worker{i}"'

        command = f'{i + 2}: -wdir {dirname} -n {n} ./nonmem -worker\n'
        s += command
    return s


def _create_directories(ncores: list[int], tmp_path: Path) -> str:
    s = "1:NONE\n2:NONE\n"
    for i in range(3, len(ncores) + 2):
        s += f"{i}:{tmp_path}/worker{i - 2}\n"
    return s
