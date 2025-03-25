import os


def is_running_on_slurm() -> bool:
    # Are we running inside a Slurm allocation?
    jobid = os.getenv("SLURM_JOB_ID")
    return jobid is not None


def get_slurm_nodename():
    nodename = os.getenv("SLURMD_NODENAME")
    return nodename


def get_slurm_nodedict() -> dict[str, int]:
    # Dict of node: ncores
    nodelist = get_slurm_nodelist()
    corelist = get_slurm_corelist()
    nodemap = {host: ncores for host, ncores in zip(nodelist, corelist)}
    return nodemap


def get_slurm_nodelist() -> list[str]:
    # Get the SLURM_JOB_NODELIST environment variable and expand into
    # list of hostnames
    s = os.getenv("SLURM_JOB_NODELIST")
    res = _expand_slurm_nodelist(s)
    return res


def get_slurm_corelist() -> list[int]:
    # Get the SLURM_TASKS_PER_NODE environment variable and expand into
    # list of cpus per job with the help of SLURM_CPUS_PER_TASK
    cpus_per_node = os.getenv("SLURM_JOB_CPUS_PER_NODE", default="1")
    cpus_per_node = _expand_tasks_per_node(cpus_per_node)
    return cpus_per_node


def _expand_cpus_per_node(tasks_per_node: str, cpus_per_job: str) -> list[int]:
    tpn_list = _expand_tasks_per_node(tasks_per_node)
    cpus_per_node = [int(cpus_per_job) * e for e in tpn_list]
    return cpus_per_node


def _expand_tasks_per_node(s: str) -> list[int]:
    a = s.split(",")
    res = []
    for e in a:
        res += _expand_tasks_per_node_element(e)
    return res


def _expand_tasks_per_node_element(s: str) -> list[int]:
    # Expand "2(x3)" into [2, 2, 2]
    if "(" in s:
        s = s.replace("(", "").replace(")", "")
        a = s.split("x")
        res = [int(a[0])] * int(a[1])
    else:
        res = [int(s)]
    return res


def _expand_slurm_nodelist(s: str) -> list[str]:
    # Expand the compact nodelist given from Slurm
    # into a list of hostnames
    a = _split_slurm_nodelist(s)
    res = []
    for e in a:
        res += _expand_slurm_node_def(e)
    return res


def _split_slurm_nodelist(value: str) -> list[str]:
    # Split "compute-b24-[1-3,5-9],compute-b25-[1,4,8]"
    # into ["compute-b24-[1-3,5-9]", "compute-b25-[1,4,8]"]
    in_bracket = False
    parts = []
    s = ""
    for ch in value:
        if ch == '[':
            in_bracket = True
        elif ch == ']':
            in_bracket = False
        elif ch == ',' and not in_bracket:
            parts.append(s)
            s = ""
            continue
        s += ch
    parts.append(s)
    return parts


def _expand_slurm_node_def(s: str) -> list[str]:
    # Expand "compute-b24-[1-3,5-9]"
    # into ["compute-b24-1", "compute-b24-2", "compute-b24-3", "compute-b24-5", ...]
    bracket_string = ""
    template_string = ""
    in_bracket = False
    for ch in s:
        if ch == '[':
            in_bracket = True
            template_string += "{value}"
            bracket_string += ch
        elif ch == ']':
            in_bracket = False
            bracket_string += ch
        elif in_bracket:
            bracket_string += ch
        else:
            template_string += ch
    a = _expand_slurm_bracket(bracket_string)
    res = [template_string.format(value=e) for e in a]
    return res


def _expand_slurm_bracket(value: str) -> list[str]:
    # Expand "[1-3,5-9]" into ["1", "2", "3", "5", "6", "7", "8", "9"]
    s = value[1:-1]
    a = s.split(",")
    expansion = []
    for e in a:
        expansion += _expand_slurm_bracket_element(e)
    return expansion


def _expand_slurm_bracket_element(s: str) -> list[str]:
    # Expand "1-3" into ["1", "2", "3"]
    # Put single elements into lists: "1" -> ["1"]
    if "-" in s:
        a = s.split("-")
        res = [str(i) for i in range(int(a[0]), int(a[1]) + 1)]
    else:
        res = [s]
    return res
