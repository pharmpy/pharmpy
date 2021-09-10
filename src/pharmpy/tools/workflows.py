import copy
import uuid

import networkx as nx


class Workflow:
    def __init__(self, tasks=None):
        self._g = nx.DiGraph()
        if tasks:
            for task in tasks:
                self.add_task(task)

    def add_task(self, task, predecessors=None):
        """Add a task to the workflow.

        Predecessors will be connected if given.
        """
        self._g.add_node(task)
        if predecessors is not None:
            if not isinstance(predecessors, list):
                self._g.add_edge(predecessors, task)
            else:
                for pred in predecessors:
                    self._g.add_edge(pred, task)

    def insert_workflow(self, other, predecessors=None):
        if predecessors is None:
            output_tasks = self.output_tasks
        else:
            if isinstance(predecessors, list):
                output_tasks = predecessors
            else:
                output_tasks = [predecessors]
        input_tasks = other.input_tasks
        self._g = nx.compose(self._g, other._g)
        if len(input_tasks) == len(output_tasks):
            for inp, outp in zip(input_tasks, output_tasks):
                self._g.add_edge(outp, inp)
        elif len(input_tasks) == 1:
            for outp in output_tasks:
                self._g.add_edge(outp, input_tasks[0])
        elif len(output_tasks) == 1:
            for inp in input_tasks:
                self._g.add_edge(output_tasks[0], inp)
        else:
            raise ValueError('Having N:M connections between workflows is currently not supported')

    def as_dask_dict(self):
        ids = dict()
        for task in self._g.nodes():
            ids[task] = f'{task.name}-{uuid.uuid4()}'

        if len(self.output_tasks) == 1:
            ids[self.output_tasks[0]] = 'results'
        else:
            raise ValueError("Workflow can only have one output task")

        as_dict = dict()
        for task in nx.dfs_tree(self._g):
            key = ids[task]
            if task.has_input():
                input_list = list(task.task_input)
            else:
                input_list = []
            input_list.extend([ids[t] for t in self._g.predecessors(task)])
            value = (task.function, *input_list)
            as_dict[key] = value
        return as_dict

    @property
    def input_tasks(self):
        return [node for node in self._g.nodes if self._g.in_degree(node) == 0]

    @property
    def output_tasks(self):
        return [node for node in self._g.nodes if self._g.out_degree(node) == 0]

    def get_upstream_tasks(self, task):
        edges = list(nx.edge_dfs(self._g, task, orientation='reverse'))
        return [node for node, _, _ in edges]

    def get_predecessors(self, task):
        return list(self._g.predecessors(task))

    def get_successors(self, task):
        return list(self._g.successors(task))

    def copy(self):
        wf_copy = copy.deepcopy(self)
        return wf_copy

    def plot_dask(self, filename):
        from dask import visualize

        visualize(self.as_dask_dict(), filename=filename, collapse_outputs=True)

    @property
    def tasks(self):
        """All tasks"""
        return list(self._g.nodes())

    def __len__(self):
        return len(self._g.nodes)

    def __str__(self):
        return '\n'.join([str(task) for task in self._g])


class Task:
    def __init__(self, name, function, *task_input):
        self.name = name
        self.function = function
        self.task_input = task_input

    def has_input(self):
        return len(self.task_input) > 0

    def __repr__(self):
        return self.name
