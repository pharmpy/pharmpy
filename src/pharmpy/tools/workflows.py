import uuid

import networkx as nx
from dask import visualize


class Workflow:
    def __init__(self, tasks=None):
        self.tasks = nx.DiGraph()
        if tasks:
            self.add_tasks(tasks)

    def add_tasks(self, tasks):
        if isinstance(tasks, list):
            self.tasks.add_nodes_from(tasks)
        else:
            self.tasks.add_node(tasks)

    def connect_tasks(self, connect_dict):
        """Connects task with dict: {from: to}"""
        edges = []
        for key, value in connect_dict.items():
            if key == value:
                raise ValueError(f'Cannot connect {key.name} to itself')
            if isinstance(value, list):
                key_edges = [(key, v) for v in value]
                edges.extend(key_edges)
            else:
                edges.append((key, value))
        self.tasks.add_edges_from(edges)

    def merge_workflows(self, other, connect=True, edges=None):
        """Keep all nodes and edges, create edge between self sink and other source
        (assumes 1:M, M:1 or 1:1 connections)"""
        self_leaf_tasks, other_root_tasks = self.get_leaf_tasks(), other.get_root_tasks()

        self.tasks = nx.compose(self.tasks, other.tasks)

        if not connect:
            return

        if edges:
            workflow_connection = edges
        elif len(other_root_tasks) > 1 and len(self_leaf_tasks) > 1:
            raise ValueError('Having N:M connections currently not supported')
        elif len(self_leaf_tasks) > 1:
            workflow_connection = {s: other_root_tasks[0] for s in self_leaf_tasks}
        elif len(other_root_tasks) > 1:
            workflow_connection = {s: self_leaf_tasks[0] for s in other_root_tasks}
        else:
            workflow_connection = {self_leaf_tasks[0]: other_root_tasks[0]}

        self.connect_tasks(workflow_connection)

    def as_dict(self):
        as_dict = dict()
        for task in nx.dfs_tree(self.tasks):
            key = task.task_id
            input_list = [self.id_convert(t) for t in task.task_input]
            value = (task.function, *input_list)
            as_dict[key] = value
        return as_dict

    @staticmethod
    def id_convert(task):
        # TODO: see if better to do recursively
        if isinstance(task, Task):
            return task.task_id
        elif isinstance(task, list):
            return [t.task_id if isinstance(t, Task) else t for t in task]
        else:
            return task

    def get_leaf_tasks(self):
        return [node for node in self.tasks.nodes if self.tasks.out_degree(node) == 0]

    def get_root_tasks(self):
        return [node for node in self.tasks.nodes if self.tasks.in_degree(node) == 0]

    def plot_dask(self, filename):
        visualize(self.as_dict(), filename=filename, collapse_outputs=True)

    # def plot_graph(self, filename):
    #     nx.draw(self.tasks, with_labels=True, font_weight='bold')
    #     plt.savefig(filename)

    def __str__(self):
        return '\n'.join([str(task) for task in self.tasks])


class Task:
    def __init__(self, name, function, *task_input, final_task=False):
        self.name = name
        self.function = function
        self.task_input = task_input
        if not final_task:
            self.task_id = f'{name}-{uuid.uuid1()}'
        else:
            self.task_id = 'results'

    def __repr__(self):
        return self.name

    def __str__(self):
        # input_str = ', '.join([str(type(elem).__name__) for elem in self.task_input])
        return self.name
